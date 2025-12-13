"""
Training and validation loops for jersey number recognition.
Handles sequence models (CNN+RNN) and basic single-frame models (CNN-only).

NOTE: Data splits are done at track level in data.py to avoid data leakage
across train/val/test. All sequences from the same track (player) stay in the
same split. See stratified_track_split() in data.py for details.
"""

import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
from typing import Dict, Tuple, List
import math

from models import build_model, multitask_loss
from utils import get_device, Logger, plot_curves
from config import Config


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    model_type: str,
    config: Config,
    scaler: GradScaler = None,
    use_amp: bool = False,
    scheduler: optim.lr_scheduler.LambdaLR = None
) -> Tuple[float, Dict[str, float]]:
    """
    Runs one training epoch.

    For model_type == "seq":
        - batch["frames"]: (B, T, 3, H, W)
        - batch["lengths"]: (B,)
        - labels sequence-level.

    Returns:
        avg_loss: float
        avg_metrics: dict with keys like "acc_number", "acc_tens", "acc_ones".
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    # Accumulate correct/total counts instead of ratios for correct averaging
    correct_counts = {
        "acc_tens": 0,
        "acc_ones": 0,
        "acc_number": 0,
    }
    total_counts = {
        "acc_tens": 0,
        "acc_ones": 0,
        "acc_number": 0,
    }

    for batch in tqdm(loader, desc="Training", unit="batch"):
        optimizer.zero_grad()

        # Determine if this is a basic model (Phase 0) or sequence model (Phase A/B)
        is_basic_model = model_type in ["basic", "anchor"] or model_type.startswith("basic_") or model_type.startswith("anchor_")
        
        # Use mixed precision if enabled
        with autocast(device_type=device.type, enabled=use_amp):
            if is_basic_model:
                # Basic model: single image per sample
                images = batch["image"].to(device)  # (B, 3, H, W)
                tens_label = batch["tens_label"].to(device)  # (B,)
                ones_label = batch["ones_label"].to(device)  # (B,)
                
                outputs = model(images)
                loss, _ = multitask_loss(outputs, tens_label, ones_label, weights=config.loss_weights)
                labels_for_metrics = {
                    "tens": tens_label,
                    "ones": ones_label,
                }
                batch_size = images.shape[0]
            else:
                # Sequence model (Phase A or Phase B): frames + lengths
                frames = batch["frames"].to(device)          # (B, T, 3, H, W)
                lengths = batch["lengths"].to(device)        # (B,)
                tens_label = batch["tens_label"].to(device)  # (B,)
                ones_label = batch["ones_label"].to(device)  # (B,)
                
                outputs = model(frames, lengths)
                # sequence-level logits vs sequence-level labels
                loss, _ = multitask_loss(outputs, tens_label, ones_label, weights=config.loss_weights)
                labels_for_metrics = {
                    "tens": tens_label,
                    "ones": ones_label,
                }
                batch_size = frames.shape[0]

        # Backward pass with mixed precision
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        
        # Step scheduler after optimizer.step()
        # Note: OneCycleLR steps per batch, LambdaLR steps per batch
        if scheduler is not None:
            scheduler.step()

        # Compute correct/total counts directly (not ratios)
        tens_pred = outputs["tens_logits"].argmax(dim=-1)
        ones_pred = outputs["ones_logits"].argmax(dim=-1)
        
        correct_counts["acc_tens"] += (tens_pred == labels_for_metrics["tens"]).sum().item()
        correct_counts["acc_ones"] += (ones_pred == labels_for_metrics["ones"]).sum().item()
        correct_counts["acc_number"] += ((tens_pred == labels_for_metrics["tens"]) & (ones_pred == labels_for_metrics["ones"])).sum().item()
        total_counts["acc_tens"] += labels_for_metrics["tens"].numel()
        total_counts["acc_ones"] += labels_for_metrics["ones"].numel()
        total_counts["acc_number"] += labels_for_metrics["tens"].numel()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    # Compute final metrics from accumulated counts
    avg_metrics = {
        k: correct_counts[k] / max(total_counts[k], 1) 
        for k in correct_counts.keys()
    }

    return avg_loss, avg_metrics


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
    config: Config,
    phase: str = "Validating",
) -> Tuple[float, Dict[str, float]]:
    """
    Similar to train_one_epoch, but:
        - no optimizer step
        - no backprop

    Returns:
        avg_val_loss, avg_val_metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    # Accumulate correct/total counts instead of ratios for correct averaging
    correct_counts = {
        "acc_tens": 0,
        "acc_ones": 0,
        "acc_number": 0,
    }
    total_counts = {
        "acc_tens": 0,
        "acc_ones": 0,
        "acc_number": 0,
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=phase, unit="batch"):
            # Determine if this is a basic model (Phase 0) or sequence model (Phase A/B)
            is_basic_model = model_type in ["basic", "anchor"] or model_type.startswith("basic_") or model_type.startswith("anchor_")
            
            if is_basic_model:
                # Anchor model: single image per sample
                images = batch["image"].to(device)  # (B, 3, H, W)
                tens_label = batch["tens_label"].to(device)  # (B,)
                ones_label = batch["ones_label"].to(device)  # (B,)
                
                outputs = model(images)
                loss, _ = multitask_loss(outputs, tens_label, ones_label, weights=config.loss_weights)
                labels_for_metrics = {
                    "tens": tens_label,
                    "ones": ones_label,
                }
                batch_size = images.shape[0]
            else:
                # Sequence model (Phase A or Phase B): frames + lengths
                frames = batch["frames"].to(device)
                lengths = batch["lengths"].to(device)
                tens_label = batch["tens_label"].to(device)
                ones_label = batch["ones_label"].to(device)
                
                outputs = model(frames, lengths)
                loss, _ = multitask_loss(outputs, tens_label, ones_label, weights=config.loss_weights)
                labels_for_metrics = {
                    "tens": tens_label,
                    "ones": ones_label,
                }
                batch_size = frames.shape[0]

            # Compute correct/total counts directly (not ratios)
            tens_pred = outputs["tens_logits"].argmax(dim=-1)
            ones_pred = outputs["ones_logits"].argmax(dim=-1)
            
            correct_counts["acc_tens"] += (tens_pred == labels_for_metrics["tens"]).sum().item()
            correct_counts["acc_ones"] += (ones_pred == labels_for_metrics["ones"]).sum().item()
            correct_counts["acc_number"] += ((tens_pred == labels_for_metrics["tens"]) & (ones_pred == labels_for_metrics["ones"])).sum().item()
            total_counts["acc_tens"] += labels_for_metrics["tens"].numel()
            total_counts["acc_ones"] += labels_for_metrics["ones"].numel()
            total_counts["acc_number"] += labels_for_metrics["tens"].numel()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    # Compute final metrics from accumulated counts
    avg_metrics = {
        k: correct_counts[k] / max(total_counts[k], 1) 
        for k in correct_counts.keys()
    }

    return avg_loss, avg_metrics


def _get_discriminative_param_groups(model: torch.nn.Module, model_type: str, config: Config) -> List[Dict]:
    """
    Create parameter groups for discriminative learning rates.
    
    For basic models: backbone + heads
    For sequence/attention models: backbone + temporal + heads
    
    Returns:
        List of parameter group dicts for optimizer
    """
    param_groups = []
    
    # Determine if this is a basic model or sequence model
    is_basic_model = model_type in ["basic", "anchor"] or model_type.startswith("basic_") or model_type.startswith("anchor_")
    
    if is_basic_model:
        # Basic models: backbone + heads
        # Backbone parameters
        if hasattr(model, 'backbone'):
            param_groups.append({
                "params": model.backbone.parameters(),
                "lr": config.lr_backbone,
                "name": "backbone"
            })
        
        # Head parameters (fc_tens, fc_ones)
        head_params = []
        if hasattr(model, 'fc_tens'):
            head_params.extend(list(model.fc_tens.parameters()))
        if hasattr(model, 'fc_ones'):
            head_params.extend(list(model.fc_ones.parameters()))
        
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": config.lr_heads,
                "name": "heads"
            })
    else:
        # Sequence/Attention models: encoder (backbone) + temporal (RNN/GRU/LSTM) + heads
        # Backbone (inside encoder)
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'backbone'):
            param_groups.append({
                "params": model.encoder.backbone.parameters(),
                "lr": config.lr_backbone,
                "name": "backbone"
            })
        
        # Temporal layers (RNN, GRU, LSTM)
        # Collect from named_modules to avoid duplicates
        temporal_params = []
        seen_modules = set()
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM)):
                # Avoid adding same module multiple times
                if id(module) not in seen_modules:
                    seen_modules.add(id(module))
                    temporal_params.extend(list(module.parameters()))
        
        # Remove duplicate parameters (in case same parameter appears in multiple modules)
        if temporal_params:
            seen_params = set()
            unique_temporal_params = []
            for p in temporal_params:
                if id(p) not in seen_params:
                    seen_params.add(id(p))
                    unique_temporal_params.append(p)
            
            if unique_temporal_params:
                param_groups.append({
                    "params": unique_temporal_params,
                    "lr": config.lr_temporal,
                    "name": "temporal"
                })
        
        # Attention layers (if any) - use temporal LR
        attention_params = []
        for name, module in model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                attention_params.extend(list(module.parameters()))
        
        if attention_params:
            # Remove duplicates
            seen = set()
            unique_attn_params = []
            for p in attention_params:
                if id(p) not in seen:
                    seen.add(id(p))
                    unique_attn_params.append(p)
            
            if unique_attn_params:
                param_groups.append({
                    "params": unique_attn_params,
                    "lr": config.lr_temporal,  # Attention uses temporal LR
                    "name": "attention"
                })
        
        # Head parameters
        head_params = []
        if hasattr(model, 'fc_tens'):
            head_params.extend(list(model.fc_tens.parameters()))
        if hasattr(model, 'fc_ones'):
            head_params.extend(list(model.fc_ones.parameters()))
        
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": config.lr_heads,
                "name": "heads"
            })
    
    # Fallback: if no groups found, use all parameters with default LR
    if not param_groups:
        print("⚠️  Warning: Could not identify parameter groups, using uniform LR")
        param_groups = [{"params": model.parameters(), "lr": config.learning_rate}]
    
    return param_groups


def run_training(model_type: str, config: Config, backbone_name: str = None) -> Dict[str, list]:
    """
    Full training loop:
        - build data loaders
        - build model, optimizer
        - loop epochs: train + val
        - save best checkpoint
        - save history + curves

    Returns:
        history dict: {
          "train_loss": [...],
          "val_loss": [...],
          "train_acc_number": [...],
          "val_acc_number": [...],
          ...
        }
    """
    from data import build_dataloaders
    
    device = get_device()
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  Warning: CUDA not available, training on CPU (will be slow)")
    
    train_loader, val_loader, test_loader = build_dataloaders(config, model_type=model_type)
    model = build_model(model_type, config, backbone_name=backbone_name).to(device)
    
    # Setup optimizer with discriminative learning rates if enabled
    if config.use_discriminative_lr:
        param_groups = _get_discriminative_param_groups(model, model_type, config)
        optimizer = optim.Adam(
            param_groups,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        print(f"✅ Discriminative LR enabled:")
        print(f"   Backbone: {config.lr_backbone}")
        print(f"   Temporal: {config.lr_temporal}")
        print(f"   Heads: {config.lr_heads}")
    else:
        # Standard optimizer: single LR for all parameters
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
    
    # Learning rate scheduler
    if config.scheduler_type == "onecycle":
        # OneCycleLR: peaks mid-training, good for discriminative LR
        # For discriminative LR, use max_lr per group (OneCycleLR handles this automatically)
        if config.use_discriminative_lr:
            # OneCycleLR will use the max_lr from each parameter group
            max_lr = [group['lr'] for group in optimizer.param_groups]
        else:
            max_lr = config.learning_rate
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=config.max_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # 30% of training for warmup
            anneal_strategy='cos'
        )
        if config.use_discriminative_lr:
            print(f"✅ OneCycleLR scheduler (discriminative): max_lr per group = {max_lr}")
        else:
            print(f"✅ OneCycleLR scheduler: max_lr={max_lr}, epochs={config.max_epochs}")
    else:
        # Cosine annealing with warmup (default)
        total_steps = len(train_loader) * (config.max_epochs - config.warmup_epochs)
        warmup_steps = len(train_loader) * config.warmup_epochs
        
        if config.use_discriminative_lr:
            # For discriminative LR, we need separate schedulers or a custom lambda
            # Using LambdaLR with a function that scales each group's base LR
            def lr_lambda(step):
                if step < warmup_steps:
                    return (step + 1) / warmup_steps if warmup_steps > 0 else 1.0
                else:
                    if total_steps > 0:
                        progress = (step - warmup_steps) / total_steps
                        return 0.5 * (1 + math.cos(math.pi * progress))
                    else:
                        return 1.0
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            # Standard cosine decay with warmup
            def lr_lambda(step):
                if step < warmup_steps:
                    return (step + 1) / warmup_steps if warmup_steps > 0 else 1.0
                else:
                    if total_steps > 0:
                        progress = (step - warmup_steps) / total_steps
                        return 0.5 * (1 + math.cos(math.pi * progress))
                    else:
                        return 1.0
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        print(f"✅ Cosine annealing scheduler: warmup={config.warmup_epochs} epochs")
    
    # Track global step for scheduler
    global_step = 0
    
    # Mixed precision training
    use_amp = config.use_mixed_precision and device.type == "cuda"
    scaler = GradScaler(device=device.type) if use_amp else None
    
    if use_amp:
        print("✅ Mixed precision training (AMP) enabled")
    else:
        print("⚠️  Mixed precision training disabled")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc_tens": [],
        "train_acc_ones": [],
        "train_acc_number": [],
        "val_acc_tens": [],
        "val_acc_ones": [],
        "val_acc_number": [],
    }

    best_val_metric = -float("inf")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    # Use a clear name for model checkpoint
    # For new model types, use the model_type directly; for legacy, construct name
    if model_type.startswith("anchor_"):
        checkpoint_name = f"{model_type}_best.pth"
    elif model_type == "anchor":
        if backbone_name:
            checkpoint_name = f"anchor_{backbone_name}_best.pth"
        else:
            checkpoint_name = "anchor_model_best.pth"
    else:
        # Sequence or attention models
        checkpoint_name = f"{model_type}_best.pth"
    best_checkpoint_path = Path(config.checkpoint_dir) / checkpoint_name

    # Logger
    log_file = Path(config.log_dir) / f"{model_type}_training.log"
    logger = Logger(str(log_file))
    logger.log(f"Starting training for {model_type} model")
    logger.log(f"Device: {device}, Epochs: {config.max_epochs}, Batch size: {config.batch_size}")
    logger.log(f"LR: {config.learning_rate}, Warmup: {config.warmup_epochs} epochs, Grad clip: {config.grad_clip}")
    logger.log(f"Mixed precision: {use_amp}")
    logger.log(f"Early stopping: patience={config.early_stopping_patience} epochs (based on val loss)")

    for epoch in range(config.max_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.max_epochs}")
        print(f"{'='*60}")
        logger.log(f"\n{'='*60}")
        logger.log(f"Epoch {epoch+1}/{config.max_epochs}")
        logger.log(f"{'='*60}")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"Learning rate: {current_lr:.6f}")
            logger.log(f"Learning rate: {current_lr:.6f}")
        
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, model_type, config, 
            scaler=scaler, use_amp=use_amp, scheduler=scheduler
        )
        val_loss, val_metrics = evaluate(model, val_loader, device, model_type, config, phase="Validating")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc_tens"].append(train_metrics["acc_tens"])
        history["train_acc_ones"].append(train_metrics["acc_ones"])
        history["train_acc_number"].append(train_metrics["acc_number"])
        history["val_acc_tens"].append(val_metrics["acc_tens"])
        history["val_acc_ones"].append(val_metrics["acc_ones"])
        history["val_acc_number"].append(val_metrics["acc_number"])

        # Early stopping check (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # print logs
        epoch_summary = (f"Epoch {epoch+1}/{config.max_epochs} | "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | "
              f"Train Acc#: {train_metrics['acc_number']:.4f}, "
              f"Val Acc#: {val_metrics['acc_number']:.4f}")
        if epochs_without_improvement > 0:
            epoch_summary += f" | No improvement: {epochs_without_improvement}/{config.early_stopping_patience}"
        print(epoch_summary)
        logger.log(epoch_summary)

        # Early stopping
        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered!")
            print(f"Validation loss hasn't improved for {config.early_stopping_patience} epochs.")
            print(f"Best validation loss: {best_val_loss:.4f} (at epoch {epoch + 1 - config.early_stopping_patience})")
            print(f"Stopped at epoch {epoch + 1}/{config.max_epochs}")
            print(f"{'='*60}")
            logger.log(f"\nEarly stopping triggered at epoch {epoch + 1}")
            logger.log(f"Validation loss hasn't improved for {config.early_stopping_patience} epochs")
            logger.log(f"Best validation loss: {best_val_loss:.4f}")
            break

        # save best (based on validation accuracy)
        if val_metrics["acc_number"] > best_val_metric:
            best_val_metric = val_metrics["acc_number"]
            checkpoint_data = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "val_acc": val_metrics["acc_number"],
                "config": config.__dict__,
            }
            if scaler is not None:
                checkpoint_data["scaler_state"] = scaler.state_dict()
            torch.save(checkpoint_data, best_checkpoint_path)
            logger.log(f"Saved best checkpoint: {best_checkpoint_path} (Val Acc: {best_val_metric:.4f})")

    # save history as JSON
    hist_path = Path(config.log_dir) / f"history_{model_type}.json"
    with hist_path.open("w") as f:
        json.dump(history, f, indent=2)

    # plot curves
    plot_curves(history, out_prefix=model_type, config=config)

    # optional: load best and eval on test_loader
    print("\n" + "="*60)
    print("Evaluating on test set with best model...")
    print("="*60)
    logger.log("\nEvaluating on test set with best model...")
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_metrics = evaluate(model, test_loader, device, model_type, config, phase="Testing")
    test_summary = (
        f"Test Metrics:\n"
        f"  Loss: {test_loss:.4f}\n"
        f"  Acc Number: {test_metrics['acc_number']:.4f}\n"
        f"  Acc Tens: {test_metrics['acc_tens']:.4f}\n"
        f"  Acc Ones: {test_metrics['acc_ones']:.4f}"
    )
    print(test_summary)
    logger.log(test_summary)

    return history
