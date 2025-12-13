"""
CLI entrypoint for jersey number recognition training.
"""

import argparse
from config import Config
from utils import set_seed, ensure_dirs
from trainer import run_training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train jersey number recognition models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase 0 (Basic models):
  basic_r18, basic_effb0, basic_effl0, basic_mv3l, basic_mv3s, basic_sv2
  (Legacy: anchor_r18, anchor_effb0, etc. also supported)

Phase A (Sequence baselines):
  seq_brnn_mp, seq_urnn_fs, seq_bgru_mp, seq_ugru_fs, seq_blstm_mp, seq_ulstm_fs

Phase B (Attention models):
  attn_bgru_bahdanau, attn_bgru_luong, attn_bgru_gate, attn_bgru_hc
  attn_ugru_gate, attn_ugru_hc
  (Legacy: seq_bgru_bahdanau, seq_bgru_luong, etc. also supported)

Legacy (backward compatibility):
  seq, seq_attn, seq_uni, seq_bilstm, anchor, basic
        """
    )
    
    # Extended model type choices
    model_choices = [
        # Phase 0 (new naming)
        "basic_r18", "basic_effb0", "basic_effl0", "basic_mv3l", "basic_mv3s", "basic_sv2",
        # Phase 0 (legacy naming - backward compatibility)
        "anchor_r18", "anchor_effb0", "anchor_effl0", "anchor_mv3l", "anchor_mv3s", "anchor_sv2",
        # Phase A
        "seq_brnn_mp", "seq_urnn_fs", "seq_bgru_mp", "seq_ugru_fs", "seq_blstm_mp", "seq_ulstm_fs",
        # Phase B (new naming)
        "attn_bgru_bahdanau", "attn_bgru_luong", "attn_bgru_gate", "attn_bgru_hc",
        "attn_ugru_gate", "attn_ugru_hc",
        # Phase B (legacy naming - backward compatibility)
        "seq_bgru_bahdanau", "seq_bgru_luong", "seq_bgru_gate", "seq_bgru_hc",
        "seq_ugru_gate", "seq_ugru_hc",
        # Legacy
        "seq", "seq_attn", "seq_uni", "seq_bilstm", "anchor", "basic",
    ]
    
    parser.add_argument("--model_type", type=str, choices=model_choices, required=True,
                       help="Model type identifier (see choices above)")
    parser.add_argument("--backbone", type=str, default=None,
                       help="Backbone for basic model (e.g., resnet18, efficientnet_b0, mobilenet_v3_large). "
                            "Only used with legacy 'anchor' or 'basic' model type.")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_discriminative_lr", action="store_true",
                       help="Enable discriminative learning rates (backbone/temporal/heads)")
    parser.add_argument("--scheduler", type=str, choices=["cosine", "onecycle"], default=None,
                       help="Scheduler type: cosine (default) or onecycle")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    
    if args.data_root is not None:
        config.data_root = args.data_root
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.seed is not None:
        config.seed = args.seed
    if args.backbone is not None:
        config.backbone = args.backbone
    if args.use_discriminative_lr:
        config.use_discriminative_lr = True
    if args.scheduler is not None:
        config.scheduler_type = args.scheduler

    set_seed(config.seed)
    ensure_dirs(config)

    run_training(args.model_type, config, backbone_name=args.backbone)


if __name__ == "__main__":
    main()
