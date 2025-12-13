"""
Dataset and dataloader builders for jersey number recognition.
Handles sequence indexing, transforms, and batching.

NOTE: Data splits are done at track level to avoid data leakage across train/val/test.
See stratified_track_split() for details.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
import numpy as np
import json
import random
from collections import Counter, defaultdict

from config import Config, get_class_mapping


@dataclass
class SequenceRecord:
    """Record for a single sequence."""
    jersey_str: str                     # e.g., "89"
    track_id: str                       # e.g., "8/track_001" (jersey + track folder)
    sequence_path: Path                 # path to the sequence folder
    full_label: int                     # 0..9
    frame_paths: List[Path]             # ordered list of image paths (anchor + others)
    anchor_idx: int                     # index in frame_paths where anchor is
    anchor_path: Optional[Path] = None  # explicit path to anchor file (if found)


def build_sequence_index(config: Config) -> List[SequenceRecord]:
    """
    Traverses the dataset directory structure and builds a list of SequenceRecord.

    Assumes structure:
        data_root/
          jersey_number/         # e.g., "8", "49", ...
            track_folder/        # Level-2: track folders (one player)
              sequence_folder/   # Level-3: sequence folders
                *.jpg / *.png    # Level-4: frames, one of which is anchor.jpg

    Returns:
        List of SequenceRecord, one per sequence folder. No splitting is done here.
    """
    sequences = []
    str2idx, _ = get_class_mapping(config)
    root_path = Path(config.data_root)
    
    # Walk through jersey number folders (Level 1)
    for jersey_folder in root_path.iterdir():
        if not jersey_folder.is_dir():
            continue
        
        jersey_str = jersey_folder.name
        
        # Skip if not a valid jersey number
        if jersey_str not in str2idx:
            continue
        
        full_label = str2idx[jersey_str]
        
        # Walk through track folders (Level 2)
        for track_folder in jersey_folder.iterdir():
            if not track_folder.is_dir():
                continue
            
            # Derive track_id: combination of jersey + track folder name
            track_id = f"{jersey_str}/{track_folder.name}"
            
            # Walk through sequence folders (Level 3)
            for seq_folder in track_folder.iterdir():
                if not seq_folder.is_dir():
                    continue
                
                # Collect all image files in this sequence folder
                image_files = []
                anchor_path = None
                
                for img_file in seq_folder.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        if 'anchor' in img_file.name.lower():
                            anchor_path = img_file
                        else:
                            image_files.append(img_file)
                
                # Sort frame files by name to preserve temporal order
                image_files.sort(key=lambda x: x.name)
                
                # Handle anchor placement
                frame_paths = []
                anchor_idx = -1
                stored_anchor_path = anchor_path  # Store the actual anchor path
                
                if config.use_anchor_always and anchor_path:
                    # Place anchor at the beginning
                    frame_paths = [anchor_path] + image_files
                    anchor_idx = 0
                elif anchor_path:
                    # Include anchor but don't force position
                    all_files = [anchor_path] + image_files
                    all_files.sort(key=lambda x: x.name)
                    frame_paths = all_files
                    anchor_idx = all_files.index(anchor_path)
                else:
                    # No anchor found, use all frames
                    frame_paths = image_files
                    anchor_idx = 0 if frame_paths else -1
                    stored_anchor_path = frame_paths[0] if frame_paths else None  # Use first frame as fallback
                
                # Skip empty sequences
                if not frame_paths:
                    continue
                
                sequences.append(SequenceRecord(
                    jersey_str=jersey_str,
                    track_id=track_id,
                    sequence_path=seq_folder,
                    full_label=full_label,
                    frame_paths=frame_paths,
                    anchor_idx=anchor_idx,
                    anchor_path=stored_anchor_path
                ))
    
    return sequences


def load_and_preprocess_image(path: Path, config: Config, train: bool) -> torch.Tensor:
    """
    Loads an image from disk, resizes with aspect ratio preserved,
    pads to (config.img_height, config.img_width), and applies augmentations
    if train=True.

    Returns:
        Tensor of shape (3, H, W), dtype=torch.float32, normalized (ImageNet stats).
    """
    # Load and convert to RGB
    img = Image.open(path).convert("RGB")
    
    # Get original dimensions
    orig_w, orig_h = img.size  # PIL: (width, height)
    
    # Compute scale factor to fit within target size
    scale = min(config.img_height / orig_h, config.img_width / orig_w)
    
    # Resize
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Apply augmentations if training
    if train:
        # Color jitter (applied first to work on original colors)
        img_resized = TF.adjust_brightness(img_resized, brightness_factor=np.random.uniform(0.8, 1.2))
        img_resized = TF.adjust_contrast(img_resized, contrast_factor=np.random.uniform(0.8, 1.2))
        img_resized = TF.adjust_saturation(img_resized, saturation_factor=np.random.uniform(0.8, 1.2))
        
        # Random rotation (reduced range from ±10 to ±5 degrees for better digit preservation)
        angle = np.random.uniform(-5, 5)
        img_resized = TF.rotate(img_resized, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        # Motion blur simulation (common in sports video)
        if np.random.random() < 0.3:  # 30% chance
            blur_radius = np.random.choice([1.0, 1.5, 2.0])  # Light to moderate blur
            img_resized = img_resized.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Gaussian noise (sensor/compression artifacts)
        if np.random.random() < 0.2:  # 20% chance
            noise_factor = np.random.uniform(0.01, 0.03)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            noise = np.random.normal(0, noise_factor, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            img_resized = Image.fromarray((img_array * 255).astype(np.uint8))
        
        # Note: Removed horizontal flip - numbers are not horizontally symmetric
        # (e.g., "6" and "9", "89" flipped becomes invalid)
    
    # Convert to tensor (0-1 range)
    img_tensor = TF.to_tensor(img_resized)  # (C, new_h, new_w)
    
    # Create padded tensor
    padded = torch.zeros(3, config.img_height, config.img_width)
    
    # Compute padding offsets (center the image)
    pad_top = (config.img_height - new_h) // 2
    pad_left = (config.img_width - new_w) // 2
    
    # Paste resized image into center
    padded[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_tensor
    
    # Normalize with ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    padded = (padded - mean) / std
    
    return padded


class JerseySequenceDataset(Dataset):
    """
    One item = one sequence (one SequenceRecord).
    """
    
    def __init__(self, records: List[SequenceRecord], config: Config, train: bool):
        self.records = records
        self.config = config
        self.train = train
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
              "frames": Tensor (T, 3, H, W),
              "length": int (T),
              "tens_label": int,
              "ones_label": int,
            }

        Label conventions:
            - tens_label: 0..9 for digit, 10 for "blank" (for single-digit jerseys).
            - ones_label: 0..9 for digit.
        """
        record = self.records[idx]
        jersey_str = record.jersey_str
        frame_paths = record.frame_paths
        
        # Parse jersey_str into tens/ones labels
        if len(jersey_str) == 1:
            tens_label = 10  # blank
            ones_label = int(jersey_str)
        elif len(jersey_str) == 2:
            tens_label = int(jersey_str[0])
            ones_label = int(jersey_str[1])
        else:
            # Edge case
            tens_label = 10
            ones_label = 0
        
        # Decide subset of frames (sample/pad to config.max_seq_len)
        if len(frame_paths) > self.config.max_seq_len:
            # Always include anchor if present
            selected_indices = [0] if self.config.use_anchor_always and len(frame_paths) > 0 else []
            
            # Sample remaining frames uniformly
            other_indices = list(range(1, len(frame_paths))) if self.config.use_anchor_always else list(range(len(frame_paths)))
            if len(other_indices) > 0:
                step = len(other_indices) / (self.config.max_seq_len - len(selected_indices))
                selected_others = [other_indices[int(i * step)] for i in range(self.config.max_seq_len - len(selected_indices))]
                selected_indices.extend(selected_others)
            
            selected_frames = [frame_paths[i] for i in selected_indices]
            seq_len = len(selected_frames)
        else:
            selected_frames = frame_paths
            seq_len = len(selected_frames)
        
        # Load each frame
        frame_tensors = []
        for frame_path in selected_frames:
            try:
                img_tensor = load_and_preprocess_image(frame_path, self.config, self.train)
                frame_tensors.append(img_tensor)
            except Exception as e:
                # If image fails to load, create a black image
                frame_tensors.append(torch.zeros(3, self.config.img_height, self.config.img_width))
        
        # Stack into (T, 3, H, W)
        if frame_tensors:
            frames_tensor = torch.stack(frame_tensors)
        else:
            # Empty sequence fallback
            frames_tensor = torch.zeros(1, 3, self.config.img_height, self.config.img_width)
            seq_len = 1
        
        return {
            "frames": frames_tensor,
            "length": seq_len,
            "tens_label": tens_label,
            "ones_label": ones_label,
        }


class BasicDataset(Dataset):
    """
    Dataset for basic single-frame models.
    Returns only a single frame from each sequence (typically the anchor frame).
    One item = one image (single frame from a sequence).
    
        Returns:
        {
          "image": Tensor (3, H, W),   # anchor RGB image
          "tens_label": int,
          "ones_label": int,
        }
    """
    
    def __init__(self, records: List[SequenceRecord], config: Config, train: bool):
        self.records = records
        self.config = config
        self.train = train
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns only the anchor frame from the sequence.
        Ensures we use the actual anchor file if it exists, otherwise falls back to first frame.
        """
        record = self.records[idx]
        jersey_str = record.jersey_str
        frame_paths = record.frame_paths
        anchor_idx = record.anchor_idx
        anchor_path_explicit = record.anchor_path  # Explicit anchor path stored in record
        
        # Parse jersey_str into tens/ones labels
        if len(jersey_str) == 1:
            tens_label = 10  # blank
            ones_label = int(jersey_str)
        elif len(jersey_str) == 2:
            tens_label = int(jersey_str[0])
            ones_label = int(jersey_str[1])
        else:
            # Edge case
            tens_label = 10
            ones_label = 0
        
        # Priority: use explicit anchor_path if available and valid
        # Otherwise use anchor_idx, otherwise fallback to first frame
        anchor_path = None
        if anchor_path_explicit is not None and anchor_path_explicit.exists():
            # Use the explicitly stored anchor path (most reliable)
            # Verify it's in frame_paths for consistency (should always be true)
            if anchor_path_explicit in frame_paths:
                anchor_path = anchor_path_explicit
            elif anchor_idx >= 0 and anchor_idx < len(frame_paths):
                # Fallback to index if explicit path not in list (shouldn't happen, but safe)
                anchor_path = frame_paths[anchor_idx]
            elif len(frame_paths) > 0:
                anchor_path = frame_paths[0]
        elif anchor_idx >= 0 and anchor_idx < len(frame_paths):
            # Use anchor by index
            anchor_path = frame_paths[anchor_idx]
        elif len(frame_paths) > 0:
            # Fallback to first frame if anchor_idx is invalid
            anchor_path = frame_paths[0]
        else:
            # Empty sequence fallback
            anchor_path = None
        
        # Load anchor image
        if anchor_path is not None:
            try:
                anchor_image = load_and_preprocess_image(anchor_path, self.config, self.train)
            except Exception as e:
                # If image fails to load, create a black image
                anchor_image = torch.zeros(3, self.config.img_height, self.config.img_width)
        else:
            anchor_image = torch.zeros(3, self.config.img_height, self.config.img_width)
        
        return {
            "image": anchor_image,  # (3, H, W)
            "tens_label": tens_label,
            "ones_label": ones_label,
        }


def stratified_track_split(
    records: List[SequenceRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    splits_path: Path,
) -> Tuple[List[SequenceRecord], List[SequenceRecord], List[SequenceRecord]]:
    """
    Split records into train/val/test based on unique (jersey_str, track_id) pairs.

    - All sequences from the same track_id must go to the same split.
    - Splitting is stratified by jersey_str:
      For each jersey, split its track_ids into train/val/test in the given ratios.
    - If splits_path exists, load splits from there instead of recomputing.
    - If not, compute splits, save them to splits_path as JSON, then return.

    Returns:
        train_records, val_records, test_records
    """
    # Check if splits file exists
    if splits_path.exists():
        print(f"Loading splits from {splits_path}...")
        with open(splits_path, 'r') as f:
            splits_data = json.load(f)
        
        train_tracks = set(splits_data["train_tracks"])
        val_tracks = set(splits_data["val_tracks"])
        test_tracks = set(splits_data["test_tracks"])
        
        # Build mapping: track_id -> split_name
        track_to_split = {}
        for track_id in train_tracks:
            track_to_split[track_id] = "train"
        for track_id in val_tracks:
            track_to_split[track_id] = "val"
        for track_id in test_tracks:
            track_to_split[track_id] = "test"
        
        # Filter records into splits
        train_records = [r for r in records if track_to_split.get(r.track_id) == "train"]
        val_records = [r for r in records if track_to_split.get(r.track_id) == "val"]
        test_records = [r for r in records if track_to_split.get(r.track_id) == "test"]
        
        print(f"Loaded splits: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
        return train_records, val_records, test_records
    
    # Compute new splits
    print(f"Computing stratified track-level splits...")
    
    # Group track_ids by jersey_str
    jersey_to_tracks = defaultdict(set)
    track_to_records = defaultdict(list)
    
    for record in records:
        jersey_to_tracks[record.jersey_str].add(record.track_id)
        track_to_records[record.track_id].append(record)
    
    # Initialize global track sets
    train_tracks = set()
    val_tracks = set()
    test_tracks = set()
    
    # Use random with seed for deterministic shuffling
    rng = random.Random(seed)
    
    # Split track_ids for each jersey
    for jersey_str, track_ids_set in sorted(jersey_to_tracks.items()):
        track_ids = sorted(list(track_ids_set))  # Sort for reproducibility
        n = len(track_ids)
        
        if n == 0:
            continue
        
        # Shuffle deterministically
        rng.shuffle(track_ids)
        
        # Compute split sizes
        n_train = max(1, int(n * train_ratio)) if n >= 3 else (1 if n >= 2 else 0)
        n_val = max(1, int(n * val_ratio)) if n >= 3 else (1 if n >= 2 and n_train > 0 else 0)
        n_test = n - n_train - n_val
        
        # Ensure we have at least train + val if possible
        if n >= 2 and n_train == 0:
            n_train = 1
            n_val = 1
            n_test = n - 2
        elif n >= 2 and n_val == 0 and n_test > 0:
            n_val = 1
            n_test = n - n_train - 1
        
        # Assign tracks to splits
        jersey_train = set(track_ids[:n_train])
        jersey_val = set(track_ids[n_train:n_train + n_val])
        jersey_test = set(track_ids[n_train + n_val:])
        
        train_tracks.update(jersey_train)
        val_tracks.update(jersey_val)
        test_tracks.update(jersey_test)
    
    # Verify no overlaps
    assert train_tracks.isdisjoint(val_tracks), "Train and Val tracks overlap!"
    assert train_tracks.isdisjoint(test_tracks), "Train and Test tracks overlap!"
    assert val_tracks.isdisjoint(test_tracks), "Val and Test tracks overlap!"
    
    # Save splits to disk
    splits_data = {
        "train_tracks": sorted(list(train_tracks)),
        "val_tracks": sorted(list(val_tracks)),
        "test_tracks": sorted(list(test_tracks)),
    }
    
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, 'w') as f:
        json.dump(splits_data, f, indent=2)
    
    print(f"Saved splits to {splits_path}")
    
    # Convert tracks to records
    train_records = [r for r in records if r.track_id in train_tracks]
    val_records = [r for r in records if r.track_id in val_tracks]
    test_records = [r for r in records if r.track_id in test_tracks]
    
    # Log distribution statistics
    print("\nSplit distribution by jersey:")
    print(f"{'Jersey':<10} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8} {'Train%':<8} {'Val%':<8} {'Test%':<8}")
    print("-" * 70)
    
    jersey_counter_all = Counter([r.jersey_str for r in records])
    jersey_counter_train = Counter([r.jersey_str for r in train_records])
    jersey_counter_val = Counter([r.jersey_str for r in val_records])
    jersey_counter_test = Counter([r.jersey_str for r in test_records])
    
    for jersey in sorted(jersey_counter_all.keys()):
        total = jersey_counter_all[jersey]
        train_count = jersey_counter_train.get(jersey, 0)
        val_count = jersey_counter_val.get(jersey, 0)
        test_count = jersey_counter_test.get(jersey, 0)
        
        train_pct = (train_count / total * 100) if total > 0 else 0
        val_pct = (val_count / total * 100) if total > 0 else 0
        test_pct = (test_count / total * 100) if total > 0 else 0
        
        print(f"{jersey:<10} {total:<8} {train_count:<8} {val_count:<8} {test_count:<8} "
              f"{train_pct:<8.1f} {val_pct:<8.1f} {test_pct:<8.1f}")
    
    # Track-level statistics
    train_track_counts = Counter([r.track_id for r in train_records])
    val_track_counts = Counter([r.track_id for r in val_records])
    test_track_counts = Counter([r.track_id for r in test_records])
    
    print(f"\nTrack-level statistics:")
    print(f"  Total unique tracks: {len(set(r.track_id for r in records))}")
    print(f"  Train tracks: {len(train_track_counts)}")
    print(f"  Val tracks: {len(val_track_counts)}")
    print(f"  Test tracks: {len(test_track_counts)}")
    print(f"  Train sequences: {len(train_records)}")
    print(f"  Val sequences: {len(val_records)}")
    print(f"  Test sequences: {len(test_records)}")
    
    return train_records, val_records, test_records


def sequence_collate_fn(batch: List[Dict[str, Any]], max_seq_len: int = 16) -> Dict[str, torch.Tensor]:
    """
    Pads batch of variable-length sequences to same length.

    Input batch: list of dicts from JerseySequenceDataset.__getitem__.

    Returns dict with:
        "frames": Tensor (B, T_max, 3, H, W)
        "lengths": Tensor (B,)  # actual lengths before padding
        "tens_label": Tensor (B,)
        "ones_label": Tensor (B,)
        "full_label": Tensor (B,)

    Padding strategy:
        - Let T_max be min(max(lengths_in_batch), max_seq_len).
        - For sequences shorter than T_max: pad by repeating last frame
          until length == T_max.
    """
    lengths = [item["length"] for item in batch]
    max_len = min(max(lengths) if lengths else 1, max_seq_len)  # Cap at max_seq_len
    
    # Get dimensions from first item
    # frames shape is (T, C, H, W)
    T_first, C, H, W = batch[0]["frames"].shape
    
    B = len(batch)
    
    # Initialize batch tensors
    frames_batch = torch.zeros(B, max_len, C, H, W)
    lengths_tensor = torch.zeros(B, dtype=torch.long)
    tens_labels = torch.zeros(B, dtype=torch.long)
    ones_labels = torch.zeros(B, dtype=torch.long)
    
    # Fill batch
    for i, item in enumerate(batch):
        seq_len = min(item["length"], max_len)
        frames_seq = item["frames"][:seq_len]
        
        # Pad by repeating last frame
        if seq_len < max_len:
            last_frame = frames_seq[-1:].repeat(max_len - seq_len, 1, 1, 1)
            frames_seq = torch.cat([frames_seq, last_frame], dim=0)
        
        frames_batch[i] = frames_seq
        lengths_tensor[i] = seq_len
        tens_labels[i] = item["tens_label"]
        ones_labels[i] = item["ones_label"]
    
    return {
        "frames": frames_batch,
        "lengths": lengths_tensor,
        "tens_label": tens_labels,
        "ones_label": ones_labels,
    }


def basic_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for BasicDataset.
    
    Input batch: list of dicts from AnchorDataset.__getitem__.
    
    Returns dict with:
        "image": Tensor (B, 3, H, W)
        "tens_label": Tensor (B,)
        "ones_label": Tensor (B,)
    """
    B = len(batch)
    C, H, W = batch[0]["image"].shape
    
    images = torch.zeros(B, C, H, W)
    tens_labels = torch.zeros(B, dtype=torch.long)
    ones_labels = torch.zeros(B, dtype=torch.long)
    
    for i, item in enumerate(batch):
        images[i] = item["image"]
        tens_labels[i] = item["tens_label"]
        ones_labels[i] = item["ones_label"]
    
    return {
        "image": images,
        "tens_label": tens_labels,
        "ones_label": ones_labels,
    }


def verify_anchor_usage(records: List[SequenceRecord], sample_size: int = 100) -> Dict[str, Any]:
    """
    Verifies that anchor frames are being identified and used correctly.
    
    Args:
        records: List of SequenceRecord to verify
        sample_size: Number of records to sample for verification (0 = all)
    
    Returns:
        Dictionary with verification statistics
    """
    import random
    
    stats = {
        "total_sequences": len(records),
        "sequences_with_explicit_anchor": 0,
        "sequences_with_anchor_in_name": 0,
        "sequences_using_first_frame": 0,
        "anchor_at_index_0": 0,
        "anchor_at_other_index": 0,
        "no_anchor_found": 0,
        "sample_checked": 0,
    }
    
    # Sample records for verification
    records_to_check = records if sample_size == 0 else random.sample(records, min(sample_size, len(records)))
    
    for record in records_to_check:
        stats["sample_checked"] += 1
        
        # Check if explicit anchor path exists
        if record.anchor_path is not None:
            stats["sequences_with_explicit_anchor"] += 1
            # Check if anchor file name contains 'anchor'
            if 'anchor' in record.anchor_path.name.lower():
                stats["sequences_with_anchor_in_name"] += 1
        
        # Check anchor index
        if record.anchor_idx == 0:
            stats["anchor_at_index_0"] += 1
        elif record.anchor_idx > 0:
            stats["anchor_at_other_index"] += 1
        else:
            stats["no_anchor_found"] += 1
            # If no anchor found, check if we're using first frame
            if len(record.frame_paths) > 0:
                stats["sequences_using_first_frame"] += 1
    
    return stats


def build_dataloaders(config: Config, model_type: str = "seq"):
    """
    Builds train/val/test dataloaders from the SequenceRecord index.

    Strategy:
        - build_sequence_index(config) -> all_records (no splitting)
        - stratified_track_split() -> splits records by track_id (all sequences
          from same track stay together), stratified by jersey number
        - Create dataset for each split (JerseySequenceDataset or BasicDataset)
        - Wrap with DataLoader using appropriate collate_fn

    Args:
        config: Config object
        model_type: "basic" or "anchor" (legacy) uses BasicDataset, otherwise uses JerseySequenceDataset

    Returns:
        train_loader, val_loader, test_loader
    """
    # Build sequence index (no splitting yet)
    print(f"Building sequence index from {config.data_root}...")
    all_records = build_sequence_index(config)
    print(f"Found {len(all_records)} sequences")
    
    # Verify anchor usage if using basic model
    is_basic_model = model_type in ["basic", "anchor"] or model_type.startswith("basic_") or model_type.startswith("anchor_")
    if is_basic_model:
        print("\nVerifying anchor frame usage...")
        anchor_stats = verify_anchor_usage(all_records, sample_size=min(500, len(all_records)))
        print(f"  Total sequences: {anchor_stats['total_sequences']}")
        print(f"  Sequences with explicit anchor path: {anchor_stats['sequences_with_explicit_anchor']}")
        print(f"  Anchor files with 'anchor' in name: {anchor_stats['sequences_with_anchor_in_name']}")
        print(f"  Anchors at index 0: {anchor_stats['anchor_at_index_0']}")
        print(f"  Anchors at other indices: {anchor_stats['anchor_at_other_index']}")
        print(f"  No anchor found (using first frame): {anchor_stats['no_anchor_found']}")
        print(f"  Sample checked: {anchor_stats['sample_checked']}")
    
    # Perform stratified track-level split
    splits_path = Path(config.output_dir) / "splits.json"
    train_records, val_records, test_records = stratified_track_split(
        all_records,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        splits_path=splits_path,
    )
    
    # Create datasets based on model_type
    is_basic_model = model_type in ["basic", "anchor"] or model_type.startswith("basic_") or model_type.startswith("anchor_")
    if is_basic_model:
        train_dataset = BasicDataset(train_records, config, train=True)
        val_dataset = BasicDataset(val_records, config, train=False)
        test_dataset = BasicDataset(test_records, config, train=False)
        collate_fn = basic_collate_fn
    else:
        train_dataset = JerseySequenceDataset(train_records, config, train=True)
        val_dataset = JerseySequenceDataset(val_records, config, train=False)
        test_dataset = JerseySequenceDataset(test_records, config, train=False)
        from functools import partial
        collate_fn = partial(sequence_collate_fn, max_seq_len=config.max_seq_len)
    
    # Create dataloaders
    # NOTE: shuffle=True only for train_loader to maintain deterministic val/test evaluation
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
