import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, Optional, Tuple
import numpy as np
from .dataset import TimingDataset


def collate_batch(batch):
    """Custom collate function for batching samples."""
    frames = torch.stack([item["frame"] for item in batch])
    scalars = torch.stack([item["scalar"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    weights = torch.stack([item["weight"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]

    return {
        "frame": frames,      # [B, seq_len, frame_dim]
        "scalar": scalars,    # [B, scalar_dim]
        "label": labels,      # [B]
        "weight": weights,    # [B]
        "sample_id": sample_ids,
    }


def get_dataloaders(
    dataset_path: str,
    batch_size: int = 32,
    val_batch_size: Optional[int] = None,
    test_batch_size: Optional[int] = None,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    use_weighted_sampler: bool = True,
    frame_seq_len: int = 120,
    frame_dim: int = 7,
    scalar_dim: int = 6,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create train, val, test dataloaders.

    Args:
        dataset_path: Path to parquet files directory
        batch_size: Training batch size
        val_batch_size: Validation batch size (default: 2x batch_size)
        test_batch_size: Test batch size (default: 2x batch_size)
        num_workers: Number of data loading workers
        prefetch_factor: Prefetch batches per worker
        use_weighted_sampler: Use weighted sampler for class balancing
        frame_seq_len: Length of frame sequence
        frame_dim: Number of frame features
        scalar_dim: Number of scalar features
        seed: Random seed

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    if val_batch_size is None:
        val_batch_size = batch_size * 2
    if test_batch_size is None:
        test_batch_size = batch_size * 2

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load datasets
    train_dataset = TimingDataset(
        parquet_path=f"{dataset_path}/train.parquet",
        split="train",
        frame_seq_len=frame_seq_len,
        frame_dim=frame_dim,
        scalar_dim=scalar_dim,
        normalize=True,
    )

    val_dataset = TimingDataset(
        parquet_path=f"{dataset_path}/validation.parquet",
        split="validation",
        frame_seq_len=frame_seq_len,
        frame_dim=frame_dim,
        scalar_dim=scalar_dim,
        normalize=True,
    )

    test_dataset = TimingDataset(
        parquet_path=f"{dataset_path}/test.parquet",
        split="test",
        frame_seq_len=frame_seq_len,
        frame_dim=frame_dim,
        scalar_dim=scalar_dim,
        normalize=True,
    )

    # Create samplers for train set (weighted by class frequency)
    if use_weighted_sampler:
        class_weights = train_dataset.get_class_weights()
        sample_weights = class_weights[train_dataset.labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_batch,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_batch,
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )

    # Val/test loaders (no shuffling)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }
