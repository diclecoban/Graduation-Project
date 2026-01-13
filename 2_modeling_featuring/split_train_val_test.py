"""
Split the engineered feature table into train/validation/test partitions.

The script groups rows by ``cell_id`` so that every early-cycle window for
the same battery ends up in the same split, avoiding label leakage.

Usage:
    python 2_modeling_featuring/split_train_val_test.py \
        --source data/intermediate/features_top8_cycles.csv \
        --ratios 0.7 0.15 0.15 \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
SPLITS_DIR = DATA_DIR / "splits"
DEFAULT_SOURCE = INTERMEDIATE_DIR / "features_top8_cycles.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val/test splits from the feature table."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Input CSV to split (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.7, 0.15, 0.15),
        help="Split ratios that sum to 1.0 (default: 0.7 0.15 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the RNG used to shuffle cell IDs.",
    )
    return parser.parse_args()


def normalize_ratios(ratios: Iterable[float]) -> Tuple[float, float, float]:
    ratios = np.asarray(list(ratios), dtype=float)
    if (ratios <= 0).any():
        raise ValueError("All ratios must be positive.")
    total = ratios.sum()
    if not np.isclose(total, 1.0):
        ratios = ratios / total
    return tuple(float(r) for r in ratios)


def allocate_counts(num_cells: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    raw_counts = np.array([ratio * num_cells for ratio in ratios], dtype=float)
    counts = np.floor(raw_counts).astype(int)
    remainder = num_cells - counts.sum()
    if remainder > 0:
        ordering = np.argsort(raw_counts - counts)[::-1]
        for idx in ordering[:remainder]:
            counts[idx] += 1

    # Ensure no split is completely empty if possible.
    for idx in range(len(counts)):
        if counts[idx] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                continue
            counts[idx] += 1
            counts[donor] -= 1

    if counts.sum() != num_cells:
        raise RuntimeError("Failed to allocate cell counts across splits.")
    return tuple(int(c) for c in counts)


def write_split(
    df: pd.DataFrame, cells: set[str], stem: str, split_name: str, output_dir: Path
) -> tuple[Path, int]:
    subset = df[df["cell_id"].isin(cells)].copy()
    out_path = output_dir / f"{stem}_{split_name}.csv"
    subset.to_csv(out_path, index=False)
    return out_path, subset.shape[0]


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"Source CSV not found: {source}")

    ratios = normalize_ratios(args.ratios)
    df = pd.read_csv(source)
    if "cell_id" not in df.columns:
        raise SystemExit("Input CSV must include a 'cell_id' column.")

    cells = np.array(sorted(df["cell_id"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(cells)

    train_count, val_count, test_count = allocate_counts(len(cells), ratios)
    train_cells = set(cells[:train_count])
    val_cells = set(cells[train_count : train_count + val_count])
    test_cells = set(cells[train_count + val_count :])

    stem = source.stem
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("train", train_cells),
        ("val", val_cells),
        ("test", test_cells),
    ]

    for split_name, split_cells in outputs:
        path, row_count = write_split(df, split_cells, stem, split_name, SPLITS_DIR)
        print(f"Wrote {len(split_cells)} cells / {row_count} rows to {path}")

    print(
        f"Split complete: train={len(train_cells)} cells, "
        f"val={len(val_cells)} cells, test={len(test_cells)} cells"
    )


if __name__ == "__main__":
    main()
