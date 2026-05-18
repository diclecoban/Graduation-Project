"""
Production-oriented CORAL + residual target calibration experiment.

This is the compact end-to-end implementation for the thesis defense story:

    1. Train a source-domain neural regressor on log(cycle_life).
    2. Align source and target representations with CORAL.
    3. Use k labeled target cells for residual target-side calibration.
    4. Evaluate on the remaining target cells.

The script is intentionally focused on the best-performing domain-adaptation
variant from docs/domain_adaptation_results.md: CORAL representation alignment
plus residual-mean target calibration at k=20.

Usage:
    python3 3_analysis/coral_target_calibration.py
    python3 3_analysis/coral_target_calibration.py --source hust --target matr --k-target 20
    python3 3_analysis/coral_target_calibration.py --source matr --target hust --features-from data/intermediate/feature_set_top12_shap.txt

Outputs:
    outputs/results_v2_coral_target_calibration/results_detailed.csv
    outputs/results_v2_coral_target_calibration/results_summary.csv
    outputs/results_v2_coral_target_calibration/results_coral_target_calibration.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Install torch with `pip install torch`.") from exc

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
from metrics_utils import compute_metrics  # noqa: E402
from run_experiments import META_COLS, SEEDS  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURES_PATH = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_combined.csv"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_coral_target_calibration"


class CoralRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 32, latent_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.regressor = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        y_scaled_log = self.regressor(z).squeeze(-1)
        return y_scaled_log, z


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def coral_loss(z_source: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """CORAL loss: squared Frobenius distance between latent covariance matrices."""
    if z_source.shape[0] < 2 or z_target.shape[0] < 2:
        return torch.tensor(0.0, device=z_source.device)
    zs = z_source - z_source.mean(dim=0, keepdim=True)
    zt = z_target - z_target.mean(dim=0, keepdim=True)
    cov_s = zs.T @ zs / (z_source.shape[0] - 1)
    cov_t = zt.T @ zt / (z_target.shape[0] - 1)
    return torch.mean((cov_s - cov_t) ** 2)


def load_split(dataset: str, seed: int) -> dict:
    with (SPLITS_DIR / f"{dataset}_{seed}.json").open() as f:
        return json.load(f)


def dataset_window(df: pd.DataFrame, dataset: str, n_cycles: int) -> pd.DataFrame:
    return df[(df["dataset"] == dataset) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()


def train_coral_model(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target_unlabeled: np.ndarray,
    *,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    lambda_coral: float,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
) -> tuple[CoralRegressor, StandardScaler, float, float]:
    set_seed(seed)
    scaler = StandardScaler()
    xs_np = scaler.fit_transform(x_source)
    xt_np = scaler.transform(x_target_unlabeled)
    xs_np = np.clip(np.nan_to_num(xs_np, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    xt_np = np.clip(np.nan_to_num(xt_np, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)

    y_log = np.log(y_source)
    y_log_mean = float(np.mean(y_log))
    y_log_std = float(np.std(y_log))
    if y_log_std < 1e-8:
        y_log_std = 1.0
    y_scaled = (y_log - y_log_mean) / y_log_std

    device = torch.device("cpu")
    model = CoralRegressor(
        n_features=xs_np.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    xs = torch.as_tensor(xs_np, dtype=torch.float32, device=device)
    ys = torch.as_tensor(y_scaled, dtype=torch.float32, device=device)
    xt = torch.as_tensor(xt_np, dtype=torch.float32, device=device)

    for _ in range(epochs):
        model.train()
        pred_s, z_s = model(xs)
        _, z_t = model(xt)
        loss_reg = F.mse_loss(pred_s, ys)
        loss_align = coral_loss(z_s, z_t)
        loss = loss_reg + lambda_coral * loss_align
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, scaler, y_log_mean, y_log_std


def predict_cycles(
    model: CoralRegressor,
    scaler: StandardScaler,
    x: np.ndarray,
    *,
    y_log_mean: float,
    y_log_std: float,
) -> np.ndarray:
    x_scaled = scaler.transform(x)
    x_scaled = np.clip(np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(x_scaled, dtype=torch.float32)
        pred_scaled_log, _ = model(xt)
    pred_log = pred_scaled_log.detach().cpu().numpy() * y_log_std + y_log_mean
    pred_log = np.clip(pred_log, np.log(50.0), np.log(5000.0))
    pred = np.exp(pred_log)
    return np.clip(np.nan_to_num(pred, nan=50.0, posinf=5000.0, neginf=50.0), 50.0, 5000.0)


def evaluate_once(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    source: str,
    target: str,
    n_cycles: int,
    seed: int,
    k_target: int,
    n_repeats: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    lambda_coral: float,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
) -> list[dict]:
    source_df = dataset_window(df, source, n_cycles)
    target_df = dataset_window(df, target, n_cycles)
    split = load_split(source, seed)
    source_train = source_df[source_df["cell_id"].isin(split["train"])].copy()

    x_source = source_train[feature_cols].to_numpy(dtype=float)
    y_source = source_train["cycle_life"].to_numpy(dtype=float)
    x_target = target_df[feature_cols].to_numpy(dtype=float)
    y_target = target_df["cycle_life"].to_numpy(dtype=float)

    model, scaler, y_log_mean, y_log_std = train_coral_model(
        x_source,
        y_source,
        x_target,
        seed=seed,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        lambda_coral=lambda_coral,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )
    y_pred_target = predict_cycles(
        model,
        scaler,
        x_target,
        y_log_mean=y_log_mean,
        y_log_std=y_log_std,
    )

    rows = []
    baseline = compute_metrics(y_target, y_pred_target)
    rows.append({
        "source": source,
        "target": target,
        "experiment": f"{source}_to_{target}",
        "n_cycles": int(n_cycles),
        "seed": int(seed),
        "repeat": -1,
        "k_target": 0,
        "adapter": "none",
        "residual_shift": 0.0,
        "MAE": baseline["MAE"],
        "SMAPE": baseline["SMAPE"],
        "R2": baseline["R2"],
        "n_source_train": int(len(source_train)),
        "n_target_total": int(len(target_df)),
        "n_target_eval": int(len(target_df)),
    })

    rng = np.random.default_rng(seed)
    if k_target >= len(target_df) - 1:
        return rows
    for repeat in range(n_repeats):
        cal_idx = rng.choice(len(target_df), size=k_target, replace=False)
        eval_idx = np.setdiff1d(np.arange(len(target_df)), cal_idx)
        residual_shift = float(np.mean(y_target[cal_idx] - y_pred_target[cal_idx]))
        y_pred_adapted = y_pred_target[eval_idx] + residual_shift
        metrics = compute_metrics(y_target[eval_idx], y_pred_adapted)
        rows.append({
            "source": source,
            "target": target,
            "experiment": f"{source}_to_{target}",
            "n_cycles": int(n_cycles),
            "seed": int(seed),
            "repeat": int(repeat),
            "k_target": int(k_target),
            "adapter": "residual_mean",
            "residual_shift": residual_shift,
            "MAE": metrics["MAE"],
            "SMAPE": metrics["SMAPE"],
            "R2": metrics["R2"],
            "n_source_train": int(len(source_train)),
            "n_target_total": int(len(target_df)),
            "n_target_eval": int(len(eval_idx)),
        })
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows
        .groupby(["experiment", "n_cycles", "adapter", "k_target"], as_index=False)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            SMAPE_mean=("SMAPE", "mean"),
            SMAPE_std=("SMAPE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            residual_shift_mean=("residual_shift", "mean"),
            residual_shift_std=("residual_shift", "std"),
            n_runs=("R2", "count"),
            n_source_train_mean=("n_source_train", "mean"),
            n_target_total_mean=("n_target_total", "mean"),
            n_target_eval_mean=("n_target_eval", "mean"),
        )
        .sort_values(["experiment", "adapter", "k_target"])
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["matr", "hust"], default=None)
    parser.add_argument("--target", choices=["matr", "hust"], default=None)
    parser.add_argument("--features-from", type=Path, default=None)
    parser.add_argument("--n-cycles", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--k-target", type=int, default=20)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-coral", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        print(f"[error] missing {FEATURES_PATH.relative_to(PROJECT_ROOT)}")
        return 1

    if (args.source is None) != (args.target is None):
        print("[error] --source and --target must be provided together.")
        return 1
    if args.source == args.target and args.source is not None:
        print("[error] --source and --target must differ.")
        return 1

    df = pd.read_csv(FEATURES_PATH)
    available = [c for c in df.columns if c not in META_COLS]
    feature_cols = list(available)
    feature_source = f"auto-detected from CSV ({len(feature_cols)} features)"
    if args.features_from is not None:
        if not args.features_from.exists():
            print(f"[error] --features-from not found: {args.features_from}")
            return 1
        feature_cols = [line.strip() for line in args.features_from.read_text().splitlines() if line.strip()]
        unknown = [c for c in feature_cols if c not in available]
        if unknown:
            print(f"[error] unknown feature columns: {unknown}")
            return 1
        feature_source = f"loaded from {args.features_from} ({len(feature_cols)} features)"

    directions = [(args.source, args.target)] if args.source else [("matr", "hust"), ("hust", "matr")]
    all_rows: list[dict] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] feature_set: {feature_source}")
    print(f"[setup] directions: {directions}, k_target={args.k_target}, n_cycles={args.n_cycles}")

    for source, target in directions:
        for seed in args.seeds:
            print(f"[run] {source}->{target} seed={seed}")
            all_rows.extend(evaluate_once(
                df,
                feature_cols,
                source=source,
                target=target,
                n_cycles=args.n_cycles,
                seed=seed,
                k_target=args.k_target,
                n_repeats=args.n_repeats,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lambda_coral=args.lambda_coral,
                hidden_dim=args.hidden_dim,
                latent_dim=args.latent_dim,
                dropout=args.dropout,
            ))

    detailed = pd.DataFrame(all_rows)
    summary = summarize(detailed)
    detailed_path = args.output_dir / "results_detailed.csv"
    summary_path = args.output_dir / "results_summary.csv"
    json_path = args.output_dir / "results_coral_target_calibration.json"
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps({
        "protocol": "coral_target_calibration_v1",
        "feature_set": feature_source,
        "feature_columns": feature_cols,
        "n_cycles": args.n_cycles,
        "k_target": args.k_target,
        "n_repeats": args.n_repeats,
        "epochs": args.epochs,
        "lambda_coral": args.lambda_coral,
        "summary": summary.to_dict(orient="records"),
    }, indent=2))

    print(f"[save] {detailed_path.relative_to(PROJECT_ROOT)}")
    print(f"[save] {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"[save] {json_path.relative_to(PROJECT_ROOT)}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
