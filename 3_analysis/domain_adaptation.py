"""
Lightweight domain-adaptation experiments for SOPv2 battery-life transfer.

This script targets the current critical bottleneck: top-k feature selection
does not repair cross-dataset transfer because the MATR <-> HUST problem is
largely conditional shift. It adds a compact neural representation baseline
with source supervision plus optional feature-distribution alignment:

    source_mlp       train on labeled source only
    coral_mlp        source_mlp + CORAL covariance alignment on target X
    mmd_mlp          source_mlp + RBF-MMD alignment on target X

For each trained predictor, it also evaluates small labeled target-side
residual calibration:

    y_adapted = y_pred + mean(y_target_cal - y_pred_cal)

The design is intentionally modest: full-batch PyTorch MLP, log-cycle target,
source-train scaling only, 5 protocol seeds, and k-target sweeps. The goal is
not to replace the tree baselines, but to test whether feature-level alignment
adds value beyond the already-strong target-side calibration story.

Inputs:
    data/intermediate/features_sop12_combined.csv
    splits/sop_v2/{matr,hust}_{seed}.json

Outputs:
    outputs/results_v2_domain_adaptation/results_detailed.csv
    outputs/results_v2_domain_adaptation/results_summary.csv
    outputs/results_v2_domain_adaptation/results_domain_adaptation.json
    docs/domain_adaptation_results.md

Usage:
    python3 3_analysis/domain_adaptation.py
    python3 3_analysis/domain_adaptation.py --features-from data/intermediate/feature_set_top12_shap.txt
    python3 3_analysis/domain_adaptation.py --methods source_mlp coral_mlp --epochs 600
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
except ImportError as exc:  # pragma: no cover - user-facing optional dependency guard
    raise SystemExit(
        "PyTorch is required for domain_adaptation.py. Install torch or skip this optional analysis."
    ) from exc

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
from metrics_utils import compute_metrics  # noqa: E402
from run_experiments import META_COLS, SEEDS  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURES_PATH = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_combined.csv"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_domain_adaptation"
DEFAULT_DOC_PATH = PROJECT_ROOT / "docs" / "domain_adaptation_results.md"

DEFAULT_METHODS = ["source_mlp", "coral_mlp", "mmd_mlp"]
DEFAULT_KS = [5, 10, 20]
DEFAULT_WINDOWS = [100]
DEFAULT_DIRECTIONS = ["matr_to_hust", "hust_to_matr"]


class EncoderRegressor(nn.Module):
    def __init__(self, n_features: int, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        pred = self.head(z).squeeze(-1)
        return pred, z


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def coral_loss(z_source: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    if z_source.shape[0] < 2 or z_target.shape[0] < 2:
        return torch.tensor(0.0, device=z_source.device)
    zs = z_source - z_source.mean(dim=0, keepdim=True)
    zt = z_target - z_target.mean(dim=0, keepdim=True)
    cs = zs.T @ zs / (z_source.shape[0] - 1)
    ct = zt.T @ zt / (z_target.shape[0] - 1)
    return torch.mean((cs - ct) ** 2)


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x**2).sum(dim=1, keepdim=True)
    y_norm = (y**2).sum(dim=1, keepdim=True).T
    return torch.clamp(x_norm + y_norm - 2.0 * x @ y.T, min=0.0)


def median_bandwidth(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pooled = torch.cat([x, y], dim=0)
    dists = pairwise_sq_dists(pooled, pooled)
    positive = dists[dists > 0]
    if positive.numel() == 0:
        return torch.tensor(1.0, device=x.device)
    return torch.sqrt(torch.median(positive)).detach().clamp(min=1e-3)


def mmd_loss(z_source: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    if z_source.shape[0] < 2 or z_target.shape[0] < 2:
        return torch.tensor(0.0, device=z_source.device)
    sigma = median_bandwidth(z_source, z_target)
    gamma = 1.0 / (2.0 * sigma**2)
    k_ss = torch.exp(-gamma * pairwise_sq_dists(z_source, z_source)).mean()
    k_tt = torch.exp(-gamma * pairwise_sq_dists(z_target, z_target)).mean()
    k_st = torch.exp(-gamma * pairwise_sq_dists(z_source, z_target)).mean()
    return k_ss + k_tt - 2.0 * k_st


def load_split(dataset: str, seed: int) -> dict:
    with (SPLITS_DIR / f"{dataset}_{seed}.json").open() as f:
        return json.load(f)


def domain_window(df: pd.DataFrame, dataset: str, n_cycles: int) -> pd.DataFrame:
    return df[(df["dataset"] == dataset) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()


def fit_domain_model(
    x_source: np.ndarray,
    y_source: np.ndarray,
    x_target: np.ndarray,
    *,
    method: str,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
    lambda_align: float,
) -> tuple[EncoderRegressor, float, float]:
    set_global_seed(seed)
    device = torch.device("cpu")
    model = EncoderRegressor(x_source.shape[1], latent_dim, hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    xs = torch.as_tensor(x_source, dtype=torch.float32, device=device)
    y_log = np.log(y_source)
    y_mean = float(np.mean(y_log))
    y_std = float(np.std(y_log))
    if y_std < 1e-8:
        y_std = 1.0
    ys = torch.as_tensor((y_log - y_mean) / y_std, dtype=torch.float32, device=device)
    xt = torch.as_tensor(x_target, dtype=torch.float32, device=device)

    for _ in range(epochs):
        model.train()
        pred_s, z_s = model(xs)
        _, z_t = model(xt)
        loss_reg = F.mse_loss(pred_s, ys)
        if method == "coral_mlp":
            loss_align = coral_loss(z_s, z_t)
        elif method == "mmd_mlp":
            loss_align = mmd_loss(z_s, z_t)
        else:
            loss_align = torch.tensor(0.0, device=device)
        loss = loss_reg + lambda_align * loss_align
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, y_mean, y_std


def predict_cycles(model: EncoderRegressor, x: np.ndarray, *, y_mean: float, y_std: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32)
        pred_scaled, _ = model(xt)
    pred_log_np = pred_scaled.detach().cpu().numpy() * y_std + y_mean
    # The observed SOPv2 cycle-life range is roughly 133..2024 cycles. A wider
    # clamp prevents tiny-sample neural extrapolation from dominating the
    # comparison while still allowing conservative cross-domain errors.
    pred_log_np = np.clip(pred_log_np, np.log(50.0), np.log(5000.0))
    pred = np.exp(pred_log_np)
    return np.clip(np.nan_to_num(pred, nan=50.0, posinf=5000.0, neginf=50.0), 50.0, 5000.0)


def evaluate_target_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    k_values: list[int],
    n_repeats: int,
    seed: int,
) -> list[dict]:
    rows = []
    n = len(y_true)
    rng = np.random.default_rng(seed)
    for k in k_values:
        if k >= n - 1:
            continue
        for repeat in range(n_repeats):
            cal_idx = rng.choice(n, size=k, replace=False)
            test_idx = np.setdiff1d(np.arange(n), cal_idx)
            residual_shift = float(np.mean(y_true[cal_idx] - y_pred[cal_idx]))
            adapted = y_pred[test_idx] + residual_shift
            metrics = compute_metrics(y_true[test_idx], adapted)
            rows.append({
                "adapter": "residual_mean",
                "k_target": int(k),
                "repeat": int(repeat),
                "residual_shift": residual_shift,
                "MAE": metrics["MAE"],
                "SMAPE": metrics["SMAPE"],
                "R2": metrics["R2"],
                "n_eval": int(len(test_idx)),
            })
    return rows


def evaluate_direction(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    source: str,
    target: str,
    n_cycles: int,
    seed: int,
    methods: list[str],
    k_values: list[int],
    n_repeats: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    latent_dim: int,
    dropout: float,
    lambda_align: float,
) -> list[dict]:
    src = domain_window(df, source, n_cycles)
    tgt = domain_window(df, target, n_cycles)
    split = load_split(source, seed)
    train = src[src["cell_id"].isin(split["train"])].copy()
    if train.empty or tgt.empty:
        return []

    x_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["cycle_life"].to_numpy(dtype=float)
    x_target = tgt[feature_cols].to_numpy(dtype=float)
    y_target = tgt["cycle_life"].to_numpy(dtype=float)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_target_s = scaler.transform(x_target)
    x_train_s = np.clip(np.nan_to_num(x_train_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    x_target_s = np.clip(np.nan_to_num(x_target_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)

    rows = []
    for method in methods:
        model, y_mean, y_std = fit_domain_model(
            x_train_s,
            y_train,
            x_target_s,
            method=method,
            seed=seed,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            lambda_align=lambda_align,
        )
        y_pred = predict_cycles(model, x_target_s, y_mean=y_mean, y_std=y_std)
        baseline = compute_metrics(y_target, y_pred)
        rows.append({
            "source": source,
            "target": target,
            "experiment": f"{source}_to_{target}",
            "n_cycles": int(n_cycles),
            "seed": int(seed),
            "method": method,
            "adapter": "none",
            "k_target": 0,
            "repeat": -1,
            "residual_shift": 0.0,
            "MAE": baseline["MAE"],
            "SMAPE": baseline["SMAPE"],
            "R2": baseline["R2"],
            "n_train": int(len(train)),
            "n_target": int(len(tgt)),
            "n_eval": int(len(tgt)),
        })

        for cal_row in evaluate_target_calibration(
            y_target,
            y_pred,
            k_values=k_values,
            n_repeats=n_repeats,
            seed=seed,
        ):
            rows.append({
                "source": source,
                "target": target,
                "experiment": f"{source}_to_{target}",
                "n_cycles": int(n_cycles),
                "seed": int(seed),
                "method": method,
                "n_train": int(len(train)),
                "n_target": int(len(tgt)),
                **cal_row,
            })
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["experiment", "n_cycles", "method", "adapter", "k_target"]
    summary = (
        rows
        .groupby(group_cols, as_index=False)
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            SMAPE_mean=("SMAPE", "mean"),
            SMAPE_std=("SMAPE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            residual_shift_mean=("residual_shift", "mean"),
            n_runs=("R2", "count"),
            n_train_mean=("n_train", "mean"),
            n_target_mean=("n_target", "mean"),
            n_eval_mean=("n_eval", "mean"),
        )
        .sort_values(["experiment", "n_cycles", "adapter", "k_target", "R2_mean"], ascending=[True, True, True, True, False])
        .reset_index(drop=True)
    )
    return summary


def write_report(summary: pd.DataFrame, feature_source: str, path: Path) -> None:
    lines = [
        "# Domain Adaptation Results",
        "",
        f"Feature set: {feature_source}",
        "",
        "This experiment tests lightweight neural feature alignment against the current critical bottleneck: cross-dataset transfer under conditional shift.",
        "",
        "Methods:",
        "",
        "- `source_mlp`: source-only log-cycle MLP.",
        "- `coral_mlp`: source MLP plus CORAL covariance alignment on target features.",
        "- `mmd_mlp`: source MLP plus RBF-MMD alignment on target features.",
        "- `residual_mean`: small labeled target-side residual calibration applied after each predictor.",
        "",
        "## Best Rows By Direction",
        "",
    ]
    for experiment, part in summary.groupby("experiment"):
        best_no_adapter = part[part["adapter"] == "none"].sort_values("R2_mean", ascending=False).head(1)
        best_adapter = part[part["adapter"] != "none"].sort_values("R2_mean", ascending=False).head(1)
        lines.append(f"### {experiment}")
        lines.append("")
        if not best_no_adapter.empty:
            r = best_no_adapter.iloc[0]
            lines.append(
                f"- Best feature-alignment-only row: `{r['method']}` with R2={r['R2_mean']:.3f}, MAE={r['MAE_mean']:.1f}."
            )
        if not best_adapter.empty:
            r = best_adapter.iloc[0]
            lines.append(
                f"- Best target-calibrated row: `{r['method']} + {r['adapter']}` at k={int(r['k_target'])} with R2={r['R2_mean']:.3f}, MAE={r['MAE_mean']:.1f}."
            )
        lines.append("")
    lines.extend([
        "## Interpretation",
        "",
        "If CORAL/MMD improves only slightly while residual target calibration improves much more, the result strengthens the thesis claim: marginal feature alignment is insufficient under conditional shift, and labeled target-side calibration is the practical repair.",
        "",
        "Full tables:",
        "",
        "- `outputs/results_v2_domain_adaptation/results_detailed.csv`",
        "- `outputs/results_v2_domain_adaptation/results_summary.csv`",
    ])
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-from", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--directions", nargs="+", default=DEFAULT_DIRECTIONS, choices=DEFAULT_DIRECTIONS)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        print(f"[error] missing {FEATURES_PATH.relative_to(PROJECT_ROOT)}")
        return 1
    df = pd.read_csv(FEATURES_PATH)
    available = [c for c in df.columns if c not in META_COLS]
    feature_cols = list(available)
    feature_source = f"auto-detected from CSV ({len(feature_cols)} features)"
    if args.features_from is not None:
        if not args.features_from.exists():
            print(f"[error] --features-from not found: {args.features_from}")
            return 1
        listed = [line.strip() for line in args.features_from.read_text().splitlines() if line.strip()]
        unknown = [c for c in listed if c not in available]
        if unknown:
            print(f"[error] unknown feature columns: {unknown}")
            return 1
        feature_cols = listed
        feature_source = f"loaded from {args.features_from} ({len(feature_cols)} features)"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] features: {feature_source}")
    print(f"[setup] methods: {args.methods}")
    print(f"[setup] output_dir: {args.output_dir}")

    direction_pairs = []
    for direction in args.directions:
        source, _, target = direction.partition("_to_")
        direction_pairs.append((source, target))

    all_rows: list[dict] = []
    for source, target in direction_pairs:
        for n_cycles in args.windows:
            for seed in args.seeds:
                print(f"[run] {source}->{target} N={n_cycles} seed={seed}")
                rows = evaluate_direction(
                    df,
                    feature_cols,
                    source=source,
                    target=target,
                    n_cycles=n_cycles,
                    seed=seed,
                    methods=args.methods,
                    k_values=args.k_values,
                    n_repeats=args.n_repeats,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    hidden_dim=args.hidden_dim,
                    latent_dim=args.latent_dim,
                    dropout=args.dropout,
                    lambda_align=args.lambda_align,
                )
                all_rows.extend(rows)

    detailed = pd.DataFrame(all_rows)
    if detailed.empty:
        print("[error] no rows produced")
        return 1

    detailed_path = args.output_dir / "results_detailed.csv"
    summary_path = args.output_dir / "results_summary.csv"
    json_path = args.output_dir / "results_domain_adaptation.json"
    detailed.to_csv(detailed_path, index=False)
    summary = summarize(detailed)
    summary.to_csv(summary_path, index=False)
    payload = {
        "protocol": "domain_adaptation_v1",
        "feature_set": feature_source,
        "feature_columns": feature_cols,
        "methods": args.methods,
        "directions": args.directions,
        "windows": args.windows,
        "seeds": args.seeds,
        "k_values": args.k_values,
        "n_repeats": args.n_repeats,
        "epochs": args.epochs,
        "lambda_align": args.lambda_align,
        "summary": summary.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2))
    write_report(summary, feature_source, args.doc_path)

    print(f"[save] {detailed_path.relative_to(PROJECT_ROOT)}")
    print(f"[save] {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"[save] {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[save] {args.doc_path.relative_to(PROJECT_ROOT)}")
    print("\n=== DOMAIN ADAPTATION SUMMARY ===")
    display_cols = ["experiment", "method", "adapter", "k_target", "MAE_mean", "R2_mean", "n_runs"]
    print(summary[display_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
