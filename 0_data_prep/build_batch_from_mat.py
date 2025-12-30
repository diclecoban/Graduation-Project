"""
Convert the 2017-05-12 Stanford battery .mat dataset into a pickle file.

The resulting pickle matches the structure expected by the feature
engineering notebooks (same schema as the original `batch1.pkl`).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAT_PATH = (
    PROJECT_ROOT
    / "data-driven-prediction-of-battery-cycle-life-before-capacity-degradation-master"
    / "dataset"
    / "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
)
OUTPUT_PATH = PROJECT_ROOT / "batch1.pkl"


def main() -> None:
    if not MAT_PATH.exists():
        raise SystemExit(f"MAT file not found: {MAT_PATH}")

    print(f"Loading {MAT_PATH.name} ...")
    with h5py.File(MAT_PATH, "r") as f:
        batch = f["batch"]
        num_cells = batch["summary"].shape[0]
        bat_dict: dict[str, dict] = {}

        for i in range(num_cells):
            if i % 10 == 0:
                print(f"  Processing cell {i+1}/{num_cells}")

            cell_key = f"b1c{i}"
            cycle_life = f[batch["cycle_life"][i, 0]][()]
            policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()

            summary_group = f[batch["summary"][i, 0]]
            summary = {
                "IR": np.hstack(summary_group["IR"][0, :].tolist()),
                "QC": np.hstack(summary_group["QCharge"][0, :].tolist()),
                "QD": np.hstack(summary_group["QDischarge"][0, :].tolist()),
                "Tavg": np.hstack(summary_group["Tavg"][0, :].tolist()),
                "Tmin": np.hstack(summary_group["Tmin"][0, :].tolist()),
                "Tmax": np.hstack(summary_group["Tmax"][0, :].tolist()),
                "chargetime": np.hstack(summary_group["chargetime"][0, :].tolist()),
                "cycle": np.hstack(summary_group["cycle"][0, :].tolist()),
            }

            cycles_group = f[batch["cycles"][i, 0]]
            cycle_dict: dict[str, dict] = {}
            for j in range(cycles_group["I"].shape[0]):
                cycle_dict[str(j)] = {
                    "I": np.hstack((f[cycles_group["I"][j, 0]][()])),
                    "Qc": np.hstack((f[cycles_group["Qc"][j, 0]][()])),
                    "Qd": np.hstack((f[cycles_group["Qd"][j, 0]][()])),
                    "Qdlin": np.hstack((f[cycles_group["Qdlin"][j, 0]][()])),
                    "T": np.hstack((f[cycles_group["T"][j, 0]][()])),
                    "Tdlin": np.hstack((f[cycles_group["Tdlin"][j, 0]][()])),
                    "V": np.hstack((f[cycles_group["V"][j, 0]][()])),
                    "dQdV": np.hstack((f[cycles_group["discharge_dQdV"][j, 0]][()])),
                    "t": np.hstack((f[cycles_group["t"][j, 0]][()])),
                }

            bat_dict[cell_key] = {
                "cycle_life": cycle_life,
                "charge_policy": policy,
                "summary": summary,
                "cycles": cycle_dict,
            }

    print(f"Writing pickle to {OUTPUT_PATH}")
    with OUTPUT_PATH.open("wb") as fp:
        pickle.dump(bat_dict, fp)
    print("Done.")


if __name__ == "__main__":
    main()
