"""Pick best window/step config from sweep metrics (CNN mean_subj_auc_smooth)."""
import csv
import sys
from pathlib import Path

CONFIGS = {"w50s5": (50, 5), "w60s5": (60, 5), "w60s10": (60, 10)}
METRICS = Path("results/window_sweep/val/metrics.csv")
RF_CSV = Path("results/window_sweep/val/baseline_rf_metrics.csv")


def norm(name: str) -> str:
    return name.replace("eeg_", "").replace("eeg_ecg_", "")


def last_auc(path: Path, key: str) -> dict[str, float]:
    best: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = norm(row.get("feature_set", ""))
            val = row.get(key) or row.get("mean_subj_auc") or ""
            try:
                best[name] = float(val)
            except ValueError:
                pass
    return best


def main() -> None:
    cnn = last_auc(METRICS, "mean_subj_auc_smooth")
    rf = last_auc(RF_CSV, "mean_subj_auc_smooth") or last_auc(RF_CSV, "mean_subj_auc")
    if not cnn:
        sys.exit("No sweep CNN metrics in results/window_sweep/val/metrics.csv")
    order = {"w60s5": 0, "w50s5": 1, "w60s10": 2}
    winner = sorted(cnn.keys(), key=lambda n: (-cnn[n], order.get(n, 9)))[0]
    w, s = CONFIGS[winner]
    print(f"{winner}|{w}|{s}|{cnn[winner]:.4f}|{rf.get(winner, float('nan')):.4f}")
    for n in sorted(cnn, key=lambda x: -cnn[x]):
        rf_v = rf.get(n, float("nan"))
        print(f"  {n}: CNN smooth={cnn[n]:.4f}  RF smooth={rf_v:.4f}")


if __name__ == "__main__":
    main()
