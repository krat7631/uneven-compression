"""Where does the quality cost of compression land? (runs on any machine, no GPU)

Reads results/quality/<run>.json (from score_quality.py) and the run's FP32 vectors.

  Confirmatory  rare-20 vs common-20 intents by training count: mean F1 drop per
                group, rare-minus-common gap with a paired bootstrap interval,
                and McNemar's test on each precision's flips.
  Exploratory   FP32 margin of every test message; flip rate per margin band and
                each precision's reach (largest margin it flipped).

Writes results/analysis/<run>.json.

Usage:  python scripts/analyze_quality.py <run id>      e.g. 12145703
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uneven_compression.analysis import (bootstrap_gap, count_groups, flips_by_margin, group_drops,
                                         margins, mcnemar_exact, reach)
from uneven_compression.evaluation import classification_proba
from uneven_compression.ingestion import BANKING77_NUM_LABELS, load_banking77

GROUP_SIZE = 20                                   # fixed before INT8/NF4 results were seen
MARGIN_EDGES = [0, 0.05, 0.10, 0.20, 0.50, 1.01]  # 1.01 so a margin of exactly 1 is counted


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_id")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    q = json.loads(Path(f"results/quality/{args.run_id}.json").read_text())
    if q["run"]["limit"]:
        raise SystemExit("this run used --limit; analyse a full run")
    n = BANKING77_NUM_LABELS
    y = np.array([r.label for r in load_banking77("test")])
    preds = {pr: np.array(s["predictions"]) for pr, s in q["precisions"].items()}
    base = preds["fp32"]
    others = [pr for pr in preds if pr != "fp32"]

    rare, middle, common = count_groups(q["train_count_per_intent"], GROUP_SIZE)
    counts = np.array(q["train_count_per_intent"])
    groups = {"rare": rare, "middle": middle, "common": common, "all": np.arange(n)}

    # FP32 margins, from the identical classifier; prove it by matching predictions
    vec = Path(f"results/embeddings/{args.run_id}/fp32")
    train_y = [r.label for r in load_banking77("train")]
    proba = classification_proba(np.load(vec / "banking77_train.npy"), train_y,
                                 np.load(vec / "banking77_test.npy"), n)
    assert (proba.argmax(1) == base).all(), "refit classifier disagrees with the scored FP32 predictions"
    m = margins(proba)

    out = {"run_id": args.run_id, "group_size": GROUP_SIZE,
           "group_train_counts": {g: [int(counts[idx].min()), int(counts[idx].max())]
                                  for g, idx in groups.items() if g != "all"},
           "fp32_group_f1": {g: float(np.mean(np.array(q["precisions"]["fp32"]["per_intent_f1"])[idx]))
                             for g, idx in groups.items()},
           "margin_quartiles": [float(v) for v in np.percentile(m, [25, 50, 75])],
           "precisions": {}}
    base_right = base == y
    for pr in others:
        right = preds[pr] == y
        r2w, w2r = int((base_right & ~right).sum()), int((~base_right & right).sum())
        flipped = preds[pr] != base
        out["precisions"][pr] = {
            "group_drop": group_drops(y, base, preds[pr], groups, n),
            "rare_minus_common": bootstrap_gap(y, base, preds[pr], rare, common, n, n_boot=args.n_boot),
            "right_to_wrong": r2w, "wrong_to_right": w2r, "mcnemar_p": mcnemar_exact(r2w, w2r),
            "changed": int(flipped.sum()), "reach": reach(m, flipped),
            "flipped_median_margin": float(np.median(m[flipped])) if flipped.any() else None,
            "flips_by_margin": flips_by_margin(m, flipped, MARGIN_EDGES),
        }

    path = Path("results/analysis") / f"{args.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))

    print(f"groups by training count: {out['group_train_counts']}")
    print("\nCONFIRMATORY  mean F1 drop vs FP32 (points; + = worse)")
    print(f"{'':>5} {'rare':>7} {'middle':>7} {'common':>7} {'all':>7}   rare-common [95% CI]      R->W W->R  McNemar p")
    for pr in others:
        s = out["precisions"][pr]
        d, g = s["group_drop"], s["rare_minus_common"]
        print(f"{pr:>5} {d['rare']:7.2f} {d['middle']:7.2f} {d['common']:7.2f} {d['all']:7.2f}   "
              f"{g['gap']:6.2f} [{g['ci_low']:5.2f}, {g['ci_high']:5.2f}]   {s['right_to_wrong']:4d} "
              f"{s['wrong_to_right']:4d}  {s['mcnemar_p']:9.3f}")
    print("\nEXPLORATORY  flips by FP32 margin (flips / messages)")
    bands = [f"[{r['low']:.2f},{min(r['high'], 1):.2f})" for r in out["precisions"][others[0]]["flips_by_margin"]]
    print(f"{'':>5} " + " ".join(f"{b:>12}" for b in bands) + "   reach")
    for pr in others:
        s = out["precisions"][pr]
        cells = " ".join(f"{r['flips']:>5}/{r['messages']:<6}" for r in s["flips_by_margin"])
        print(f"{pr:>5} {cells}   {s['reach']:.3f}" if s["reach"] is not None else f"{pr:>5} {cells}   -")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
