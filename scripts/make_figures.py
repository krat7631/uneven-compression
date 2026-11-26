"""Report figures from saved results (any machine).

  fig1_speed_memory   throughput and peak GPU memory vs batch size, per precision
  fig2_where_cost     (a) rare-minus-common F1 gap with 95% interval  (b) flip rate by FP32 margin

Writes figures/<name>.pdf (for the report) and .png (for slides/README).

Usage:  python scripts/make_figures.py --timing 12126225 --quality 12145703
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# One fixed colour and marker per precision in every figure (Okabe-Ito, colour-blind safe);
# the marker carries identity in greyscale print.
STYLE = {"fp32": ("#000000", "o", "FP32"), "fp16": ("#0072B2", "s", "FP16"),
         "int8": ("#D55E00", "^", "INT8"), "nf4": ("#009E73", "D", "NF4")}
INK, MUTED = "#222222", "#777777"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 200,
})


def save(fig, name: str) -> None:
    out = Path("figures")
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    print(f"wrote figures/{name}.pdf/.png")


def fig_speed_memory(timing_run: str) -> None:
    d = pd.read_parquet(f"results/timing/{timing_run}/summary.parquet")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for pr, (c, mk, label) in STYLE.items():
        s = d[d.precision == pr].sort_values("batch_size")
        a.plot(s.batch_size, s.throughput_median, color=c, marker=mk, label=label)
        b.plot(s.batch_size, s.peak_memory_mb, color=c, marker=mk, label=label)
        # direct label at the right end of each memory line
        nudge = {"int8": 5, "fp16": -5}.get(pr, 0)    # INT8 and FP16 end 12 MB apart: separate labels
        b.annotate(label, (s.batch_size.iloc[-1], s.peak_memory_mb.iloc[-1]), xytext=(4, nudge),
                   textcoords="offset points", va="center", fontsize=7, color=INK)
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 32, 128], ["1", "8", "32", "128"])
        ax.set_xlabel("batch size")
    a.set_yscale("log")
    a.set_ylabel("throughput (messages/s, log)")
    a.set_title("(a) Throughput: FP16 fastest; INT8 slowest")
    b.set_ylabel("peak GPU memory (MB)")
    b.set_title("(b) Peak memory: batch size erodes the saving")
    b.set_xlim(right=230)
    a.legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig1_speed_memory")


def fig_where_cost(quality_run: str) -> None:
    r = json.loads(Path(f"results/analysis/{quality_run}.json").read_text())["precisions"]
    prs = [p for p in ("fp16", "int8", "nf4") if p in r]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw={"width_ratios": [1, 1.45]})

    for i, pr in enumerate(prs):
        c, mk, label = STYLE[pr]
        g = r[pr]["rare_minus_common"]
        a.errorbar(g["gap"], i, xerr=[[g["gap"] - g["ci_low"]], [g["ci_high"] - g["gap"]]],
                   fmt=mk, color=c, capsize=3, elinewidth=1.4)
    a.axvline(0, color=MUTED, lw=0.8)
    a.set_yticks(range(len(prs)), [STYLE[p][2] for p in prs])
    a.set_ylim(-0.6, len(prs) - 0.4)
    a.invert_yaxis()
    a.set_xlabel("rare − common F1 drop (points)\n+ = rare intents lose more")
    a.set_title("(a) Rare vs common: no detectable gap")
    a.grid(axis="y", visible=False)

    bands = r[prs[0]]["flips_by_margin"]
    # merge everything >= 0.10 into one band: no precision flipped anything there
    groups = [("< 0.05", [0]), ("0.05–0.10", [1]), ("≥ 0.10", list(range(2, len(bands))))]
    width = 0.8 / len(prs)
    for j, pr in enumerate(prs):
        c, _, label = STYLE[pr]
        fb = r[pr]["flips_by_margin"]
        rates, ns, fs = [], [], []
        for _, idx in groups:
            f, n = sum(fb[k]["flips"] for k in idx), sum(fb[k]["messages"] for k in idx)
            rates.append(100 * f / n)
            fs.append(f)
            ns.append(n)
        xs = [k + (j - (len(prs) - 1) / 2) * width for k in range(len(groups))]
        b.bar(xs, rates, width=width - 0.03, color=c, label=label)
        for x, rate, f in zip(xs, rates, fs):
            b.annotate(str(f), (x, rate), xytext=(0, 2), textcoords="offset points",
                       ha="center", fontsize=6.5, color=INK)
    b.set_xticks(range(len(groups)),
                 [f"{name}\n(n = {sum(bands[k]['messages'] for k in idx):,})" for name, idx in groups])
    b.set_xlabel("FP32 margin of the message (top-1 − top-2 probability)")
    b.set_ylabel("predictions changed (%)")
    b.set_title("(b) Flips happen only near the decision border")
    b.grid(axis="x", visible=False)
    b.legend(frameon=False)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig2_where_cost")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timing", required=True)
    p.add_argument("--quality", required=True)
    args = p.parse_args()
    fig_speed_memory(args.timing)
    fig_where_cost(args.quality)


if __name__ == "__main__":
    main()
