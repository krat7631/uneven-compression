"""Command line: run the service on a file of new messages.

    python -m uneven_compression classify examples/new_messages.csv
    python -m uneven_compression classify my.csv --out predictions.csv --risk-margin 0.05

Exit status: 0 on success, 2 on bad input or an unusable setting (the message says which).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .classify import DEFAULT_RISK_MARGIN, classify, intent_names, read_messages, train_vectors, write_predictions
from .ingestion import BANKING77_NUM_LABELS, ValidationError, load_banking77
from .serving import PRECISIONS


def build_service(precision: str, batch_size: int, cache_dir: Path | None):
    """Load the model and fit the intent classifier. Returns (embed, classifier, names, device).

    The classifier is the same logistic regression as every quality number in the
    study, trained on all 9,993 Banking77 training messages embedded at `precision`.
    """
    from .batching import make_batches
    from .evaluation import fit_classifier
    from .serving import embed_batches, load_model

    tok, model, device = load_model(precision)

    def embed(texts):
        return embed_batches(tok, model, device, make_batches(list(texts), batch_size))

    train = load_banking77("train")
    cache = cache_dir / f"banking77_train_{precision}.npy" if cache_dir else None
    x = train_vectors(embed, [r.text for r in train], cache)
    clf = fit_classifier(x, [r.label for r in train])
    if list(clf.classes_) != list(range(BANKING77_NUM_LABELS)):
        raise ValidationError("training data is missing some intents")
    return embed, clf, intent_names(), device


def cmd_classify(args: argparse.Namespace) -> int:
    # every check that needs no model runs first, so bad input fails in a second, not after the model loads
    if not 0 <= args.risk_margin <= 1:
        raise ValueError(f"--risk-margin must be between 0 and 1, got {args.risk_margin}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be at least 1, got {args.batch_size}")
    requests = read_messages(args.input)
    print(f"{len(requests)} messages from {args.input}; loading model at {args.precision} ...", file=sys.stderr)
    embed, clf, names, device = build_service(args.precision, args.batch_size, args.cache_dir)
    proba = clf.predict_proba(embed([r.text for r in requests]))
    results = classify(proba, names, args.risk_margin)

    print(f"{'id':<10} {'intent':<34} {'conf':>5} {'margin':>6}  risk  text")
    for req, res in zip(requests, results):
        text = req.text if len(req.text) <= 48 else req.text[:45] + "..."
        flag = "YES" if res["at_risk"] else ""
        print(f"{req.request_id[:10]:<10} {res['intent'][:34]:<34} {res['confidence']:5.2f} "
              f"{res['margin']:6.3f}  {flag:<4}  {text}")
    n_risk = sum(r["at_risk"] for r in results)
    print(f"\n{n_risk} of {len(results)} messages at risk (margin < {args.risk_margin}) on {device}")
    if args.precision != "fp32":
        print("note: the risk threshold was measured on FP32 margins; at other precisions treat it as a guide",
              file=sys.stderr)
    if args.out:
        write_predictions(args.out, requests, results)
        print(f"wrote {args.out}")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m uneven_compression", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("classify", help="predict the Banking77 intent of each message in a CSV and flag risky ones")
    c.add_argument("input", type=Path, help="CSV with a 'text' column (and optionally 'id')")
    c.add_argument("--out", type=Path, help="also write the predictions to this CSV")
    c.add_argument("--precision", default="fp32", choices=PRECISIONS,
                   help="int8/nf4 need a CUDA GPU (default: fp32)")
    c.add_argument("--risk-margin", type=float, default=DEFAULT_RISK_MARGIN,
                   help=f"flag messages whose margin is below this (default: {DEFAULT_RISK_MARGIN})")
    c.add_argument("--batch-size", type=int, default=32)
    c.add_argument("--cache-dir", type=Path, default=Path("results/cache"),
                   help="where to keep the embedded training set between runs")
    c.set_defaults(func=cmd_classify)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValidationError, ValueError) as e:      # bad input or an unusable setting: say so, no traceback
        print(f"error: {e}", file=sys.stderr)
        return 2
