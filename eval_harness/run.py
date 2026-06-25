"""CLI entrypoint for the dispute eval harness.

Examples:
  python -m eval_harness.run                          # engine mode, seed set
  python -m eval_harness.run --manifest my.jsonl      # engine mode, custom set
  python -m eval_harness.run --mode e2e --threshold 0.95   # real Gemini calls
  python -m eval_harness.run --threshold 0.95         # exit 1 if below 95%

Exit code is non-zero when decision accuracy is below --threshold, so it can gate
a CI job once a real labelled set exists.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from eval_harness.dataset import SEED_MANIFEST, load_manifest
from eval_harness.metrics import compute_metrics
from eval_harness.report import format_report
from eval_harness.runner import run_dataset, run_dataset_e2e


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dispute accuracy eval harness")
    parser.add_argument("--manifest", default=str(SEED_MANIFEST), help="JSONL labelled cases")
    parser.add_argument("--mode", choices=["engine", "e2e"], default="engine",
                        help="engine = labelled observations; e2e = real Gemini over images")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="fail (exit 1) if decision accuracy is below this (0..1)")
    args = parser.parse_args(argv)

    cases = load_manifest(args.manifest)
    if args.mode == "e2e":
        preds = asyncio.run(run_dataset_e2e(cases))
    else:
        preds = run_dataset(cases)

    metrics = compute_metrics(preds)
    print(format_report(metrics, title=f"Dispute Eval Report ({args.mode})"))

    if metrics.decision_accuracy < args.threshold:
        print(f"\nFAIL: decision accuracy {metrics.decision_accuracy:.2%} "
              f"< threshold {args.threshold:.2%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
