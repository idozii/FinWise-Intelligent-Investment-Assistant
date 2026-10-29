"""M5-v2: separately locked Tiingo raw-close external asset evaluation.

Yahoo-v1 stays frozen and unchanged. This new protocol shares fixed design
parameters but has a distinct provider, source hashes and experiment namespace.
Run preflight before evaluation. Explicit human authorization is mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research.frozen_eval import _assets_exact, evaluate, preflight
from scripts.m5_download_tiingo import (
    END_DATE, PROVIDER, RESERVED_ASSETS, START_DATE, verify_snapshot,
)

SPEC_PATH = Path(__file__).parent / "protocols" / "m5_tiingo_external_assets_v2.json"
LOCKED_V2_SHA256 = "7c844332597962da3fa1fb68eba776279d695a19fe991cd545934fd3347f274a"


def load_v2_protocol(path: Path = SPEC_PATH) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LOCKED_V2_SHA256:
        raise ValueError("M5-v2 protocol fingerprint mismatch; no silent amendments")
    spec = json.loads(raw)
    if spec["protocol"] != "finwise-m5-tiingo-external-assets-v2":
        raise ValueError("Wrong M5-v2 protocol ID")
    if spec["status"] != "FROZEN_PROVIDER_AMENDMENT_NOT_YET_EVALUATED":
        raise ValueError("Unexpected protocol status")
    if tuple(sorted(spec["reserved_external_assets"])) != tuple(sorted(RESERVED_ASSETS)):
        raise ValueError("Reserved assets changed")
    if spec["dataset"]["source_field"] != "close (raw, not adjClose)":
        raise ValueError("M5-v2 requires raw Close, never adjusted close")
    if not spec["dataset"]["vendor"].startswith("Tiingo End-of-Day Composite"):
        raise ValueError("M5-v2 source vendor changed")
    if spec["dataset"]["download_period"] != (
        f"{START_DATE} through {END_DATE} inclusive, fixed absolute dates"
    ):
        raise ValueError("Downloader and frozen protocol date range differ")
    return spec


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "evaluate"))
    parser.add_argument("--assets", nargs="+", required=True, metavar="ASSET=CSV")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/backtest/m5_tiingo_v2"))
    parser.add_argument("--authorize-outcome-review", action="store_true")
    parser.add_argument("--attest-unused-assets", action="store_true")
    args = parser.parse_args(argv)
    spec = load_v2_protocol()
    pairs = _assets_exact(spec, args.assets)
    for asset, file in pairs:
        verify_snapshot(asset, file)

    if args.stage == "preflight":
        report = preflight(
            spec, pairs, args.output_dir, protocol_sha256=LOCKED_V2_SHA256
        )
        print("M5-v2 Tiingo preflight: PASS (date-only, no model outcome review)")
        for row in report["sources"]:
            print(f"{row['asset']}: {row['n_daily_records']} observations; "
                  f"{row['n_eligible_origins']} eligible origins; "
                  f"sha256={row['source_sha256']}")
        print("Do not evaluate until all source and noninspection attestations are reviewed.")
    else:
        if not (args.authorize_outcome_review and args.attest_unused_assets):
            raise PermissionError(
                "M5-v2 requires BOTH --authorize-outcome-review and "
                "--attest-unused-assets; no hidden model test execution"
            )
        result = evaluate(
            spec, pairs, args.output_dir, protocol_sha256=LOCKED_V2_SHA256
        )
        print("M5-v2 first external-asset evaluation completed: "
              f"{result['n_rows']} origin/model/horizon records")


if __name__ == "__main__":
    main()
