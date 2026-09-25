"""Command-line entry point. Heavy embedding dependencies are loaded lazily."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, curate, and review image datasets"
    )
    parser.add_argument(
        "command",
        choices=[
            "extract",
            "embed",
            "report",
            "timeline",
            "filter",
            "select",
            "audit",
            "export",
        ],
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Job TOML beside the dataset"
    )
    parser.add_argument("--step-seconds", type=float, default=10)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.command in {"filter", "select", "export"} and args.policy is None:
        parser.error(f"{args.command} requires --policy")
    if args.command == "export" and args.plan is None:
        parser.error("export requires --plan")
    try:
        config = load_config(args.config)
        _execute(args, config)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"dataset-curation: {error}\n")


def _execute(args: argparse.Namespace, config: Config) -> None:
    if args.command == "extract":
        from .extraction import extract

        extract(config)
    elif args.command == "embed":
        from .embeddings import embed

        embed(config)
    elif args.command == "timeline":
        from .timeline import build

        build(config, args.step_seconds)
    elif args.command == "filter":
        from .eligibility import apply

        apply(config, args.policy)
    elif args.command == "select":
        from .selection import select

        select(config, args.policy)
    elif args.command == "audit":
        from .audit import audit

        audit(config)
    elif args.command == "export":
        from .export import export

        export(config, args.policy, args.plan)
    else:
        from .report import build

        build(config)
