"""argument parser and dispatch: `python -m src <subcommand>`."""

from __future__ import annotations

import argparse
from multiprocessing import freeze_support
from pathlib import Path

from .stages.orchestrate import (
    run_cache_embed_only,
    run_full_pipeline,
    run_train_from_artifact,
)
from .subprocess_runner import run_isolated_pipeline

_EMBED = ("sentence_transformer", "bert_italian", "e5")


def _add_embed_args(p):
    p.add_argument("--embed", choices=_EMBED, default="sentence_transformer")


def _add_train_args(p):
    p.add_argument(
        "--mode",
        choices=("sklearn", "cnn"),
        default="sklearn",
    )
    p.add_argument("--kmeans", type=int, default=0, metavar="K")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--out", type=str, default=None)


def build_parser():
    parser = argparse.ArgumentParser(description="Crime-news NLP pipeline stages.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Full pipeline in one process (fastest if RAM fits).")
    _add_train_args(p_run)
    _add_embed_args(p_run)

    p_ce = sub.add_parser(
        "cache-embed",
        help="Stage 1 only: texts to NPZ + .meta.json (unload encoder before training).",
    )
    p_ce.add_argument("--out", type=str, required=True, help="Path to .npz")
    _add_embed_args(p_ce)

    p_tr = sub.add_parser("train", help="Stage 2 only: load NPZ then train heads.")
    p_tr.add_argument("--artifact", type=str, required=True, help="NPZ from cache-embed")
    _add_train_args(p_tr)

    p_iso = sub.add_parser(
        "run-subprocess",
        help="Run cache-embed then train as two separate Python processes.",
    )
    p_iso.add_argument("--work-dir", type=str, required=True, help="Folder for stage_embed.npz")
    _add_embed_args(p_iso)
    _add_train_args(p_iso)

    p_leg = sub.add_parser(
        "legacy-run",
        help="Same flags as old `python -m src.run_experiment` (flat, no subcommand).",
    )
    _add_train_args(p_leg)
    _add_embed_args(p_leg)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_path = Path(args.out) if getattr(args, "out", None) else None

    if args.command == "run":
        run_full_pipeline(args.mode, args.embed, args.kmeans, args.n_jobs, out_path)
    elif args.command == "cache-embed":
        run_cache_embed_only(Path(args.out), args.embed)
    elif args.command == "train":
        run_train_from_artifact(
            Path(args.artifact), args.mode, args.kmeans, args.n_jobs, out_path
        )
    elif args.command == "run-subprocess":
        run_isolated_pipeline(
            Path(args.work_dir),
            args.embed,
            args.mode,
            args.kmeans,
            args.n_jobs,
            out_path,
        )
    elif args.command == "legacy-run":
        run_full_pipeline(args.mode, args.embed, args.kmeans, args.n_jobs, out_path)


def main_cli():
    freeze_support()
    main()


def main_legacy_flat():
    """Used by src/run_experiment.py: old single-level argparse."""
    freeze_support()
    p = argparse.ArgumentParser(description="Reproducible crime-news embedding + classification pipeline.")
    p.add_argument("--mode", choices=("sklearn", "cnn"), default="sklearn")
    p.add_argument("--embed", choices=_EMBED, default="sentence_transformer")
    p.add_argument("--kmeans", type=int, default=0, metavar="K")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    ns = p.parse_args()
    run_full_pipeline(
        ns.mode,
        ns.embed,
        ns.kmeans,
        ns.n_jobs,
        Path(ns.out) if ns.out else None,
    )
