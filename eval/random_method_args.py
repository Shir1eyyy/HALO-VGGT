"""Shared CLI arguments for the random K/V reduction implementation."""

import argparse


def add_random_method_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--method",
        choices=("vggt", "fastvggt", "random"),
        default="fastvggt",
        help="attention reduction method",
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.05,
        help="K/V ratio for non-sensitive blocks when --method random",
    )
    parser.add_argument("--level1", type=float, default=0.05)
    parser.add_argument("--level2", type=float, default=0.1)
    parser.add_argument("--level3", type=float, default=0.15)
    parser.add_argument("--level4", type=float, default=0.2)
    parser.add_argument(
        "--head_group_count",
        type=int,
        choices=(2, 4),
        default=4,
        help="number of per-head keep-ratio groups in sensitive blocks",
    )
    parser.add_argument(
        "--rand_token_select",
        choices=("random", "uniform"),
        default="random",
        help="select retained K/V tokens randomly or at uniform intervals",
    )
    parser.add_argument(
        "--causal_online_controller",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the sampled-QK score to choose between base and per-head ratios",
    )
    parser.add_argument(
        "--causal_controller_enter_threshold",
        type=float,
        default=0.7,
        help="sampled-QK score threshold for per-head adaptive selection",
    )
    parser.add_argument(
        "--causal_controller_head_topk",
        type=int,
        default=4,
        help="number of highest-scoring heads used by the controller",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=33,
        help="seed for deterministic random token selection",
    )
    parser.add_argument(
        "--random_keep_first_frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep every K/V token from the first frame",
    )


def random_method_kwargs(args) -> dict:
    return {
        "method": args.method,
        "keep_ratio": args.keep_ratio,
        "level1": args.level1,
        "level2": args.level2,
        "level3": args.level3,
        "level4": args.level4,
        "head_group_count": args.head_group_count,
        "causal_online_controller": args.causal_online_controller,
        "causal_controller_enter_threshold": args.causal_controller_enter_threshold,
        "causal_controller_head_topk": args.causal_controller_head_topk,
        "rand_token_select": args.rand_token_select,
        "random_seed": args.random_seed,
        "random_keep_first_frame": args.random_keep_first_frame,
    }
