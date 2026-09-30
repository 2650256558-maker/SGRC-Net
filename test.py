"""Test-only convenience entry point.

AID/DFC15 default to final.pth. MLRSNet defaults to best.pth.
Use --checkpoint to override when testing a single ablation experiment.
"""
from main import build_parser, run


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.mode = "test"
    run(args)
