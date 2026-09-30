"""Train-only convenience entry point."""
from main import build_parser, run


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.mode = "train"
    run(args)
