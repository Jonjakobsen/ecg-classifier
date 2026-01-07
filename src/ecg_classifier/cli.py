import argparse
from importlib.resources import files

from ecg_classifier.inference import run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ECG Classifier CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ---------- run ----------
    run_parser = subparsers.add_parser(
        "run",
        help="Run inference on user-provided ECG data",
    )
    run_parser.add_argument(
        "--input",
        required=True,
        help="Path to ECG file or directory",
    )
    run_parser.add_argument(
        "--format",
        choices=["wfdb", "csv"],
        default="wfdb",
        help="Input ECG format (default: wfdb)",
    )
    run_parser.add_argument(
        "--model",
        choices=["logreg", "gru"],
        default="logreg",
        help="Model to use (default: logreg)",
    )

    # ---------- demo ----------
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run inference on bundled demo ECG data",
    )
    demo_parser.add_argument(
        "--format",
        choices=["wfdb", "csv"],
        default="wfdb",
        help="Input ECG format (default: wfdb)",
    )

    demo_parser.add_argument(
        "--model",
        choices=["logreg", "gru"],
        default="logreg",
        help="Model to use (default: logreg)",
    )

    return parser


def print_result(result: dict) -> None:
    print("ECG classification result")
    print("-------------------------")
    print(f"Label      : {result['label']}")
    print(f"Confidence : {result['confidence']:.2f}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        result = run_inference(
            path=args.input,
            fmt=args.format,
            model_type=args.model,
        )

    elif args.command == "demo":
        demo_root = files("ecg_classifier") / "demo"

        if args.format == "wfdb":
            demo_path = demo_root / "wfdb"  / "demo_wfdb"
        elif args.format == "csv":
            demo_path = demo_root / "test_ecg_12lead.csv"
        else:
            raise ValueError(f"Unsupported demo format: {args.format}")

        result = run_inference(
            path=str(demo_path),
            fmt=args.format,
            model_type=args.model,
        )

    else:
        parser.print_help()
        return

    print_result(result)


if __name__ == "__main__":
    main()
