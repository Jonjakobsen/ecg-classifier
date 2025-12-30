import argparse
from ecg_classifier.inference import run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ECG Classifier CLI"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Inference command
    infer_parser = subparsers.add_parser(
        "infer", help="Run inference on an ECG file"
    )
    infer_parser.add_argument(
        "--input",
        required=True,
        help="Path to ECG file"
    )

    infer_parser.add_argument(
        "--format",
        choices=["wfdb", "csv"],
        default="wfdb",
        help="Input ECG format (default: wfdb)"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "infer":
        result = run_inference(
            path=args.input,
            fmt=args.format
            )
        
        print("ECG classification result")
        print("-------------------------")
        print(f"Label      : {result['label']}")
        print(f"Confidence : {result['confidence']:.2f}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()



