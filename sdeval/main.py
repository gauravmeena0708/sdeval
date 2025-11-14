from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import load_settings_from_args
from .evaluator import Evaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic Data Evaluator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python -m sdeval.main --input-path synthetic.csv --real-data-csv-path real.csv --output-dir outputs

  # Batch evaluation with timestamped output
  python -m sdeval.main --input-path synthetic_dir/ --real-data-csv-path real.csv --output-dir outputs --timestamp

  # Quiet mode with specific metrics
  python -m sdeval.main --input-path synthetic.csv --real-data-csv-path real.csv --output-dir outputs --quiet --metrics statistical coverage

  # Verbose mode with comparison table
  python -m sdeval.main --input-path synthetic_dir/ --real-data-csv-path real.csv --output-dir outputs --verbose --compare
        """
    )

    # Required arguments
    parser.add_argument("--input-path", required=True,
                       help="Synthetic CSV file or directory.")
    parser.add_argument("--real-data-csv-path", required=True,
                       help="Real dataset CSV for evaluation.")
    parser.add_argument("--output-dir", required=True,
                       help="Directory for evaluation outputs.")

    # Optional configuration
    parser.add_argument("--model-path",
                       help="Optional plausibility model directory.")
    parser.add_argument("--configs",
                       help="Optional JSON configs file.")
    parser.add_argument("--constraints-path",
                       help="Optional JSON file containing constraint rules.")
    parser.add_argument("--metrics", nargs="+",
                       help="Subset of metrics to run (default: all).")
    parser.add_argument("--seed", type=int,
                       help="Random seed for reproducibility.")

    # CLI enhancements
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output with detailed progress.")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress all output except errors.")
    parser.add_argument("--no-progress", action="store_true",
                       help="Disable progress bars.")
    parser.add_argument("--timestamp", "-t", action="store_true",
                       help="Add timestamp to output directory name.")
    parser.add_argument("--compare", "-c", action="store_true",
                       help="Generate comparison table for batch evaluations.")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite output directory if it exists.")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate diagnostic plots for each synthetic file.")
    parser.add_argument("--html-report", action="store_true",
                       help="Generate standalone HTML report(s) summarizing metrics and visuals.")

    return parser


def create_output_dir(base_dir: str, timestamp: bool = False, overwrite: bool = False) -> str:
    """Create output directory with optional timestamp."""
    output_path = Path(base_dir)

    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path / timestamp_str

    if output_path.exists() and not overwrite:
        if timestamp:
            # With timestamp, should be unique
            pass
        else:
            print(f"[WARNING] Output directory exists: {output_path}", file=sys.stderr)
            print(f"[WARNING] Use --overwrite to replace existing files", file=sys.stderr)

    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def print_header(quiet: bool = False):
    """Print CLI header."""
    if quiet:
        return

    header = """
╔═══════════════════════════════════════════════════════════╗
║        Synthetic Data Evaluator (sdeval)                  ║
║        Comprehensive Quality Assessment                   ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(header)


def print_summary(summaries: list, metrics_used: list, quiet: bool = False):
    """Print evaluation summary."""
    if quiet:
        return

    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)
    print(f"Files evaluated: {len(summaries)}")
    print(f"Metrics computed: {', '.join(metrics_used)}")
    print(f"Results saved to: {len(summaries)} JSON file(s)")
    print("="*60)


def generate_comparison_table(summaries: list, output_dir: str, verbose: bool = False) -> Optional[str]:
    """Generate comparison table for batch evaluations."""
    if len(summaries) < 2:
        return None

    if verbose:
        print("\n📊 Generating comparison table...")

    try:
        import pandas as pd

        # Load all summary JSONs
        comparison_data = []
        for summary_path in summaries:
            with open(summary_path, 'r') as f:
                data = json.load(f)

            # Flatten metrics
            row = {'file': Path(summary_path).stem}
            for category, metrics in data.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            row[f"{category}.{metric}"] = value

            comparison_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        # Save as CSV
        comparison_path = Path(output_dir) / "comparison_table.csv"
        df.to_csv(comparison_path, index=False)

        if verbose:
            print(f"   ✓ Comparison table saved: {comparison_path}")
            print(f"   ✓ Columns: {len(df.columns)}")
            print(f"   ✓ Rows: {len(df)}")

        return str(comparison_path)

    except Exception as e:
        print(f"[WARNING] Could not generate comparison table: {e}", file=sys.stderr)
        return None


def main(argv=None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle verbose/quiet modes
    verbose = args.verbose
    quiet = args.quiet

    if verbose and quiet:
        print("[ERROR] Cannot use both --verbose and --quiet", file=sys.stderr)
        return 1

    # Print header
    print_header(quiet)

    try:
        # Create output directory with optional timestamp
        output_dir = create_output_dir(args.output_dir, args.timestamp, args.overwrite)
        args.output_dir = output_dir

        if verbose:
            print(f"📁 Output directory: {output_dir}")
            print(f"📂 Input path: {args.input_path}")
            print(f"📄 Real data: {args.real_data_csv_path}")
            if args.metrics:
                print(f"📊 Metrics: {', '.join(args.metrics)}")
            else:
                print(f"📊 Metrics: all")
            print()

        # Load settings
        settings = load_settings_from_args(args)

        # Set verbosity in settings for evaluator
        settings.verbose = verbose
        settings.quiet = quiet
        settings.show_progress = not args.no_progress and not quiet

        # Create evaluator
        if verbose:
            print("🔄 Initializing evaluator...")
            print(f"   Loading real data: {settings.real_data_path}")

        evaluator = Evaluator(settings)

        if verbose:
            print(f"   ✓ Real data loaded: {len(evaluator.real_df)} rows")
            print()

        # Run evaluation
        if verbose:
            print("⚙️  Running evaluation...")

        summaries = evaluator.run()

        # Print summary
        print_summary(summaries, settings.metrics, quiet)

        # Generate comparison table if requested
        if args.compare and len(summaries) > 1:
            comparison_path = generate_comparison_table(summaries, output_dir, verbose)
            if comparison_path and not quiet:
                print(f"\n📊 Comparison table: {comparison_path}")

        # Output JSON summary list
        if not quiet:
            print(f"\n📋 Summary files:")
            for summary in summaries:
                print(f"   - {summary}")

        # Always output JSON for programmatic access
        output_json = {"summaries": summaries, "output_dir": output_dir}
        if args.compare and len(summaries) > 1:
            comparison_path = Path(output_dir) / "comparison_table.csv"
            if comparison_path.exists():
                output_json["comparison_table"] = str(comparison_path)

        if not quiet:
            print(f"\n{json.dumps(output_json, indent=2)}")

        return 0

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    except ValueError as e:
        print(f"[ERROR] Invalid input: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
