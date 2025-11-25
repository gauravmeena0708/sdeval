"""Command-line interface for differential privacy mechanisms."""

import argparse
import sys
import torch
from .basic_mechanisms import gaussian_mechanism, laplace_mechanism, exponential_mechanism
from .cdp2adp import cdp_delta, cdp_eps, cdp_rho


def main(args: list[str] | None = None):
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Differential Privacy Mechanisms CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog="sdeval dp",
        epilog="""
Examples:
  # Apply Gaussian mechanism to a value
  sdeval dp gaussian --value 100 --sigma 1.0

  # Apply Laplace mechanism to a value
  sdeval dp laplace --value 50 --scale 2.0

  # Use exponential mechanism with scores
  sdeval dp exponential --scores 1.0,2.0,3.0,4.0 --epsilon 0.5 --sensitivity 1.0

  # Convert CDP to ADP parameters
  sdeval dp cdp-to-delta --rho 0.5 --epsilon 1.0
  sdeval dp cdp-to-eps --rho 0.5 --delta 1e-5
  sdeval dp adp-to-rho --epsilon 1.0 --delta 1e-5
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Gaussian mechanism
    gaussian_parser = subparsers.add_parser("gaussian", help="Apply Gaussian mechanism")
    gaussian_parser.add_argument("--value", type=float, required=True, help="Query value to perturb")
    gaussian_parser.add_argument("--sigma", type=float, required=True, help="Standard deviation of noise")
    gaussian_parser.add_argument("--shape", type=str, default="", help="Shape of tensor (comma-separated, e.g., '2,3')")

    # Laplace mechanism
    laplace_parser = subparsers.add_parser("laplace", help="Apply Laplace mechanism")
    laplace_parser.add_argument("--value", type=float, required=True, help="Query value to perturb")
    laplace_parser.add_argument("--scale", type=float, required=True, help="Scale parameter for Laplace distribution")
    laplace_parser.add_argument("--shape", type=str, default="", help="Shape of tensor (comma-separated, e.g., '2,3')")

    # Exponential mechanism
    exp_parser = subparsers.add_parser("exponential", help="Apply exponential mechanism")
    exp_parser.add_argument("--scores", type=str, required=True, help="Comma-separated scores (e.g., '1.0,2.0,3.0')")
    exp_parser.add_argument("--epsilon", type=float, required=True, help="Privacy parameter epsilon")
    exp_parser.add_argument("--sensitivity", type=float, required=True, help="Sensitivity of score function")

    # CDP to delta
    cdp_delta_parser = subparsers.add_parser("cdp-to-delta", help="Convert CDP (rho) to ADP delta")
    cdp_delta_parser.add_argument("--rho", type=float, required=True, help="Concentrated DP parameter rho")
    cdp_delta_parser.add_argument("--epsilon", type=float, required=True, help="Privacy parameter epsilon")

    # CDP to epsilon
    cdp_eps_parser = subparsers.add_parser("cdp-to-eps", help="Convert CDP (rho) to ADP epsilon")
    cdp_eps_parser.add_argument("--rho", type=float, required=True, help="Concentrated DP parameter rho")
    cdp_eps_parser.add_argument("--delta", type=float, required=True, help="Privacy parameter delta")

    # ADP to rho
    adp_rho_parser = subparsers.add_parser("adp-to-rho", help="Convert ADP (epsilon, delta) to CDP rho")
    adp_rho_parser.add_argument("--epsilon", type=float, required=True, help="Privacy parameter epsilon")
    adp_rho_parser.add_argument("--delta", type=float, required=True, help="Privacy parameter delta")

    # Parse arguments
    # If args is provided (from sdeval main), use it. Otherwise use sys.argv.
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 1

    try:
        if parsed_args.command == "gaussian":
            if parsed_args.shape:
                shape = tuple(map(int, parsed_args.shape.split(",")))
                q = torch.full(shape, parsed_args.value)
            else:
                q = torch.tensor([parsed_args.value])

            result = gaussian_mechanism(q, parsed_args.sigma)
            print(f"Original value: {q.tolist()}")
            print(f"Noisy result: {result.tolist()}")
            print(f"Privacy guarantee: {q.numel() / (2 * parsed_args.sigma ** 2):.6f}-zCDP")

        elif parsed_args.command == "laplace":
            if parsed_args.shape:
                shape = tuple(map(int, parsed_args.shape.split(",")))
                q = torch.full(shape, parsed_args.value)
            else:
                q = torch.tensor([parsed_args.value])

            result = laplace_mechanism(q, parsed_args.scale)
            print(f"Original value: {q.tolist()}")
            print(f"Noisy result: {result.tolist()}")
            print(f"Privacy guarantee: {1.0 / parsed_args.scale:.6f}-DP (assuming sensitivity=1)")

        elif parsed_args.command == "exponential":
            scores = torch.tensor([float(x) for x in parsed_args.scores.split(",")])
            result = exponential_mechanism(scores, parsed_args.epsilon, parsed_args.sensitivity)
            print(f"Scores: {scores.tolist()}")
            print(f"Selected index: {result.item()}")
            print(f"Privacy guarantee: {parsed_args.epsilon:.6f}-DP")

        elif parsed_args.command == "cdp-to-delta":
            delta = cdp_delta(parsed_args.rho, parsed_args.epsilon)
            print(f"For rho={parsed_args.rho} and epsilon={parsed_args.epsilon}:")
            print(f"delta = {delta:.10e}")
            print(f"This means {parsed_args.rho}-CDP implies ({parsed_args.epsilon}, {delta:.2e})-DP")

        elif parsed_args.command == "cdp-to-eps":
            epsilon = cdp_eps(parsed_args.rho, parsed_args.delta)
            print(f"For rho={parsed_args.rho} and delta={parsed_args.delta}:")
            print(f"epsilon = {epsilon:.6f}")
            print(f"This means {parsed_args.rho}-CDP implies ({epsilon:.6f}, {parsed_args.delta})-DP")

        elif parsed_args.command == "adp-to-rho":
            rho = cdp_rho(parsed_args.epsilon, parsed_args.delta)
            print(f"For epsilon={parsed_args.epsilon} and delta={parsed_args.delta}:")
            print(f"rho = {rho:.6f}")
            print(f"This means {rho:.6f}-CDP implies ({parsed_args.epsilon}, {parsed_args.delta})-DP")
            
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
