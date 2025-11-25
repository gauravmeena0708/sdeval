from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .config import EvaluatorSettings
from .data_loader import detect_column_types, iter_synthetic_frames, load_real_data
from .metrics import MetricContext, REGISTRY
from .reporting import generate_html_report, write_summary
from .visualization import generate_visualization_suite


class Evaluator:
    def __init__(self, settings: EvaluatorSettings):
        self.settings = settings
        self.real_df = load_real_data(settings.real_data_path)
        self.column_types = detect_column_types(self.real_df)

    def _run_metrics(self, context: MetricContext) -> Dict[str, Dict]:
        """Run all configured metrics with optional progress tracking."""
        metric_outputs: Dict[str, Dict] = {}

        # Get metrics to run
        metrics_to_run = self.settings.metrics

        # Setup progress bar if enabled
        if self.settings.show_progress:
            try:
                from tqdm import tqdm
                metric_iterator = tqdm(metrics_to_run, desc="Computing metrics", leave=False)
            except ImportError:
                metric_iterator = metrics_to_run
        else:
            metric_iterator = metrics_to_run

        for metric_name in metric_iterator:
            metric_fn = REGISTRY.get(metric_name)
            if not metric_fn:
                if not self.settings.quiet:
                    print(f"[WARN] Unknown metric '{metric_name}' - skipping.")
                continue

            # Update progress bar description if available
            if self.settings.show_progress and hasattr(metric_iterator, 'set_description'):
                metric_iterator.set_description(f"Computing {metric_name}")

            result = metric_fn(context)
            metric_outputs[metric_name] = result

            if self.settings.verbose:
                # Print metric completion
                enabled_key = f"{metric_name}_enabled"
                if enabled_key in result and not result[enabled_key]:
                    reason = result.get(f"{metric_name}_reason", "disabled")
                    print(f"   ⊘ {metric_name}: {reason}")
                else:
                    print(f"   ✓ {metric_name}")

        return metric_outputs

    def run(self) -> List[str]:
        """Run all configured metrics for each synthetic CSV and write summaries."""
        summary_paths: List[str] = []

        # Collect all synthetic files first for progress bar
        synthetic_files = list(iter_synthetic_frames(self.settings.input_path))

        # Setup progress bar for files if enabled
        if self.settings.show_progress and len(synthetic_files) > 1:
            try:
                from tqdm import tqdm
                file_iterator = tqdm(synthetic_files, desc="Evaluating files")
            except ImportError:
                file_iterator = synthetic_files
        else:
            file_iterator = synthetic_files

        for synthetic_path, synthetic_df in file_iterator:
            # Update progress bar description if available
            if self.settings.show_progress and hasattr(file_iterator, 'set_description'):
                filename = Path(synthetic_path).name
                file_iterator.set_description(f"Evaluating {filename}")

            if self.settings.verbose:
                print(f"\n📄 Processing: {synthetic_path}")
                print(f"   Rows: {len(synthetic_df):,}")
                print(f"   Columns: {len(synthetic_df.columns)}")

            context = MetricContext(
                real_df=self.real_df,
                synthetic_df=synthetic_df,
                settings=self.settings,
                synthetic_path=synthetic_path,
            )

            metric_outputs = self._run_metrics(context)

            filename = Path(synthetic_path).stem
            summary_path = write_summary(self.settings.output_dir, filename, metric_outputs)
            summary_paths.append(summary_path)

            visuals = self._maybe_generate_visuals(
                synthetic_df,
                filename,
                metric_outputs.get("constraints"),
                metric_outputs.get("statistical"),
            ) or []

            if self.settings.html_report:
                generate_html_report(
                    self.settings.output_dir,
                    filename,
                    metric_outputs,
                    visuals,
                )

            if not self.settings.quiet and not self.settings.show_progress:
                print(f"[INFO] Wrote metrics for {synthetic_path} -> {summary_path}")

        return summary_paths

    def _maybe_generate_visuals(
        self,
        synthetic_df,
        filename: str,
        constraints_metrics: Dict | None,
        statistical_metrics: Dict | None,
    ) -> List[Path] | None:
        if not self.settings.visualize:
            return []

        constraint_details = None
        if constraints_metrics and constraints_metrics.get("constraints_enabled"):
            constraint_details = constraints_metrics.get("constraints_rule_details")

        output_dir = Path(self.settings.output_dir) / "visualizations"
        created = generate_visualization_suite(
            self.real_df,
            synthetic_df,
            self.column_types.get("numerical_columns", []),
            self.column_types.get("categorical_columns", []),
            output_dir,
            filename,
            constraint_details=constraint_details,
            stat_metrics=statistical_metrics,
        )
        if created and self.settings.verbose:
            print(f"   📉 Visualizations saved under {output_dir / filename}")
        return created
