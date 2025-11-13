from __future__ import annotations

from typing import Dict, List

from .config import EvaluatorSettings
from .data_loader import iter_synthetic_frames, load_real_data
from .metrics import MetricContext, REGISTRY
from .reporting import write_summary


class Evaluator:
    def __init__(self, settings: EvaluatorSettings):
        self.settings = settings
        self.real_df = load_real_data(settings.training_data_path)

    def _run_metrics(self, context: MetricContext) -> Dict[str, Dict]:
        metric_outputs: Dict[str, Dict] = {}
        for metric_name in self.settings.metrics:
            metric_fn = REGISTRY.get(metric_name)
            if not metric_fn:
                print(f"[WARN] Unknown metric '{metric_name}' - skipping.")
                continue
            result = metric_fn(context)
            metric_outputs[metric_name] = result
        return metric_outputs

    def run(self) -> List[str]:
        """Run all configured metrics for each synthetic CSV and write summaries."""
        summary_paths: List[str] = []
        for synthetic_path, synthetic_df in iter_synthetic_frames(self.settings.input_path):
            context = MetricContext(
                real_df=self.real_df,
                synthetic_df=synthetic_df,
                settings=self.settings,
                synthetic_path=synthetic_path,
            )
            metric_outputs = self._run_metrics(context)
            from pathlib import Path

            filename = Path(synthetic_path).stem
            summary_path = write_summary(self.settings.output_dir, filename, metric_outputs)
            summary_paths.append(summary_path)
            print(f"[INFO] Wrote metrics for {synthetic_path} -> {summary_path}")
        return summary_paths
