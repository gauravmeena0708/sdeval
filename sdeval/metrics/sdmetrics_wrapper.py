from . import register_metric, MetricContext

def _get_datavalidity_scores(context: MetricContext) -> dict:
    """
    Helper function to get data validity scores from sdmetrics.
    """
    from sdmetrics.single_table import DataValidity
    from sdmetrics.metadata import SingleTableMetadata

    real_metadata = SingleTableMetadata()
    real_metadata.detect_from_dataframe(data=context.real_data)

    return DataValidity.compute_breakdown(
        real_data=context.real_data,
        synthetic_data=context.synthetic_data,
        metadata=real_metadata
    )

@register_metric("alpha_precision")
def alpha_precision(context: MetricContext) -> float:
    """
    Calculates the alpha precision of the synthetic data.
    It measures the fraction of categories in the synthetic data that also appear in the real data.
    A low score means the model is "hallucinating" or inventing categories that don't exist.
    Higher is better.
    This is implemented using sdmetrics' "Data Synthesis" score.
    """
    scores = _get_datavalidity_scores(context)
    return scores.get("Data Synthesis", {}).get("score", 0.0)

@register_metric("beta_recall")
def beta_recall(context: MetricContext) -> float:
    """
    Calculates the beta recall of the synthetic data.
    It measures the fraction of real-data categories that are captured in the synthetic data.
    A low score means the model is failing to generate rare categories.
    Higher is better.
    This is implemented using sdmetrics' "Data Coverage" score.
    """
    scores = _get_datavalidity_scores(context)
    return scores.get("Data Coverage", {}).get("score", 0.0)
