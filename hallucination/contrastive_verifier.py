

from dataclasses import dataclass


@dataclass
class ContrastiveVerifierConfig:
    """
    Configuration for the (stub) contrastive verifier.

    In the full system described in the paper, this would correspond to a
    lightweight classifier trained on hallucinated vs non-hallucinated spans.
    Here we expose a simple, interpretable heuristic interface.
    """

    risk_threshold: float = 0.5


class ContrastiveVerifier:
    """
    Stub interface for a contrastive hallucination verifier.

    This version does **not** ship a trained classifier or PHI data.
    Instead, it provides a simple heuristic:

    - If the generated span appears verbatim in the original context,
      hallucination risk is low (score ~ 0.0).
    - Otherwise, risk is moderate to high (score ~ 1.0).

    This is sufficient to demonstrate the integration in the pipeline.
    """

    def __init__(self, config: ContrastiveVerifierConfig | None = None) -> None:
        self.config = config or ContrastiveVerifierConfig()

    def score(self, span: str, context: str) -> float:
        """
        Return a risk score in [0, 1] for the given span given the context.
        Higher = more likely hallucinated.
        """
        if not span.strip():
            return 0.0

        # Very simple heuristic: if span appears in context, consider it low risk.
        if span.lower() in context.lower():
            return 0.0

        # Otherwise, mark it as high risk.
        return 1.0

    def is_high_risk(self, span: str, context: str) -> bool:
        """
        Returns True if the hallucination risk exceeds the configured threshold.
        """
        return self.score(span, context) > self.config.risk_threshold