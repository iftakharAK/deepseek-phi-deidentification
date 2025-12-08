

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

try:
    # When used as a package: python -m Hallucination.eval_hallucination
    from .similarity_filter import SimilarityFilter
    from .entailment_checker import EntailmentChecker
    from .contrastive_verifier import ContrastiveVerifier
except ImportError:
    # Fallback when running inside the folder directly
    from similarity_filter import SimilarityFilter
    from entailment_checker import EntailmentChecker
    from contrastive_verifier import ContrastiveVerifier


@dataclass
class HallucinationDecision:
    span: str
    similarity_supported: bool
    nli_entailed: bool
    verifier_high_risk: bool
    is_hallucinated: bool
    similarity_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten / remove None if needed
        return {k: v for k, v in d.items() if v is not None}


class HallucinationDetector:
    """
    High-level hallucination detector combining:
    - SBERT similarity filter
    - NLI entailment checker
    - Contrastive verifier

    A span is considered hallucinated if:
    - Similarity filter does not support it, OR
    - NLI does not entail it, OR
    - Verifier marks it as high risk.
    """

    def __init__(
        self,
        similarity_filter: Optional[SimilarityFilter] = None,
        entailment_checker: Optional[EntailmentChecker] = None,
        contrastive_verifier: Optional[ContrastiveVerifier] = None,
    ) -> None:
        self.sim_filter = similarity_filter or SimilarityFilter()
        self.entailment = entailment_checker or EntailmentChecker()
        self.verifier = contrastive_verifier or ContrastiveVerifier()

    def analyze_span(
        self,
        original_text: str,
        generated_span: str,
    ) -> HallucinationDecision:
        """
        Analyze a single generated span with respect to the original text.
        """
        sim_score = self.sim_filter.similarity(original_text, generated_span)
        similarity_supported = sim_score >= self.sim_filter.threshold

        nli_entailed = self.entailment.is_entailed(
            premise=original_text,
            hypothesis=generated_span,
        )

        verifier_high_risk = self.verifier.is_high_risk(
            span=generated_span,
            context=original_text,
        )

        # Conservative rule: mark hallucinated if ANY detector flags risk.
        is_hallucinated = (not similarity_supported) or (not nli_entailed) or verifier_high_risk

        return HallucinationDecision(
            span=generated_span,
            similarity_supported=similarity_supported,
            nli_entailed=nli_entailed,
            verifier_high_risk=verifier_high_risk,
            is_hallucinated=is_hallucinated,
            similarity_score=sim_score,
        )

    def is_hallucinated(self, original_text: str, generated_span: str) -> bool:
        """
        Convenience wrapper: only returns True/False.
        """
        decision = self.analyze_span(original_text, generated_span)
        return decision.is_hallucinated

    def analyze_spans(
        self,
        original_text: str,
        spans: List[str],
    ) -> List[HallucinationDecision]:
        """
        Analyze multiple spans; useful when you have entity-level outputs.
        """
        return [self.analyze_span(original_text, span) for span in spans]