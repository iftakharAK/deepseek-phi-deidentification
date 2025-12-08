

from typing import Optional

from sentence_transformers import SentenceTransformer, util


class SimilarityFilter:
    """
    SBERT-based semantic similarity filter.

    Uses cosine similarity between the original context and generated span.
    If similarity < threshold -> potential hallucination trigger.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.75,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: SentenceTransformer model identifier.
            threshold: Cosine similarity threshold for considering a span supported.
            device: Optional device (e.g., "cpu", "cuda"). If None, auto-detect.
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.threshold = threshold

    def similarity(self, source_span: str, generated_span: str) -> float:
        """
        Compute cosine similarity between a source span and a generated span.
        """
        emb_src = self.model.encode(source_span, convert_to_tensor=True)
        emb_gen = self.model.encode(generated_span, convert_to_tensor=True)
        sim = util.cos_sim(emb_src, emb_gen).item()
        return float(sim)

    def is_supported(self, source_span: str, generated_span: str) -> bool:
        """
        Returns True if the generated_span is semantically supported by the
        source_span according to the configured threshold.
        """
        sim = self.similarity(source_span, generated_span)
        return sim >= self.threshold