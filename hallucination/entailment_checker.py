

from typing import Literal, Optional

from transformers import pipeline


class EntailmentChecker:
    """
    NLI-based entailment checker using a pre-trained MNLI model.

    By default, uses facebook/bart-large-mnli via the transformers pipeline.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: Optional[int] = 0,
    ) -> None:
        """
        Args:
            model_name: Hugging Face model name for the NLI classifier.
            device: Device index for the pipeline (0 for GPU, -1 for CPU).
        """
        self.nli = pipeline(
            task="text-classification",
            model=model_name,
            device=device if device is not None else -1,
        )

    def _predict_label(
        self,
        premise: str,
        hypothesis: str,
    ) -> Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]:
        """
        Run NLI and return the predicted label (ENTAILMENT / NEUTRAL / CONTRADICTION).
        """
        result = self.nli({"text": premise, "text_pair": hypothesis})[0]
        label = result["label"].upper()

        # Some models may return labels like "CONTRADICTION", "NEUTRAL", "ENTAILMENT"
        if "ENTAIL" in label:
            return "ENTAILMENT"
        if "CONTRAD" in label:
            return "CONTRADICTION"
        return "NEUTRAL"

    def is_entailed(self, premise: str, hypothesis: str) -> bool:
        """
        Returns True if the hypothesis is entailed by the premise.
        """
        label = self._predict_label(premise, hypothesis)
        return label == "ENTAILMENT"