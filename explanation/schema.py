

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ExplanationRecord:
    """
    Dataclass representing a single explanation for a PHI span.

    Fields:
        token:  The exact PHI text span.
        label:  PHI category label, e.g., NAME, DATE, HOSPITAL.
        reason: Natural-language explanation grounded in HIPAA.
    """

    token: str
    label: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain Python dict (useful for JSONL)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplanationRecord":
        """
        Build from a dict. Extra keys are ignored.

        Expected keys:
            - token
            - label
            - reason
        """
        return cls(
            token=data.get("token", ""),
            label=data.get("label", ""),
            reason=data.get("reason", ""),
        )