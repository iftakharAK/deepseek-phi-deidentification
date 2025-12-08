

from __future__ import annotations

try:
    # When run as a module: python -m Hallucination.eval_hallucination
    from .hallucination_detector import HallucinationDetector
except ImportError:
    # When run directly: python eval_hallucination.py from inside the folder
    from hallucination_detector import HallucinationDetector


def demo() -> None:
    """
    Tiny demo using synthetic (non-PHI) examples.
    """
    detector = HallucinationDetector()

    original = "Patient John Smith was admitted on 05/12/2024 for chest pain."

    supported_span = "John Smith"
    hallucinated_span = "Michael Johnson"

    print("=== Supported Span Example ===")
    decision_supported = detector.analyze_span(original, supported_span)
    print(decision_supported.to_dict())

    print("\n=== Hallucinated Span Example ===")
    decision_hallucinated = detector.analyze_span(original, hallucinated_span)
    print(decision_hallucinated.to_dict())

    print("\nSummary:")
    print(f"Supported span hallucinated? {decision_supported.is_hallucinated}")
    print(f"Hallucinated span hallucinated? {decision_hallucinated.is_hallucinated}")


if __name__ == "__main__":
    demo()