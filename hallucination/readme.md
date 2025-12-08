# Hallucination Detection Module

This folder contains components used to identify and filter hallucinated PHI in
LLM-generated de-identified texts.

Hallucinations are defined as:

• Introduction of **PHI tokens not present** in the original input  
• **Incorrect HIPAA legal explanations** supporting a redaction  

---

## Pipeline Overview

Candidate hallucinations are flagged when at least one of the following holds:

1. **Semantic mismatch**  
   Cosine similarity between source spans and generated spans is below a threshold
   using SBERT encoders.

2. **Entailment failure**
   Generated entity is not supported by the input according to NLI verification.

3. **Contrastive Verifier**
   A lightweight classifier trained on hallucinated vs non-hallucinated synthetic pairs.

Final hallucination probability:

