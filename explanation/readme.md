# Explanation Module

This folder contains a **self-contained explanation module** for the PHI de-identification pipeline.  
It provides:

- A simple schema for explanation records
- HIPAA-grounded explanation templates for PHI labels (e.g., `NAME`, `DATE`, `HOSPITAL`)
- Utilities to generate explanations for detected PHI spans (Python API + CLI)
- Optional evaluation utilities for explanation accuracy

---

## Folder Structure

```text
src/explanation/
│
├─ __init__.py
├─ schema.py          # Dataclass for explanation records
├─ templates.py       # HIPAA-grounded explanation templates
├─ generator.py       # Explanation generation logic + CLI entry
└─ eval.py            # (Optional) simple explanation accuracy evaluation
