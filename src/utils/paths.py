

# src/utils/paths.py

from pathlib import Path
import os


def get_repo_root() -> Path:
    """
    Resolve repo root as the parent of the 'src' directory.

    Assumes this file lives inside:
        <repo_root>/src/utils/paths.py

    Works when running from repo root as:
        python -m src.training.train_regex_detector
    """
    return Path(__file__).resolve().parents[2]


def get_data_path() -> Path:
    """
    Returns the path to the single JSONL file with PHI regex data.

    Priority:
    1) PHI_DATA_PATH env var, if set.
    2) <repo_root>/data/phi_data.jsonl
    3) <repo_root>/Data/phi_data.jsonl

    Raises FileNotFoundError with a helpful message if not found.
    """
    # 1) Allow override via env var
    env_path = os.getenv("PHI_DATA_PATH")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"PHI_DATA_PATH is set to '{p}', but that file does not exist."
            )
        return p

    # 2) Common repo-local locations
    root = get_repo_root()
    candidates = [
        root / "data" / "phi_data.jsonl",  # lower-case (what you said you have)
        root / "Data" / "phi_data.jsonl",  # fallback (capital D)
    ]

    for p in candidates:
        if p.is_file():
            return p

    # 3) If nothing found, fail loudly with instructions
    msg = (
        "Could not find 'phi_data.jsonl'. Looked in:\n"
        f"  - {candidates[0]}\n"
        f"  - {candidates[1]}\n\n"
        "Either:\n"
        "  • Place your file at <repo_root>/data/phi_data.jsonl, OR\n"
        "  • Set PHI_DATA_PATH to the full path of the JSONL file."
    )
    raise FileNotFoundError(msg)


def get_output_dir(run_id: str) -> Path:
    """
    Create and return the output directory for the given run.

    Outputs go under:
        <repo_root>/outputs/regex_finetune_run_<run_id>/
    """
    root = get_repo_root()
    out_dir = root / "outputs" / f"regex_finetune_run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir