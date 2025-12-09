

from typing import Dict

# HIPAA-grounded templates keyed by PHI label.
# These are aligned with the wording you used in the paper.
TEMPLATES: Dict[str, str] = {
    "NAME": (
        "Redacted because it matches a patient or staff name under "
        "HIPAA §164.514(b)(2)(i)(A)."
    ),
    "DATE": (
        "Redacted because it is a date or time linked to an individual, "
        "considered PHI under HIPAA §164.514(b)(2)(i)(C)."
    ),
    "HOSPITAL": (
        "Redacted because hospital names are healthcare identifiers under "
        "HIPAA §164.514(b)(2)(i)(B)."
    ),
    "LOCATION": (
        "Redacted because it reveals a geographic location tied to a patient, "
        "which is PHI under HIPAA §164.514(b)(2)(i)(B)."
    ),
    "ADDRESS": (
        "Redacted because it contains a mailing or street address that can "
        "identify the individual, per HIPAA §164.514(b)(2)(i)(B)."
    ),
    "ZIP": (
        "Redacted because ZIP codes are considered geographic identifiers "
        "under HIPAA §164.514(b)(2)(i)(B)."
    ),
    "CONTACT": (
        "Redacted because phone numbers or email addresses are direct contact "
        "identifiers under HIPAA §164.514(b)(2)(i)(A)."
    ),
    "ID_NUMBER": (
        "Redacted because it includes a unique identifier (e.g., MRN or SSN) "
        "under HIPAA §164.514(b)(2)(i)(C)."
    ),
    "ETHNICITY": (
        "Redacted because ethnicity can contribute to re-identification risk "
        "as a demographic attribute under HIPAA §164.514(b)(2)(i)(C)."
    ),
    "RELIGION": (
        "Redacted because religious affiliation is a sensitive demographic "
        "identifier under HIPAA §164.514(b)(2)(i)(C)."
    ),
    "MARITAL_STATUS": (
        "Redacted because marital status is a demographic attribute that may "
        "help re-identify an individual under HIPAA §164.514(b)(2)(i)(C)."
    ),
    "DIAGNOSIS": (
        "Redacted because diagnoses contain sensitive health information "
        "about an identifiable patient under HIPAA §164.514(b)(2)(i)(R)."
    ),
}


DEFAULT_TEMPLATE = (
    "Redacted because it is classified as Protected Health Information (PHI) "
    "under the HIPAA Safe Harbor provisions."
)


def get_explanation_template(label: str) -> str:
    """
    Return a HIPAA-grounded explanation template for the given PHI label.

    If the label is not found, a safe default PHI explanation is returned.

    Args:
        label: PHI category label, e.g., 'NAME', 'DATE'.

    Returns:
        A string explanation template.
    """
    if not label:
        return DEFAULT_TEMPLATE

    normalized = label.strip().upper()
    return TEMPLATES.get(normalized, DEFAULT_TEMPLATE)