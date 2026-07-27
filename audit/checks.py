"""
Rule-based data quality checks for the Clinical LLM Micro-Benchmark audit.

Each check function takes the full patient data list and returns a list of
flag dicts. Flags are for human review only — never auto-merged into gold labels.

Flag schema:
{
    "patient_id": str,
    "item_id": str,        # visit_id or test_name
    "field": str,          # which field(s) triggered the check
    "check_name": str,     # machine-readable check identifier
    "flag_source": str,    # "rule" (always, for this module)
    "severity": str,       # "high" | "moderate" | "info"
    "detail": str,         # human-readable description
    "accepted": None       # to be set by user in audit UI
}
"""

import re
from datetime import datetime
from typing import Any


def _flag(patient_id: str, item_id: str, field: str, check_name: str,
          severity: str, detail: str) -> dict:
    """Helper to construct a flag dict."""
    return {
        "patient_id": patient_id,
        "item_id": item_id,
        "field": field,
        "check_name": check_name,
        "flag_source": "rule",
        "severity": severity,
        "detail": detail,
        "accepted": None,
    }


# ---------------------------------------------------------------------------
# Negation patterns — intentionally broad, catches "no X", "denies X", "not X"
# ---------------------------------------------------------------------------
_NEGATION_RE = re.compile(
    r'\b(no|not|denies|denied|without|absent|negative)\s+(\w[\w\s]{0,40})',
    re.IGNORECASE,
)


def check_negation_mismatch(patients: list[dict]) -> list[dict]:
    """
    Scan raw_diagnosis + raw_complaints for negation patterns (e.g. 'no neuropathy').
    If the negated term appears as a positive entry in structured diagnoses/complaints
    with is_negated=false, flag it.
    """
    flags = []
    for p in patients:
        pid = p["patient_id"]
        for v in p.get("visit_list", []):
            vid = v.get("visit_id", "?")

            # Gather all raw text items
            raw_items = []
            for field_name in ("raw_diagnosis", "raw_complaints"):
                raw = v.get(field_name)
                if isinstance(raw, list):
                    raw_items.extend([(field_name, item) for item in raw if isinstance(item, str)])
                elif isinstance(raw, str):
                    raw_items.append((field_name, raw))

            # Find negated terms in raw text
            negated_terms = []
            for field_name, text in raw_items:
                for m in _NEGATION_RE.finditer(text.lower()):
                    negated_terms.append((field_name, m.group(2).strip()))

            if not negated_terms:
                continue

            # Check structured diagnoses for the negated term appearing as positive
            structured_dx = v.get("diagnoses", []) or []
            for field_name, neg_term in negated_terms:
                for dx in structured_dx:
                    dx_name = (dx.get("diagnosis") or "").lower()
                    is_neg = dx.get("is_negated", False)
                    # Check if the negated term is a substring of the structured diagnosis
                    # or vice versa (handles "neuropathy" matching "no neuropathy")
                    if neg_term in dx_name or dx_name in neg_term:
                        if not is_neg:
                            flags.append(_flag(
                                pid, vid, f"{field_name} → diagnoses",
                                "negation_mismatch", "high",
                                f'Raw text says "no {neg_term}" but structured diagnoses '
                                f'lists "{dx.get("diagnosis")}" with is_negated=false'
                            ))

                # Also check structured complaints
                structured_cx = v.get("complaints", []) or []
                for cx in structured_cx:
                    cx_name = (cx if isinstance(cx, str) else str(cx)).lower()
                    if neg_term in cx_name or cx_name.strip() in neg_term:
                        flags.append(_flag(
                            pid, vid, f"{field_name} → complaints",
                            "negation_mismatch", "moderate",
                            f'Raw text says "no {neg_term}" but structured complaints '
                            f'includes "{cx}"'
                        ))

    return flags


def check_null_structured_with_raw(patients: list[dict]) -> list[dict]:
    """
    Find medications where medicine_name, generic_name, and medicine_status
    are all null but rx_med_name is populated — means the upstream parser
    completely failed to resolve this medication.
    """
    flags = []
    for p in patients:
        pid = p["patient_id"]
        for v in p.get("visit_list", []):
            vid = v.get("visit_id", "?")
            for med in (v.get("medications") or []):
                rx = med.get("rx_med_name")
                if not rx:
                    continue
                if (med.get("medicine_name") is None
                        and med.get("generic_name") is None
                        and med.get("medicine_status") is None):
                    flags.append(_flag(
                        pid, vid, "medications",
                        "null_structured_with_raw", "moderate",
                        f'rx_med_name="{rx}" has medicine_name, generic_name, '
                        f'and medicine_status all null — completely unresolved'
                    ))
    return flags


def check_diagnosis_medication_coherence(patients: list[dict]) -> list[dict]:
    """
    Basic coherence: if raw_diagnosis mentions diabetes/DM, check that at
    least one anti-diabetic medication exists. Similarly for hypertension.
    Only flags at info severity — absence may have legitimate reasons.
    """
    flags = []
    dm_rx_keywords = {"metformin", "glimepiride", "insulin", "gliclazide",
                      "sitagliptin", "vildagliptin", "pioglitazone",
                      "empagliflozin", "dapagliflozin", "canagliflozin",
                      "liraglutide", "semaglutide", "glipizide", "acarbose"}
    htn_rx_keywords = {"amlodipine", "telmisartan", "losartan", "ramipril",
                       "enalapril", "lisinopril", "atenolol", "metoprolol",
                       "hydrochlorothiazide", "chlorthalidone", "furosemide",
                       "spironolactone", "nifedipine", "cilnidipine", "olmesartan"}

    for p in patients:
        pid = p["patient_id"]
        for v in p.get("visit_list", []):
            vid = v.get("visit_id", "?")
            raw_dx = v.get("raw_diagnosis", [])
            if isinstance(raw_dx, str):
                raw_dx = [raw_dx]

            raw_text = " ".join(str(x).lower() for x in (raw_dx or []))

            # Collect all generic names from medications
            generic_names = set()
            for med in (v.get("medications") or []):
                gn = med.get("generic_name")
                if gn is None:
                    continue
                # Handle list-typed generic_name (e.g. [None, 'paracetamol'])
                if isinstance(gn, list):
                    for g in gn:
                        if isinstance(g, str):
                            generic_names.update(g.lower().split(" + "))
                elif isinstance(gn, str):
                    generic_names.update(gn.lower().split(" + "))

            # Check diabetes
            has_dm = any(kw in raw_text for kw in
                         ["dm", "diabetes", "type 2", "type2", "dm2", "t2dm"])
            if has_dm:
                has_dm_rx = bool(generic_names & dm_rx_keywords)
                if not has_dm_rx:
                    flags.append(_flag(
                        pid, vid, "raw_diagnosis → medications",
                        "dx_med_coherence", "info",
                        f"Raw diagnosis mentions diabetes but no recognized "
                        f"anti-diabetic found in medications at this visit"
                    ))

            # Check hypertension
            has_htn = any(kw in raw_text for kw in
                          ["hypertension", "htn", "high bp"])
            if has_htn:
                has_htn_rx = bool(generic_names & htn_rx_keywords)
                if not has_htn_rx:
                    flags.append(_flag(
                        pid, vid, "raw_diagnosis → medications",
                        "dx_med_coherence", "info",
                        f"Raw diagnosis mentions hypertension but no recognized "
                        f"anti-hypertensive found in medications at this visit"
                    ))

    return flags


def check_hba1c_range(patients: list[dict]) -> list[dict]:
    """Flag HbA1c readings outside the ~3.0–20.0% physiological range."""
    flags = []
    for p in patients:
        pid = p["patient_id"]
        for t in (p.get("test_list") or []):
            if (t.get("test_name") or "").lower() != "hba1c":
                continue
            for r in (t.get("readings") or []):
                val = r.get("value")
                date = r.get("date", "?")
                if val is not None:
                    try:
                        v = float(val)
                        if v < 3.0 or v > 20.0:
                            flags.append(_flag(
                                pid, "hba1c", f"reading {date}",
                                "hba1c_out_of_range", "high",
                                f"HbA1c={v}% on {date} is outside plausible "
                                f"3.0–20.0% range"
                            ))
                    except (ValueError, TypeError):
                        flags.append(_flag(
                            pid, "hba1c", f"reading {date}",
                            "hba1c_parse_error", "high",
                            f"HbA1c value '{val}' on {date} is not a valid number"
                        ))
    return flags


def check_hba1c_gaps(patients: list[dict]) -> list[dict]:
    """Flag gaps > 6 months between consecutive HbA1c readings."""
    flags = []
    for p in patients:
        pid = p["patient_id"]
        for t in (p.get("test_list") or []):
            if (t.get("test_name") or "").lower() != "hba1c":
                continue
            readings = t.get("readings") or []
            dates = []
            for r in readings:
                try:
                    d = datetime.strptime(r["date"], "%Y-%m-%d")
                    dates.append((d, r["date"], r.get("value")))
                except (ValueError, KeyError):
                    continue
            dates.sort(key=lambda x: x[0])
            for i in range(1, len(dates)):
                gap_days = (dates[i][0] - dates[i - 1][0]).days
                if gap_days > 180:
                    flags.append(_flag(
                        pid, "hba1c", "readings",
                        "hba1c_gap", "moderate",
                        f"{gap_days}-day gap ({dates[i-1][1]} [{dates[i-1][2]}%] → "
                        f"{dates[i][1]} [{dates[i][2]}%]) — "
                        f"hallucination bait: models may invent values in this window"
                    ))
    return flags


def check_date_age_ordering(patients: list[dict]) -> list[dict]:
    """
    Verify visit_date ordering is consistent with patient_age progression.
    Age should not decrease across visits ordered by date.
    """
    flags = []
    for p in patients:
        pid = p["patient_id"]
        visits = p.get("visit_list", [])
        dated_visits = []
        for v in visits:
            try:
                d = datetime.strptime(v["visit_date"], "%Y-%m-%d")
                age = int(v.get("patient_age", -1))
                dated_visits.append((d, age, v["visit_id"]))
            except (ValueError, KeyError, TypeError):
                continue
        dated_visits.sort(key=lambda x: x[0])
        for i in range(1, len(dated_visits)):
            if dated_visits[i][1] < dated_visits[i - 1][1]:
                flags.append(_flag(
                    pid, dated_visits[i][2], "visit_date → patient_age",
                    "date_age_ordering", "high",
                    f"Age decreased from {dated_visits[i-1][1]} "
                    f"({dated_visits[i-1][2]}) to {dated_visits[i][1]} "
                    f"({dated_visits[i][2]}) despite later visit date"
                ))
    return flags


def check_duplicate_visits(patients: list[dict]) -> list[dict]:
    """Check for duplicate visit_id values within a patient."""
    flags = []
    for p in patients:
        pid = p["patient_id"]
        seen = {}
        for v in p.get("visit_list", []):
            vid = v.get("visit_id", "?")
            if vid in seen:
                flags.append(_flag(
                    pid, vid, "visit_id",
                    "duplicate_visit", "high",
                    f"visit_id '{vid}' appears more than once for patient {pid}"
                ))
            seen[vid] = True
    return flags


def check_medication_status_traps(patients: list[dict]) -> list[dict]:
    """
    Flag medications with 'entry and exit' status — these were started AND
    discontinued at the same visit. A model that lists them as 'ongoing' is wrong.
    """
    flags = []
    for p in patients:
        pid = p["patient_id"]
        for v in p.get("visit_list", []):
            vid = v.get("visit_id", "?")
            for med in (v.get("medications") or []):
                raw_status = med.get("medicine_status")
                # Handle list-typed medicine_status (e.g. [None, 'continue'])
                if isinstance(raw_status, list):
                    raw_status = next((s for s in raw_status if isinstance(s, str)), "")
                status = (raw_status or "").lower().strip()
                if status == "entry and exit":
                    rx = med.get("rx_med_name", "?")
                    flags.append(_flag(
                        pid, vid, f"medications ({rx})",
                        "med_status_trap", "high",
                        f'"{rx}" has medicine_status="entry and exit" — started '
                        f'AND discontinued at this visit. Listing it as ongoing '
                        f'after this visit would be a grounding error.'
                    ))
    return flags


def check_raw_vs_structured_diagnosis_count(patients: list[dict]) -> list[dict]:
    """
    Compare count of raw_diagnosis items vs structured diagnoses.
    Flag discrepancies — may indicate dropped or fabricated entries.
    Also flag specific raw items not found in structured diagnoses.
    """
    flags = []
    for p in patients:
        pid = p["patient_id"]
        for v in p.get("visit_list", []):
            vid = v.get("visit_id", "?")
            raw_dx = v.get("raw_diagnosis")
            if isinstance(raw_dx, str):
                raw_dx = [raw_dx]
            raw_dx = raw_dx or []

            # Filter out junk raw items (pure numbers, very short)
            meaningful_raw = [r for r in raw_dx if isinstance(r, str)
                              and len(r.strip()) > 2
                              and not r.strip().replace(" ", "").isdigit()]

            struct_dx = v.get("diagnoses") or []
            struct_names = [d.get("diagnosis", "").lower() for d in struct_dx]

            # Check for raw items that are clearly not in structured
            for raw_item in meaningful_raw:
                raw_lower = raw_item.lower().strip()
                # Skip if it's a procedure/device (not a diagnosis)
                if any(kw in raw_lower for kw in ["implanted", "surgery", "post"]):
                    continue
                # Check direct match or abbreviation match
                found = False
                for sn in struct_names:
                    if raw_lower in sn or sn in raw_lower:
                        found = True
                        break
                    # Check first-letter acronym
                    raw_initials = "".join(
                        w[0] for w in raw_lower.split() if w
                    )
                    struct_initials = "".join(
                        w[0] for w in sn.split() if w
                    )
                    if (len(raw_initials) >= 2 and
                            (raw_initials == struct_initials
                             or raw_lower == struct_initials
                             or raw_initials == sn.replace(" ", ""))):
                        found = True
                        break
                # Don't flag — this is complex and generates too many false positives
                # We only flag count discrepancies

            raw_count = len(meaningful_raw)
            struct_count = len(struct_dx)
            if abs(raw_count - struct_count) >= 2:
                flags.append(_flag(
                    pid, vid, "raw_diagnosis vs diagnoses",
                    "dx_count_mismatch", "info",
                    f"{raw_count} meaningful raw_diagnosis items vs "
                    f"{struct_count} structured diagnoses (diff={raw_count - struct_count}). "
                    f"Raw: {meaningful_raw}"
                ))
    return flags


def check_is_negated_signal(patients: list[dict]) -> list[dict]:
    """
    Check if is_negated is ever true anywhere across the entire dataset.
    If never true, flag the entire field as structurally broken.
    """
    total_dx = 0
    any_true = False
    for p in patients:
        for v in p.get("visit_list", []):
            for dx in (v.get("diagnoses") or []):
                total_dx += 1
                if dx.get("is_negated") is True:
                    any_true = True

    flags = []
    if not any_true and total_dx > 0:
        flags.append(_flag(
            "ALL", "ALL", "diagnoses.is_negated",
            "is_negated_never_true", "high",
            f"is_negated is false for ALL {total_dx} diagnosis entries "
            f"across the entire dataset — this field is structurally broken "
            f"or unused. Do NOT rely on it for negation detection."
        ))
    return flags


def check_hba1c_units(patients: list[dict]) -> list[dict]:
    """
    Check HbA1c units field. The clinical standard is % (NGSP) or
    mmol/mol (IFCC). Flag if units are empty, mixed, or unexpected.
    """
    flags = []
    for p in patients:
        pid = p["patient_id"]
        for t in (p.get("test_list") or []):
            if (t.get("test_name") or "").lower() != "hba1c":
                continue
            units_seen = set()
            for r in (t.get("readings") or []):
                u = r.get("units", r.get("unit", ""))
                units_seen.add(str(u).strip())

            if units_seen == {""} or units_seen == set():
                flags.append(_flag(
                    pid, "hba1c", "units",
                    "hba1c_units_empty", "info",
                    f"All HbA1c readings have empty units field. "
                    f"Based on value range, unit is implicitly % (NGSP). "
                    f"A model might cite the empty field or invent a different unit."
                ))
    return flags


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_negation_mismatch,
    check_null_structured_with_raw,
    check_diagnosis_medication_coherence,
    check_hba1c_range,
    check_hba1c_gaps,
    check_date_age_ordering,
    check_duplicate_visits,
    check_medication_status_traps,
    check_raw_vs_structured_diagnosis_count,
    check_is_negated_signal,
    check_hba1c_units,
]


def run_all_checks(patients: list[dict]) -> list[dict]:
    """Run all checks and return a flat list of flags."""
    all_flags = []
    for check_fn in ALL_CHECKS:
        try:
            all_flags.extend(check_fn(patients))
        except Exception as e:
            all_flags.append(_flag(
                "SYSTEM", "SYSTEM", "checks.py",
                "check_error", "high",
                f"Check {check_fn.__name__} raised an error: {e}"
            ))
    return all_flags


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import os
    import sys

    # Find data file
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "patients_sample_3.json"
    )
    if not os.path.exists(data_path):
        # Try parent directory
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "patients_sample_3.json"
        )
    if not os.path.exists(data_path):
        print(f"Data file not found. Tried: {data_path}")
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as f:
        patients = json.load(f)

    flags = run_all_checks(patients)
    print(f"\n{'='*70}")
    print(f"Total flags: {len(flags)}")
    print(f"{'='*70}")

    # Group by severity
    for sev in ("high", "moderate", "info"):
        sev_flags = [f for f in flags if f["severity"] == sev]
        if sev_flags:
            print(f"\n--- {sev.upper()} ({len(sev_flags)}) ---")
            for f in sev_flags:
                print(f"  [{f['patient_id']}/{f['item_id']}] "
                      f"{f['check_name']}: {f['detail']}")
