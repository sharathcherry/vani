"""
add_text_for_embedding.py
--------------------------
Batch-adds a `text_for_embedding` field to every JSON file in `structured_schemes/`.

The field is a pre-built natural-language block combining all semantically
meaningful fields — optimised for vector embedding (RAG retrieval).

Usage:
    python add_text_for_embedding.py

Options (edit constants below):
    STRUCTURED_DIR  : folder containing your scheme JSONs
    OVERWRITE       : if True, regenerates text_for_embedding even if it already exists
    DRY_RUN         : if True, prints output without writing files (safe preview)
"""

import os
import json
import glob
from pathlib import Path

# ==========================
# CONFIG
# ==========================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
STRUCTURED_DIR  = os.path.join(BASE_DIR, "structured_schemes")
OVERWRITE       = True   # set False to skip files that already have the field
DRY_RUN         = False  # set True to preview without writing


# ==========================
# TEXT BUILDER
# ==========================

def build_text_for_embedding(scheme: dict) -> str:
    """
    Constructs a rich, human-readable paragraph from structured JSON fields.
    Designed to maximise semantic coverage for vector embedding.
    Explicitly states "no requirement" for missing/empty fields to be informative.
    """
    parts = []

    # --- Identity ---
    name = scheme.get("scheme_name") or ""
    category = scheme.get("scheme_category") or ""
    state = scheme.get("state_or_ut") or ""
    ministry = scheme.get("ministry_or_department") or ""

    if name:
        parts.append(f"Scheme: {name}.")
    else:
        parts.append("Scheme name is not specified.")

    if category:
        parts.append(f"Category: {category}.")
    else:
        parts.append("Category is not specified.")

    if ministry:
        parts.append(f"Managed by: {ministry}.")
    else:
        parts.append("Managing ministry or department is not specified.")

    if state:
        parts.append(f"It is a State scheme applicable in {state}.")
    else:
        parts.append("It is a Central Government scheme applicable across India.")

    # --- Objective ---
    objective = scheme.get("objective") or ""
    if objective:
        parts.append(f"Objective: {objective}.")
    else:
        parts.append("The objective of the scheme is not explicitly stated.")

    # --- Beneficiaries ---
    beneficiaries = scheme.get("target_beneficiaries") or []
    tags = scheme.get("beneficiary_tags") or []
    all_bene = list(dict.fromkeys(beneficiaries + tags))  # deduplicated, order preserved
    if all_bene:
        parts.append(f"Target beneficiaries: {', '.join(all_bene)}.")
    else:
        parts.append("Target beneficiaries are not specifically defined, implying broad applicability.")

    # --- Eligibility Criteria ---
    criteria_parts = []

    # Age
    age = scheme.get("age_criteria") or {}
    min_age = age.get("min_age")
    max_age = age.get("max_age")
    if min_age is not None and max_age is not None:
        criteria_parts.append(f"age between {min_age} and {max_age} years")
    elif min_age is not None:
        criteria_parts.append(f"age at least {min_age} years")
    elif max_age is not None:
        criteria_parts.append(f"age up to {max_age} years")
    else:
        criteria_parts.append("no specific age requirement")

    # Income
    income = scheme.get("income_criteria") or {}
    max_income = income.get("max_income_inr")
    income_cat = income.get("income_category")
    if max_income:
        criteria_parts.append(f"annual income below ₹{max_income:,}")
    if income_cat:
        criteria_parts.append(f"income category: {income_cat}")
    if not max_income and not income_cat:
        criteria_parts.append("no specific income requirement")

    # Residency
    res = scheme.get("residency_requirement") or {}
    if res.get("required") and res.get("location"):
        criteria_parts.append(f"must be a resident of {res['location']}")
    else:
        criteria_parts.append("no specific residency requirement")

    # Occupation
    occupation = scheme.get("occupation_criteria") or []
    if occupation:
        criteria_parts.append(f"occupation: {', '.join(occupation)}")
    else:
        criteria_parts.append("no specific occupation requirement")

    # Caste
    caste = scheme.get("caste_criteria") or []
    if caste:
        criteria_parts.append(f"applicable to {', '.join(caste)}")
    else:
        criteria_parts.append("no specific caste requirement")

    # Gender
    gender = scheme.get("gender_criteria") or []
    if gender:
        criteria_parts.append(f"gender: {', '.join(gender)}")
    else:
        criteria_parts.append("no specific gender requirement")

    # Disability
    disability = scheme.get("disability_required")
    if disability is True:
        criteria_parts.append("applicant must have a disability")
    else:
        criteria_parts.append("no specific disability requirement")

    parts.append("Eligibility criteria: " + "; ".join(criteria_parts) + ".")

    # --- Enterprise Criteria ---
    enterprise = scheme.get("enterprise_criteria") or {}
    ent_types = enterprise.get("eligible_enterprise_types") or []
    max_invest = enterprise.get("max_investment_limit_inr")

    if ent_types:
        parts.append(f"Eligible enterprise types: {', '.join(ent_types)}.")
    else:
        parts.append("No specific enterprise type requirement.")

    if max_invest:
        parts.append(f"Maximum investment limit: ₹{max_invest:,}.")
    else:
        parts.append("No specific maximum investment limit for enterprises.")

    # --- Benefit Details ---
    benefit = scheme.get("benefit_details") or {}
    benefit_type = benefit.get("benefit_type") or ""
    subsidy_pct = benefit.get("subsidy_percentage")
    max_amt = benefit.get("max_amount_inr")
    min_amt = benefit.get("min_amount_inr")
    frequency = benefit.get("frequency") or ""
    duration = benefit.get("duration") or ""

    benefit_desc = []
    if benefit_type:
        benefit_desc.append(f"benefit type is {benefit_type}")
    if subsidy_pct is not None:
        benefit_desc.append(f"subsidy of {subsidy_pct}%")
    if max_amt and min_amt:
        benefit_desc.append(f"amount ranging from ₹{min_amt:,} to ₹{max_amt:,}")
    elif max_amt:
        benefit_desc.append(f"maximum amount of ₹{max_amt:,}")
    elif min_amt:
        benefit_desc.append(f"minimum amount of ₹{min_amt:,}")
    else:
        benefit_desc.append("no fixed monetary amount specified")

    if frequency:
        benefit_desc.append(f"paid {frequency.lower()}")
    else:
        benefit_desc.append("payment frequency not specified")

    if duration:
        benefit_desc.append(f"for {duration.lower()}")
    else:
        benefit_desc.append("no fixed duration")

    if benefit_desc:
        parts.append("Benefits: " + ", ".join(benefit_desc) + ".")
    else:
        parts.append("Benefit details are not specified.")

    # --- Exclusions ---
    exclusions = scheme.get("exclusions") or []
    if exclusions:
        parts.append(f"Not applicable to: {'; '.join(exclusions)}.")
    else:
        parts.append("No specific exclusions are listed.")

    # --- Application ---
    app_mode = scheme.get("application_mode") or []
    if app_mode:
        parts.append(f"Application mode: {', '.join(app_mode)}.")
    else:
        parts.append("Application mode is not specified.")

    return " ".join(parts)


# ==========================
# BATCH PROCESSOR
# ==========================

def process_all(structured_dir: str, overwrite: bool = True, dry_run: bool = False):
    """
    Walks all JSON files (including subdirectories) in structured_dir,
    builds and injects `text_for_embedding`, and writes back in-place.
    """
    pattern = os.path.join(structured_dir, "**", "*.json")
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f"[WARNING] No JSON files found in: {structured_dir}")
        return

    updated = 0
    skipped = 0
    errors  = 0

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                scheme = json.load(f)

            # Skip if already present and overwrite is False
            if not overwrite and "text_for_embedding" in scheme:
                skipped += 1
                continue

            text = build_text_for_embedding(scheme)

            # Insert text_for_embedding right after scheme_name (first key)
            # so it's visible and easy to inspect
            new_scheme = {}
            for key, val in scheme.items():
                new_scheme[key] = val
                if key == "scheme_name":
                    new_scheme["text_for_embedding"] = text

            # Fallback: append at end if scheme_name wasn't found
            if "text_for_embedding" not in new_scheme:
                new_scheme["text_for_embedding"] = text

            if dry_run:
                rel = os.path.relpath(filepath, structured_dir)
                print(f"\n[DRY RUN] {rel}")
                print(f"  → {text[:200]}{'...' if len(text) > 200 else ''}")
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(new_scheme, f, indent=2, ensure_ascii=False)

            updated += 1

        except Exception as e:
            print(f"[ERROR] {filepath}: {e}")
            errors += 1

    mode_label = "[DRY RUN] Would update" if dry_run else "Updated"
    print(f"\n{'='*50}")
    print(f"  {mode_label} : {updated} files")
    print(f"  Skipped      : {skipped} files (already had field)")
    print(f"  Errors       : {errors} files")
    print(f"{'='*50}")


# ==========================
# ENTRY
# ==========================

if __name__ == "__main__":
    print(f"Processing: {STRUCTURED_DIR}")
    print(f"Overwrite existing: {OVERWRITE}")
    print(f"Dry run: {DRY_RUN}\n")
    process_all(STRUCTURED_DIR, overwrite=OVERWRITE, dry_run=DRY_RUN)
    if not DRY_RUN:
        print("\nDone! All JSONs now contain `text_for_embedding`.")
