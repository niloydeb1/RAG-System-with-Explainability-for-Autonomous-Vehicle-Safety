from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


INPUT_PRE = "clean_pre_mid_2025_reports.csv"
INPUT_POST = "clean_post_mid_2025_reports.csv"
OUTPUT_DIR = Path("merged_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def is_missing(x: Any) -> bool:
    if pd.isna(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def clean_text(x: Any) -> Optional[str]:
    if is_missing(x):
        return None
    if isinstance(x, str):
        x = x.strip()
        return x if x != "" else None
    return str(x)


def to_int(x: Any) -> Optional[int]:
    if is_missing(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def to_float(x: Any) -> Optional[float]:
    if is_missing(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def year_month_from_cleaned_date(x: Any) -> Optional[str]:
    """
    Stage-2 cleaned files store date-like values as YYYY-MM-DD strings.
    For SGO public data, these should be treated as month-precision.
    So we keep only YYYY-MM here.
    """
    s = clean_text(x)
    if s is None:
        return None

    if len(s) >= 7 and s[4] == "-":
        return s[:7]

    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m")
    except Exception:
        return None


def yes_flag(x: Any) -> bool:
    s = clean_text(x)
    if s is None:
        return False
    return s.lower() in {"y", "yes", "true", "1"}


def normalize_binaryish(x: Any) -> Optional[str]:
    """
    Normalize values like Y/Yes/No/Unknown to compact values.
    Keeps narrative-qualified values when present.
    """
    s = clean_text(x)
    if s is None:
        return None

    low = s.lower()
    if low in {"y", "yes", "true", "1"}:
        return "Yes"
    if low in {"n", "no", "false", "0"}:
        return "No"
    if "unknown" in low:
        return "Unknown"
    if "no, see narrative" in low:
        return "No, see Narrative"
    if "yes, see narrative" in low:
        return "Yes, see Narrative"
    return s


def choose_first(row: pd.Series, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in row.index:
            v = clean_text(row[c])
            if v is not None:
                return v
    return None


def collect_flags(row: pd.Series, mapping: Dict[str, str]) -> List[str]:
    out = []
    for col, label in mapping.items():
        if col in row.index and yes_flag(row[col]):
            out.append(label)
    return out


def combine_boolish_any(values: List[Any]) -> Optional[str]:
    """
    For 'any airbags deployed' or 'any vehicle towed':
    - Yes if any Yes
    - Unknown if none Yes and any Unknown
    - No if at least one explicit value and all explicit values are No
    - None if all missing
    """
    normed = [normalize_binaryish(v) for v in values if normalize_binaryish(v) is not None]
    if not normed:
        return None
    if any(v == "Yes" for v in normed):
        return "Yes"
    if any(v == "Unknown" for v in normed):
        return "Unknown"
    if all(v == "No" for v in normed):
        return "No"
    return None


def narrative_redacted(row: pd.Series) -> Optional[bool]:
    flag = choose_first(row, ["Narrative - CBI?", "Narrative - CBI"])
    text = choose_first(row, ["Narrative"])

    if flag is not None:
        return yes_flag(flag)

    if text is None:
        return None

    low = text.lower()
    if "[redacted" in low or "may contain confidential business information" in low:
        return True
    return False


WEATHER_MAPPING = {
    "Weather - Clear": "clear",
    "Weather - Snow": "snow",
    "Weather - Cloudy": "cloudy",
    "Weather - Partly Cloudy": "partly_cloudy",
    "Weather - Fog/Smoke": "fog_smoke",
    "Weather - Fog/Smoke/Haze": "fog_smoke_haze",
    "Weather - Rain": "rain",
    "Weather - Severe Wind": "severe_wind",
    "Weather - Dust Storm": "dust_storm",
    "Weather - Severe Hurricane": "severe_hurricane",
    "Weather - Structure-Indoor": "structure_indoor",
    "Weather - Other": "other",
    "Weather - Unknown": "unknown",
    "Weather - Unk - See Narrative": "unknown_see_narrative",
}

ROADWAY_FLAG_MAPPING = {
    "Roadway-Degraded Surface": "degraded_surface",
    "Roadway-Missing/Degraded Marking": "missing_or_degraded_marking",
    "Roadway-No Unusual Conditions": "no_unusual_conditions",
    "Roadway-Other-See Narrative": "other_see_narrative",
    "Roadway-Traffic Incident": "traffic_incident",
    "Roadway-Unknown": "unknown",
    "Roadway-Wet Surface Condition": "wet_surface_condition",
    "Roadway-Work Zone": "work_zone",
}

NOTICE_SOURCE_MAPPING = {
    "Source - Complaint/Claim": "complaint_claim",
    "Source - Telematics": "telematics",
    "Source - Law Enforcement": "law_enforcement",
    "Source - Field Report": "field_report",
    "Source - Testing": "testing",
    "Source - Media": "media",
    "Source - Other": "other",
    "Source - Other Entity": "other_entity",
    "Source - State or Other Agency": "state_or_other_agency",
    "Source - Internal Process Review": "internal_process_review",
    "Source - NHTSA VOQ": "nhtsa_voq",
    "Source - Other - See Narrative": "other_see_narrative",
}

EVIDENCE_MAPPING = {
    "Data Availability - EDR": "edr",
    "Data Availability - Police Rpt": "police_report",
    "Data Availability - Telematics": "telematics",
    "Data Availability - Complaints": "complaints",
    "Data Availability - Video": "video",
    "Data Availability - Other": "other",
    "Data Availability - No Data": "no_data",
    "Data Availability - Unknown": "unknown",
}

CP_CONTACT_MAPPING = {
    "CP Contact Area - Rear Left": "rear_left",
    "CP Contact Area - Left": "left",
    "CP Contact Area - Front Left": "front_left",
    "CP Contact Area - Rear": "rear",
    "CP Contact Area - Top": "top",
    "CP Contact Area - Front": "front",
    "CP Contact Area - Rear Right": "rear_right",
    "CP Contact Area - Right": "right",
    "CP Contact Area - Front Right": "front_right",
    "CP Contact Area - Bottom": "bottom",
    "CP Contact Area - Unknown": "unknown",
}

SV_CONTACT_MAPPING = {
    "SV Contact Area - Rear Left": "rear_left",
    "SV Contact Area - Left": "left",
    "SV Contact Area - Front Left": "front_left",
    "SV Contact Area - Rear": "rear",
    "SV Contact Area - Top": "top",
    "SV Contact Area - Front": "front",
    "SV Contact Area - Rear Right": "rear_right",
    "SV Contact Area - Right": "right",
    "SV Contact Area - Front Right": "front_right",
    "SV Contact Area - Bottom": "bottom",
    "SV Contact Area - Unknown": "unknown",
}


def row_to_canonical(row: pd.Series) -> Dict[str, Any]:
    weather = collect_flags(row, WEATHER_MAPPING)
    roadway_flags = collect_flags(row, ROADWAY_FLAG_MAPPING)
    notice_sources = collect_flags(row, NOTICE_SOURCE_MAPPING)
    evidence = collect_flags(row, EVIDENCE_MAPPING)
    cp_contact = collect_flags(row, CP_CONTACT_MAPPING)
    sv_contact = collect_flags(row, SV_CONTACT_MAPPING)

    old_airbags = combine_boolish_any([
        row["CP Any Air Bags Deployed?"] if "CP Any Air Bags Deployed?" in row.index else None,
        row["SV Any Air Bags Deployed?"] if "SV Any Air Bags Deployed?" in row.index else None,
    ])

    old_towed = combine_boolish_any([
        row["CP Was Vehicle Towed?"] if "CP Was Vehicle Towed?" in row.index else None,
        row["SV Was Vehicle Towed?"] if "SV Was Vehicle Towed?" in row.index else None,
    ])

    any_airbags = normalize_binaryish(
        choose_first(row, ["Any Air Bags Deployed?"])
    ) or old_airbags

    any_towed = normalize_binaryish(
        choose_first(row, ["Was Any Vehicle Towed?"])
    ) or old_towed

    belted = normalize_binaryish(
        choose_first(row, ["Were All Passengers Belted?", "SV Were All Passengers Belted?"])
    )

    out = {
        # provenance
        "schema_version": choose_first(row, ["schema_version"]),
        "source_file": choose_first(row, ["source_file"]),
        "report_id": choose_first(row, ["Report ID"]),
        "report_version": to_int(row["Report Version"]) if "Report Version" in row.index else None,
        "reporting_entity": choose_first(row, ["Reporting Entity"]),
        "report_type": choose_first(row, ["Report Type"]),
        "report_month": to_int(row["Report Month"]) if "Report Month" in row.index else None,
        "report_year": to_int(row["Report Year"]) if "Report Year" in row.index else None,
        "report_submission_ym": year_month_from_cleaned_date(choose_first(row, ["Report Submission Date"])),
        "notice_received_ym": year_month_from_cleaned_date(choose_first(row, ["Notice Received Date"])),

        # identifiers
        "same_incident_id": choose_first(row, ["Same Incident ID"]),
        "same_vehicle_id": choose_first(row, ["Same Vehicle ID"]),

        # vehicle / operator
        "vehicle_make": choose_first(row, ["Make"]),
        "vehicle_model": choose_first(row, ["Model"]),
        "vehicle_model_year": to_int(row["Model Year"]) if "Model Year" in row.index else None,
        "mileage": to_float(row["Mileage"]) if "Mileage" in row.index else None,
        "driver_operator_type": choose_first(row, ["Driver / Operator Type"]),
        "operating_entity": choose_first(row, ["Operating Entity"]),

        # automation
        "automation_type_engaged": choose_first(row, ["Automation System Engaged?"]),
        "engagement_status": choose_first(row, ["Engagement Status"]),
        "within_odd": choose_first(row, ["Within ODD?"]),
        "automation_feature_version": choose_first(row, ["Automation Feature Version", "ADAS/ADS System Version"]),
        "automation_system_version": choose_first(row, ["ADAS/ADS System Version"]),
        "automation_hardware_version": choose_first(row, ["ADAS/ADS Hardware Version"]),
        "automation_software_version": choose_first(row, ["ADAS/ADS Software Version"]),
        "ads_equipped": normalize_binaryish(choose_first(row, ["ADS Equipped?"])),

        # time/location
        "incident_ym": year_month_from_cleaned_date(choose_first(row, ["Incident Date"])),
        "incident_time_local": choose_first(row, ["Incident Time (24:00)"]),
        "city": choose_first(row, ["City"]),
        "state": choose_first(row, ["State"]),
        "latitude": choose_first(row, ["Latitude"]),
        "longitude": choose_first(row, ["Longitude"]),

        # roadway/weather
        "roadway_type": choose_first(row, ["Roadway Type"]),
        "roadway_surface_text": choose_first(row, ["Roadway Surface"]),
        "roadway_description_text": choose_first(row, ["Roadway Description"]),
        "roadway_conditions_json": json.dumps(sorted(set(roadway_flags))) if roadway_flags else None,
        "lighting": choose_first(row, ["Lighting"]),
        "posted_speed_limit_mph": to_float(row["Posted Speed Limit (MPH)"]) if "Posted Speed Limit (MPH)" in row.index else None,
        "weather_conditions_json": json.dumps(sorted(set(weather))) if weather else None,
        "weather_other_text": choose_first(row, ["Weather - Other Text"]),

        # crash mechanics / outcomes
        "crash_with": choose_first(row, ["Crash With"]),
        "injury_severity": choose_first(row, ["Highest Injury Severity Alleged"]),
        "property_damage": normalize_binaryish(choose_first(row, ["Property Damage?"])),
        "cp_pre_crash_movement": choose_first(row, ["CP Pre-Crash Movement"]),
        "sv_pre_crash_movement": choose_first(row, ["SV Pre-Crash Movement"]),
        "sv_precrash_speed_mph": to_float(row["SV Precrash Speed (MPH)"]) if "SV Precrash Speed (MPH)" in row.index else None,
        "cp_contact_areas_json": json.dumps(sorted(set(cp_contact))) if cp_contact else None,
        "sv_contact_areas_json": json.dumps(sorted(set(sv_contact))) if sv_contact else None,
        "any_airbags_deployed": any_airbags,
        "any_vehicle_towed": any_towed,
        "all_passengers_belted": belted,

        # provenance / evidence
        "notice_sources_json": json.dumps(sorted(set(notice_sources))) if notice_sources else None,
        "available_evidence_json": json.dumps(sorted(set(evidence))) if evidence else None,
        "investigating_agency": choose_first(row, ["Investigating Agency"]),
        "law_enforcement_investigating": normalize_binaryish(choose_first(row, ["Law Enforcement Investigating?"])),
        "rep_ent_or_mfr_investigating": choose_first(row, ["Rep Ent Or Mfr Investigating?"]),

        # narrative
        "narrative": choose_first(row, ["Narrative"]),
        "narrative_redacted": narrative_redacted(row),
    }
    return out


def harmonize_df(df: pd.DataFrame) -> pd.DataFrame:
    records = [row_to_canonical(row) for _, row in df.iterrows()]
    out = pd.DataFrame(records).astype(object)

    cols = [
        "schema_version", "source_file", "report_id", "report_version", "reporting_entity",
        "report_type", "report_month", "report_year", "report_submission_ym", "notice_received_ym",
        "same_incident_id", "same_vehicle_id",
        "vehicle_make", "vehicle_model", "vehicle_model_year", "mileage",
        "driver_operator_type", "operating_entity",
        "automation_type_engaged", "engagement_status", "within_odd",
        "automation_feature_version", "automation_system_version",
        "automation_hardware_version", "automation_software_version", "ads_equipped",
        "incident_ym", "incident_time_local", "city", "state", "latitude", "longitude",
        "roadway_type", "roadway_surface_text", "roadway_description_text",
        "roadway_conditions_json", "lighting", "posted_speed_limit_mph",
        "weather_conditions_json", "weather_other_text",
        "crash_with", "injury_severity", "property_damage",
        "cp_pre_crash_movement", "sv_pre_crash_movement", "sv_precrash_speed_mph",
        "cp_contact_areas_json", "sv_contact_areas_json",
        "any_airbags_deployed", "any_vehicle_towed", "all_passengers_belted",
        "notice_sources_json", "available_evidence_json",
        "investigating_agency", "law_enforcement_investigating", "rep_ent_or_mfr_investigating",
        "narrative", "narrative_redacted",
    ]
    return out[cols]


def stage4_keep_latest_report_id(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["_sort_report_version"] = pd.to_numeric(x["report_version"], errors="coerce")
    x["_sort_submission"] = x["report_submission_ym"].fillna("")
    x["_row_order"] = range(len(x))

    x = x.sort_values(
        by=["report_id", "_sort_report_version", "_sort_submission", "_row_order"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )

    x = x.drop_duplicates(subset=["report_id"], keep="last").copy()
    x = x.drop(columns=["_sort_report_version", "_sort_submission", "_row_order"])
    return x


def main():
    pre = pd.read_csv(INPUT_PRE, dtype=object)
    post = pd.read_csv(INPUT_POST, dtype=object)

    harm_pre = harmonize_df(pre)
    harm_post = harmonize_df(post)

    harmonized = pd.concat([harm_pre, harm_post], ignore_index=True)

    harmonized_out = OUTPUT_DIR / "harmonized_reports.csv"
    harmonized.to_csv(harmonized_out, index=False)

    harmonized_latest = stage4_keep_latest_report_id(harmonized)
    harmonized_latest_out = OUTPUT_DIR / "harmonized_reports_latest.csv"
    harmonized_latest.to_csv(harmonized_latest_out, index=False)

    summary = {
        "rows_pre_input": int(len(pre)),
        "rows_post_input": int(len(post)),
        "rows_harmonized": int(len(harmonized)),
        "rows_harmonized_latest": int(len(harmonized_latest)),
        "duplicate_report_id_rows_before_combining": int(harmonized["report_id"].duplicated(keep=False).sum()),
        "duplicate_report_id_rows_after_combining": int(harmonized_latest["report_id"].duplicated(keep=False).sum()),
        "unique_same_incident_id_harmonized": int(harmonized["same_incident_id"].nunique(dropna=True)),
        "schema_counts_harmonized": harmonized["schema_version"].value_counts(dropna=False).to_dict(),
    }

    with open(OUTPUT_DIR / "stage3_stage4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] wrote", harmonized_out)
    print("[OK] wrote", harmonized_latest_out)
    print("[OK] wrote", OUTPUT_DIR / "harmonized_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()