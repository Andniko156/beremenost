"""Pregnancy research series keep subject, visit and relative time explicit."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

HRV29_URL = "https://zenodo.org/api/records/7102622/files/maternal_ECG_cortisol_data_FINAL.xlsx/content"
HRV29_SHA256 = "bed2512b7b99c9a4c69f4c6fe61e3a698abe53208fe36e602755edfca4b3ac8a"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ResearchProfile:
    interval_minutes: int = 5
    context_steps: int = 6
    prediction_steps: int = 12

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.context_steps < 2 or self.context_steps > 8192 or self.prediction_steps > 1024:
            raise ValueError("Unsupported context or prediction length")


def subject_split(subjects, seed=42):
    subjects = sorted(set(map(str, subjects)))
    if len(subjects) < 7:
        raise ValueError("At least seven eligible subjects are required for three subject-disjoint splits")
    order = np.random.default_rng(seed).permutation(subjects).tolist()
    n_test = max(1, round(len(order) * 0.17))
    n_val = max(1, round(len(order) * 0.14))
    return {"train": sorted(order[n_test + n_val:]),
            "validation": sorted(order[n_test:n_test + n_val]), "test": sorted(order[:n_test])}


def segment_visits(visits):
    """Missing/invalid observations split a visit; no interpolation or invented clock dates."""
    records = []
    audit = {"input_rows": 0, "excluded_invalid_hr_rows": 0, "segments": 0}
    required = {"participant_id", "Data_collection_window", "hr_ibi"}
    for visit, frame in visits.items():
        if not required.issubset(frame.columns):
            raise ValueError("Missing participant, relative window or heart-rate column")
        if frame[list(required - {"hr_ibi"})].isna().any().any():
            raise ValueError("Subject IDs and window indices must be present")
        frame = frame.copy()
        frame["participant_id"] = frame.participant_id.astype(str)
        indices = pd.to_numeric(frame.Data_collection_window, errors="raise")
        if not np.isfinite(indices).all() or (indices < 0).any() or not (indices % 1 == 0).all():
            raise ValueError("Window indices must be nonnegative integers")
        frame["Data_collection_window"] = indices.astype(int)
        if frame.duplicated(["participant_id", "Data_collection_window"]).any():
            raise ValueError("Duplicate subject/visit/window; resolve at source")
        audit["input_rows"] += len(frame)
        for subject, group in frame.groupby("participant_id", sort=True):
            group = group.sort_values("Data_collection_window")
            values = pd.to_numeric(group.hr_ibi, errors="coerce")
            valid = np.isfinite(values) & (values > 0)
            audit["excluded_invalid_hr_rows"] += int((~valid).sum())
            group = group.loc[valid].copy()
            group["hr_ibi"] = values.loc[valid]
            run_ids = group.Data_collection_window.diff().ne(1).cumsum()
            for run, part in group.groupby(run_ids, sort=True):
                records.append({"subject_id": subject, "recording_id": str(visit),
                                "segment_id": f"{subject}/{visit}/{run}",
                                "first_window": int(part.Data_collection_window.iloc[0]),
                                "values": part.hr_ibi.astype(float).tolist()})
    audit["segments"] = len(records)
    return records, audit


def save_prepared(records, output_dir, profile, provenance, seed=42, audit=None):
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Prepared output directory must be new or empty")
    required = profile.context_steps + profile.prediction_steps
    eligible = [row for row in records if len(row["values"]) >= required]
    splits = subject_split([row["subject_id"] for row in eligible], seed)
    manifest = {"schema_version": 1, "signal": "maternal_heart_rate", "unit": "bpm",
                "population": "pregnant", "profile": asdict(profile), "seed": seed,
                "subject_namespace": provenance.get("subject_namespace", provenance.get("dataset_id", "unknown")),
                "source": provenance, "splits": splits, "clinical_validity": "not_evaluated",
                "audit": dict(audit or {}, eligible_segments=len(eligible),
                              excluded_short_segments=len(records) - len(eligible)),
                "time_axis": "relative source window index, no calendar timestamps inferred"}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "segments.json", eligible)
    manifest["segments_sha256"] = sha256(out / "segments.json")
    write_json(out / "manifest.json", manifest)
    load_prepared(out)  # Validate the same contract used at training time.
    return manifest


def prepare_hrv29(output_dir, input_path=None, seed=42, profile=None):
    profile = profile or ResearchProfile()
    if profile.interval_minutes != 5:
        raise ValueError("HRV29 source windows are five minutes; resampling is not implicit")
    if input_path is None:
        target = Path(output_dir).parent / "raw" / "maternal_ECG_cortisol_data_FINAL.xlsx"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with urllib.request.urlopen(HRV29_URL, timeout=60) as response:
                target.write_bytes(response.read(2_000_001))
        input_path = target
    if sha256(input_path) != HRV29_SHA256:
        raise ValueError("HRV29 workbook checksum differs from the reviewed public version")
    workbook = pd.ExcelFile(input_path, engine="openpyxl")
    visits = {name: pd.read_excel(workbook, sheet_name=name)
              for name in ("merged_visit1", "merged_visit2")}
    records, audit = segment_visits(visits)
    return save_prepared(records, output_dir, profile, {
        "dataset_id": "hrv29_zenodo_7102622", "url": HRV29_URL,
        "doi": "10.7280/D12Q4P", "license": "CC0-1.0", "sha256": HRV29_SHA256,
        "measurement": "ECG-derived HR aggregated in sequential five-minute source windows",
        "limitations": ["29 participants; 28-32 weeks of pregnancy",
                        "Only non-stress visits are long enough for the default one-hour target",
                        "No raw ECG, signal-quality labels, device-specific validation or clinical outcomes",
                        "Cortisol, HRV and stress labels are not model inputs"]}, seed, audit)


def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("signal") != "maternal_heart_rate"
            or manifest.get("unit") != "bpm" or manifest.get("population") != "pregnant"):
        raise ValueError("Expected schema v1, maternal heart rate in bpm, pregnant population")
    profile = ResearchProfile(**manifest["profile"])
    if sha256(directory / "segments.json") != manifest.get("segments_sha256"):
        raise ValueError("Prepared segments checksum mismatch")
    records = json.loads((directory / "segments.json").read_text(encoding="utf-8"))
    splits = manifest["splits"]
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Three explicit splits are required")
    all_subjects = []
    for values in splits.values():
        if not values or not all(isinstance(s, str) and s for s in values):
            raise ValueError("Each split needs explicit subject IDs")
        all_subjects.extend(values)
    if len(all_subjects) != len(set(all_subjects)):
        raise ValueError("Subject overlap across train/validation/test")
    ids = set()
    observed = set()
    for record in records:
        sid = record["segment_id"]
        if sid in ids:
            raise ValueError("Duplicate segment ID")
        ids.add(sid)
        observed.add(record["subject_id"])
        values = np.asarray(record["values"], dtype=float)
        if (values.ndim != 1 or len(values) < profile.context_steps + profile.prediction_steps
                or not np.isfinite(values).all() or (values <= 0).any()):
            raise ValueError("Segment must contain enough finite positive observations")
        if type(record["first_window"]) is not int or record["first_window"] < 0:
            raise ValueError("Invalid relative time origin")
    if observed != set(all_subjects):
        raise ValueError("Split IDs do not match eligible segment subjects")
    return manifest, records, profile


def check_parent_holdouts(manifest, lineage):
    """Keep former test women out of learning, including validation-based selection."""
    namespace = manifest.get("subject_namespace", manifest["source"].get("dataset_id", "unknown"))
    train = set(manifest["splits"]["train"])
    validation = set(manifest["splits"]["validation"])
    for previous in lineage.get("reserved_holdouts", []):
        if previous["subject_namespace"] != namespace:
            continue
        if train.intersection(previous["validation"]) or (train | validation).intersection(previous["test"]):
            raise ValueError("Former held-out women cannot be reused for training or test-guided selection")


def training_windows(records, subjects, profile):
    size = profile.context_steps + profile.prediction_steps
    return [np.asarray(row["values"][i:i + size], dtype=np.float32)
            for row in records if row["subject_id"] in subjects
            for i in range(len(row["values"]) - size + 1)]


def evaluation_windows(records, subjects, profile):
    """Last nonoverlapping target block per segment; held-out women only."""
    size = profile.context_steps + profile.prediction_steps
    return [dict(row, values=np.asarray(row["values"][-size:], dtype=np.float32),
                 origin_window=row["first_window"] + len(row["values"]) - profile.prediction_steps - 1)
            for row in records if row["subject_id"] in subjects and len(row["values"]) >= size]
