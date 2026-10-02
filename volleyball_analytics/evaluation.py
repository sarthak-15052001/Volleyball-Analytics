# -*- coding: utf-8 -*-
"""Phase 0 / Phase 5 — smoke & holdout evaluation reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

EVENT_NAMES = [
    "Serve",
    "Reception",
    "Set",
    "Attack",
    "Block",
    "Dig",
    "Rally",
    "Point",
]


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def analyze_events_csv(events_csv: Path, duration_sec: float = 90.0) -> Dict[str, Any]:
    df = _safe_read_csv(events_csv)
    if df.empty:
        return {
            "path": str(events_csv), "n_events": 0, "events_per_10s": 0.0,
            "event_counts": {}, "n_rallies": 0, "jerseys_filled": 0,
            "ball_conf_median": None, "ball_speed_median": None,
            "gap_median_sec": None, "single_frame_events": 0,
            "flags": ["no_confirmed_events"],
        }

    counts = df["event"].value_counts().to_dict() if "event" in df.columns else {}
    n_rallies = df["rally_id"].nunique() if "rally_id" in df.columns else 0
    jersey_filled = int(df["jersey_number"].notna().sum()) if "jersey_number" in df.columns else 0
    if "jersey_number" in df.columns:
        jersey_filled = int(
            df["jersey_number"].astype(str).str.strip().replace("nan", "").ne("").sum()
        )

    ball_conf = pd.to_numeric(df.get("ball_conf", pd.Series(dtype=float)), errors="coerce")
    speeds = pd.to_numeric(df.get("ball_speed_peak", pd.Series(dtype=float)), errors="coerce")

    same_frame = 0
    if "frame_start" in df.columns and "frame_end" in df.columns:
        same_frame = int((df["frame_start"] == df["frame_end"]).sum())

    gaps = []
    if "time_start_sec" in df.columns:
        times = sorted(pd.to_numeric(df["time_start_sec"], errors="coerce").dropna())
        gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]

    missing_events = sorted(set(EVENT_NAMES) - set(counts.keys()))
    events_per_10s = len(df) / max(duration_sec, 1) * 10

    flags: List[str] = []
    if events_per_10s > 8:
        flags.append("too_many_events_per_10s")
    if n_rallies <= 1 and duration_sec >= 60:
        flags.append("rally_segmentation_stuck")
    if jersey_filled == 0:
        flags.append("no_jerseys_read")
    if missing_events:
        flags.append(f"missing_event_types:{','.join(missing_events)}")
    if same_frame == len(df) and len(df) > 0:
        flags.append("all_single_frame_events")

    return {
        "path": str(events_csv),
        "n_events": len(df),
        "events_per_10s": round(events_per_10s, 2),
        "event_counts": counts,
        "n_rallies": int(n_rallies),
        "jerseys_filled": jersey_filled,
        "ball_conf_median": round(float(ball_conf.median()), 3) if ball_conf.notna().any() else None,
        "ball_speed_median": round(float(speeds.median()), 1) if speeds.notna().any() else None,
        "gap_median_sec": round(float(sorted(gaps)[len(gaps) // 2]), 2) if gaps else None,
        "single_frame_events": same_frame,
        "flags": flags,
    }


def compare_to_ground_truth(
    events_csv: Path,
    ground_truth_csv: Path,
    tolerance_sec: float = 1.5,
) -> Dict[str, Any]:
    pred = _safe_read_csv(events_csv)
    gt = _safe_read_csv(ground_truth_csv)
    if pred.empty or gt.empty:
        return {"error": "pred or ground truth empty"}

    results: Dict[str, Any] = {"by_event": {}, "matched": 0, "missed": 0, "extra": 0}
    pred_times = pred.copy()
    gt_times = gt.copy()
    pred_times["time_start_sec"] = pd.to_numeric(pred_times["time_start_sec"], errors="coerce")
    gt_times["time_start_sec"] = pd.to_numeric(gt_times["time_start_sec"], errors="coerce")

    used_gt = set()
    for _, prow in pred_times.iterrows():
        pe, pt = prow.get("event"), prow["time_start_sec"]
        if pd.isna(pt):
            continue
        match = None
        for gi, grow in gt_times.iterrows():
            if gi in used_gt:
                continue
            if grow.get("event") == pe and abs(grow["time_start_sec"] - pt) <= tolerance_sec:
                match = gi
                break
        if match is not None:
            used_gt.add(match)
            results["matched"] += 1
        else:
            results["extra"] += 1

    results["missed"] = len(gt_times) - len(used_gt)
    for ev in EVENT_NAMES:
        gt_n = int((gt_times["event"] == ev).sum()) if "event" in gt_times.columns else 0
        pr_n = int((pred_times["event"] == ev).sum()) if "event" in pred_times.columns else 0
        results["by_event"][ev] = {"ground_truth": gt_n, "predicted": pr_n}
    return results


def write_evaluation_report(
    events_csv: Path,
    out_json: Path,
    duration_sec: float = 90.0,
    ground_truth_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "events_analysis": analyze_events_csv(events_csv, duration_sec=duration_sec),
    }
    if ground_truth_csv and ground_truth_csv.exists():
        report["ground_truth"] = compare_to_ground_truth(events_csv, ground_truth_csv)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def summarize_holdout_runs(events_dir: Path, out_csv: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(events_dir.glob("*_events.csv")):
        stem = p.stem.replace("_events", "")
        dur = 90.0
        if "smoke" not in stem:
            dur = 90.0
        analysis = analyze_events_csv(p, duration_sec=dur)
        rows.append({"stem": stem, **{k: v for k, v in analysis.items() if k != "path"}})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_csv, index=False)
    return df
