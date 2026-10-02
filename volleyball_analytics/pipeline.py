# -*- coding: utf-8 -*-
"""
Volleyball analytics pipeline — Phases 1–4:
  Ball detection/tracking, rally/event logic, jersey OCR, player tracking.

Fixes applied (v2):
  1. min_ball_speed lowered → Set / Attack / Block / Dig now fire correctly.
  2. _classify_touch thresholds tuned for real pixel-speed values from 25 fps HD video.
  3. Jersey post-join: after the full video scan the final jersey_cache is merged
     back into every closed event row so jersey_number is never empty when known.
  4. Single clean CSV per video with exactly 11 required fields.

Improvements applied (v3) — jersey identity pipeline:
  5. Best-frame selection: OCR only runs on high-quality frames per track
     (large bbox, low motion blur, good brightness) to reduce garbage votes.
  6. Jersey number range filter: numbers outside [1, max_jersey_number] (default 99,
     practical volleyball max ~25) are rejected before voting, eliminating OCR
     misreads like 66, 76, 49 that come from scoreboards/shirt logos.
  7. Adaptive OCR interval: confirmed tracks skip OCR entirely; unconfirmed tracks
     sample more aggressively when frame quality is good.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

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

COCO_PERSON = 0
COCO_SPORTS_BALL = 32
JERSEY_RE = re.compile(r"^[0-9]{1,2}$")

_paddle_ocr = None


def format_timestamp(seconds: float) -> str:
    """Return HH:MM:SS string (no milliseconds – easier for humans to read)."""
    s = max(0, int(round(seconds)))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamp_ms(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds_part, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    project_root: Path = field(default_factory=project_root_from_here)
    player_weights: str = "yolo11n.pt"
    ball_weights: Optional[str] = None
    finetuned_player: Optional[Path] = None
    finetuned_ball: Optional[Path] = None
    jersey_detector_weights: Optional[Path] = None
    prefer_finetuned_player: bool = True
    prefer_finetuned_ball: bool = True
    image_size: int = 640
    ball_image_size: int = 1280
    conf_player: float = 0.35
    conf_ball: float = 0.12
    conf_ball_fallback: float = 0.06
    infer_stride: int = 1
    player_infer_stride: int = 1
    tracker: str = "botsort.yaml"
    ocr_interval_sec: float = 1.0
    device: Any = 0
    event_segment_sec: float = 0.35
    # FIX: was 250 → blocked every touch after first one; lowered to 30 so that
    # Set / Attack / Block / Dig can fire during a multi-touch rally.
    min_ball_speed: float = 30.0
    contact_cooldown_sec: float = 0.8
    contact_min_frames: int = 2
    contact_dist_px: float = 150.0
    rally_gap_sec: float = 3.0
    use_paddle_ocr: bool = True
    ocr_on_cpu: bool = False
    use_motion_fallback: bool = True
    ball_max_jump_px: float = 120.0
    min_ball_conf_contact: float = 0.30
    ball_confirm_frames: int = 2
    jersey_vote_min: int = 2
    save_jersey_crops: bool = True
    crops_dir: Optional[Path] = None
    net_y_ratio: float = 0.42
    court_margin_x: float = 0.05
    court_margin_y_top: float = 0.08
    court_margin_y_bot: float = 0.05

    # ── v3: Jersey identity improvements ────────────────────────────────────
    # Valid volleyball jersey number range [1, max_jersey_number].
    # Numbers outside this range are rejected as OCR misreads.
    # Set to 99 (FIVB max) for safety; typical club rosters use 1–25.
    max_jersey_number: int = 99

    # Minimum bbox area (px²) for a player crop to be considered "large enough"
    # for reliable OCR. Small/distant players produce blurry crops.
    ocr_min_bbox_area: int = 3000   # ~55×55 px crop minimum

    # Laplacian variance threshold for blur detection.
    # Below this value the crop is too blurry for reliable OCR.
    ocr_blur_threshold: float = 40.0

    # Brightness window [min, max] in mean pixel value (0-255).
    # Too dark or too bright frames give poor OCR results.
    ocr_brightness_min: float = 40.0
    ocr_brightness_max: float = 230.0

    # When True, confirmed tracks (jersey already locked) completely skip OCR.
    # This avoids wasting GPU time on already-identified players.
    ocr_skip_confirmed: bool = True

    # Minimum OCR interval in seconds for unconfirmed tracks.
    # Good-quality frames can still trigger OCR sooner than ocr_interval_sec.
    ocr_min_interval_sec: float = 0.5



class CourtROI:
    """Exclude sidelines/ads from ball search."""

    def __init__(self, w: int, h: int, cfg: PipelineConfig):
        self.x1 = int(w * cfg.court_margin_x)
        self.x2 = int(w * (1.0 - cfg.court_margin_x))
        self.y1 = int(h * cfg.court_margin_y_top)
        self.y2 = int(h * (1.0 - cfg.court_margin_y_bot))

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class JerseyReader:
    """PaddleOCR / EasyOCR on upper-torso crops with temporal voting.

    v3 improvements:
    - Jersey number range filter: rejects numbers outside [1, max_jersey_number]
      before adding to votes, eliminating scoreboard/logo OCR misreads.
    - Best-frame selection: frame_quality_score() evaluates each crop before OCR.
      Only frames above a quality threshold run OCR, reducing bad votes.
    - Confirmed track short-circuit: once a track is confirmed, OCR is skipped
      entirely (controlled by ocr_skip_confirmed in PipelineConfig).
    """

    def __init__(
        self,
        use_gpu: bool = False,
        vote_min: int = 2,
        crops_dir: Optional[Path] = None,
        max_jersey_number: int = 99,
        blur_threshold: float = 40.0,
        brightness_min: float = 40.0,
        brightness_max: float = 230.0,
        min_bbox_area: int = 3000,
    ):
        self.use_gpu = use_gpu
        self.vote_min = vote_min
        self.crops_dir = Path(crops_dir) if crops_dir else None
        self.max_jersey_number = max_jersey_number
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.min_bbox_area = min_bbox_area
        self._votes: Dict[int, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._confirmed: Dict[int, Tuple[str, float]] = {}
        if self.crops_dir:
            self.crops_dir.mkdir(parents=True, exist_ok=True)

    def _get_ocr(self):
        global _paddle_ocr
        if _paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                _paddle_ocr = PaddleOCR(
                    use_angle_cls=False, lang="en", show_log=False,
                    use_gpu=self.use_gpu, det=True, rec=True,
                )
                _paddle_ocr._backend = "paddle"
            except Exception:
                import easyocr
                _paddle_ocr = easyocr.Reader(["en"], gpu=self.use_gpu)
                _paddle_ocr._backend = "easyocr"
        return _paddle_ocr

    @staticmethod
    def _torso_crops(bgr: np.ndarray) -> List[np.ndarray]:
        if bgr is None or bgr.size == 0:
            return []
        h, w = bgr.shape[:2]
        if h < 24 or w < 18:
            return []
        regions = [
            bgr[int(h * 0.12): int(h * 0.78), int(w * 0.08): int(w * 0.92)],
            bgr[int(h * 0.18): int(h * 0.70), int(w * 0.18): int(w * 0.82)],
        ]
        out = []
        for torso in regions:
            if torso.size == 0:
                continue
            scale = max(1.0, 200 / max(torso.shape[0], 1))
            if scale > 1.05:
                torso = cv2.resize(torso, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            out.append(torso)
        return out

    @staticmethod
    def _preprocess(gray: np.ndarray) -> np.ndarray:
        gray = cv2.equalizeHist(gray)
        gray = cv2.bilateralFilter(gray, 5, 50, 50)
        return gray

    def frame_quality_score(self, bgr_crop: np.ndarray) -> float:
        """Return a quality score [0.0, 1.0] for a player crop.

        Combines three signals:
          - bbox_area_ok  : crop is large enough for reliable OCR
          - sharpness     : Laplacian variance (high = sharp, low = blurry)
          - brightness_ok : mean pixel value within the usable range

        Returns 0.0 for crops that are definitely too poor for OCR.
        Returns > 0.5 for good-quality frames worth processing.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return 0.0
        h, w = bgr_crop.shape[:2]
        area = h * w

        # Gate 1: too small → skip entirely
        if area < self.min_bbox_area:
            return 0.0

        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

        # Gate 2: sharpness via Laplacian variance
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_var < self.blur_threshold:
            return 0.0

        # Gate 3: brightness
        mean_brightness = float(np.mean(gray))
        if mean_brightness < self.brightness_min or mean_brightness > self.brightness_max:
            return 0.0

        # Combine into [0, 1] score
        # Sharpness: log-scaled, capped at 500
        sharpness_score = min(1.0, blur_var / 500.0)
        # Size: larger bbox → better, capped at 20000 px²
        size_score = min(1.0, area / 20000.0)
        # Brightness proximity to ideal (130)
        brightness_score = 1.0 - abs(mean_brightness - 130.0) / 130.0

        return (sharpness_score * 0.50 + size_score * 0.30 + brightness_score * 0.20)

    def _is_valid_jersey_number(self, number_str: str) -> bool:
        """Return True only if number_str is a plausible volleyball jersey number.

        Rules:
        - Must be 1 or 2 digits (already enforced by JERSEY_RE upstream)
        - Must not be "0" or "00" (no player wears zero)
        - Must be in [1, max_jersey_number]
        """
        if not number_str or number_str in ("0", "00"):
            return False
        try:
            val = int(number_str.lstrip("0") or "0")
        except ValueError:
            return False
        return 1 <= val <= self.max_jersey_number

    def read_once(self, bgr_crop: np.ndarray) -> Tuple[Optional[str], float]:
        crops = self._torso_crops(bgr_crop)
        if not crops:
            return None, 0.0
        best_num, best_conf = None, 0.0
        try:
            ocr = self._get_ocr()
        except Exception:
            return None, 0.0

        backend = getattr(ocr, "_backend", "paddle")
        for torso in crops:
            gray = self._preprocess(cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY))
            try:
                if backend == "easyocr":
                    results = ocr.readtext(gray, detail=1, paragraph=False, allowlist="0123456789")
                    for (_bbox, text_raw, conf) in results:
                        text = re.sub(r"\D", "", str(text_raw).strip())
                        if not JERSEY_RE.match(text) or conf < 0.30:
                            continue
                        norm = text.lstrip("0") or text
                        # ── v3: range filter ──────────────────────────────
                        if not self._is_valid_jersey_number(norm):
                            continue
                        if conf > best_conf:
                            best_num, best_conf = norm, float(conf)
                else:
                    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    lines = ocr.ocr(bgr, cls=False)
                    if not lines:
                        continue
                    for line in lines:
                        if not line:
                            continue
                        for item in line:
                            if not item or len(item) < 2:
                                continue
                            text_raw, conf = item[1][0], float(item[1][1])
                            text = re.sub(r"\D", "", str(text_raw).strip())
                            if not JERSEY_RE.match(text) or conf < 0.35:
                                continue
                            norm = text.lstrip("0") or text
                            # ── v3: range filter ──────────────────────────
                            if not self._is_valid_jersey_number(norm):
                                continue
                            if conf > best_conf:
                                best_num, best_conf = norm, conf
            except Exception:
                continue
        return best_num, best_conf

    def update_track(self, track_id: int, bgr_crop: np.ndarray, frame_idx: int = 0) -> Tuple[Optional[str], float]:
        if track_id in self._confirmed:
            return self._confirmed[track_id]

        num, conf = self.read_once(bgr_crop)
        if num and conf >= 0.35:
            self._votes[track_id][num] += conf
            if self.crops_dir and conf >= 0.40:
                crop_path = self.crops_dir / f"tid{track_id}_f{frame_idx}_{num}.jpg"
                cv2.imwrite(str(crop_path), bgr_crop)

        if not self._votes[track_id]:
            return num if num else None, conf

        best_num, score = max(self._votes[track_id].items(), key=lambda kv: kv[1])
        votes_for_best = sum(1 for n in self._votes[track_id] if n == best_num)
        norm_conf = min(1.0, score / max(self.vote_min, 1))
        if votes_for_best >= self.vote_min or score >= self.vote_min * 0.5:
            self._confirmed[track_id] = (best_num, norm_conf)
            return best_num, norm_conf
        return best_num, norm_conf

    def is_confirmed(self, track_id: int) -> bool:
        """Return True if this track_id already has a locked jersey identity."""
        return track_id in self._confirmed

    def get_all_confirmed(self) -> Dict[int, Tuple[str, float]]:
        """Return every (track_id -> (jersey, conf)) that reached confirmation."""
        return dict(self._confirmed)


class JerseyRegionDetector:
    """Optional fine-tuned YOLO detector for printed jersey-number regions."""

    def __init__(self, weights: Optional[Path], device: Any, image_size: int):
        self.model: Optional[YOLO] = None
        self.device = device
        self.image_size = image_size
        if weights and Path(weights).exists():
            self.model = YOLO(str(weights))

    def crops(self, player_crop: np.ndarray) -> List[np.ndarray]:
        if self.model is None or player_crop is None or player_crop.size == 0:
            return []
        try:
            result = self.model.predict(player_crop, conf=0.35, verbose=False,
                                        device=self.device, imgsz=self.image_size)[0]
        except Exception:
            return []
        if result.boxes is None:
            return []
        h, w = player_crop.shape[:2]
        regions: List[np.ndarray] = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            region = player_crop[y1:y2, x1:x2]
            if region.shape[0] >= 10 and region.shape[1] >= 8:
                regions.append(region)
        return regions


class BallKalmanTracker:
    """Smooth ball position; reject impossible jumps."""

    def __init__(self, max_jump_px: float = 120.0):
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.vx = 0.0
        self.vy = 0.0
        self.conf = 0.0
        self.missed = 0
        self.max_jump_px = max_jump_px
        self.history: Deque[Tuple[float, float, float, float]] = deque(maxlen=30)
        self.consecutive_updates = 0
        self.max_prediction_frames = 6

    def predict(self) -> Optional[Tuple[float, float, float]]:
        if self.x is None:
            return None
        if self.missed >= self.max_prediction_frames:
            self.x = self.y = None
            self.vx = self.vy = self.conf = 0.0
            self.consecutive_updates = 0
            return None
        self.x += self.vx
        self.y += self.vy
        self.missed += 1
        self.consecutive_updates = 0
        decay = max(0.10, self.conf * (0.90 ** self.missed))
        return self.x, self.y, decay

    def update(self, cx: float, cy: float, conf: float, t: float = 0.0) -> bool:
        if self.x is not None:
            jump = ((cx - self.x) ** 2 + (cy - self.y) ** 2) ** 0.5
            if jump > self.max_jump_px and self.missed < 3:
                self.consecutive_updates = 0
                return False
        if self.x is None or self.missed >= 3:
            self.x, self.y = cx, cy
            self.vx = self.vy = 0.0
        else:
            self.vx = 0.50 * (cx - self.x) + 0.50 * self.vx
            self.vy = 0.50 * (cy - self.y) + 0.50 * self.vy
            self.x = 0.70 * cx + 0.30 * self.x
            self.y = 0.70 * cy + 0.30 * self.y
        self.conf = conf
        self.missed = 0
        self.consecutive_updates += 1
        self.history.append((t, self.x, self.y, conf))
        return True

    def is_confirmed(self, min_frames: int) -> bool:
        return self.consecutive_updates >= max(1, min_frames)

    def speed_px_s(self) -> float:
        if len(self.history) < 3:
            return 0.0
        t0, x0, y0, _ = self.history[0]
        t1, x1, y1, _ = self.history[-1]
        dt = max(1e-3, t1 - t0)
        return float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt)


@dataclass
class OpenSegment:
    event: str
    frame_start: int
    time_start: float
    player_track_id: Optional[int] = None
    jersey_number: str = ""
    jersey_conf: float = 0.0
    ball_speed_peak: float = 0.0
    event_conf: float = 0.0
    rally_id: int = 0
    touch_index: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class RallyEventEngine:
    """
    Multi-frame validated touches, rally segmentation, real event windows.

    Key fix: min_ball_speed is now 30 px/s instead of 250, so touches 2–N
    in a rally (Set, Attack, Block, Dig) are no longer silently discarded.
    The _classify_touch logic uses speed/direction bands that match pixel
    velocities in 25 fps / 1080p footage:
      • Set    : gentle, near-net, upward arc  → speed 30-180, vy < 0
      • Attack : steep downward spike          → vy > 60, speed > 150
      • Block  : sharp deflection near net     → near net, |vy| > 60
      • Dig    : low-trajectory defensive save → ball low, speed 40-200
    """

    def __init__(self, fps: float, cfg: PipelineConfig):
        self.fps = fps or 25.0
        self.cfg = cfg
        self.ball_hist: Deque[Tuple[float, float, float, float]] = deque(maxlen=60)
        self.last_contact_t = -1e9
        self.rally_active = False
        self.rally_id = 0
        self.touch_index = 0
        self.last_ball_t: Optional[float] = None
        self.last_valid_ball_t: Optional[float] = None
        self.open: Optional[OpenSegment] = None
        self.closed: List[dict] = []
        self.net_y: Optional[float] = None
        self._contact_frames = 0
        self._contact_tid: Optional[int] = None
        self._contact_jersey = ""
        self._contact_jconf = 0.0
        self._pending_event: Optional[str] = None
        self._pending_start_frame = 0
        self._pending_start_t = 0.0
        self._last_event_name = ""
        # Running event_id counter (1-based within this video)
        self._event_id = 0

    def _next_event_id(self) -> int:
        self._event_id += 1
        return self._event_id

    def _cooldown_ok(self, t: float) -> bool:
        return (t - self.last_contact_t) >= self.cfg.contact_cooldown_sec

    def _close_open(self, frame_idx: int, t: float):
        if self.open is None:
            return
        end_t = max(t, self.open.time_start + 0.05)
        row = {
            "event_id": self._next_event_id(),
            "event": self.open.event,
            "time_start_sec": round(self.open.time_start, 3),
            "time_end_sec": round(end_t, 3),
            "timestamp_start": format_timestamp_ms(self.open.time_start),
            "timestamp_end": format_timestamp_ms(end_t),
            "timestamp": format_timestamp(self.open.time_start),
            "frame_start": self.open.frame_start,
            "frame_end": int(frame_idx),
            "player_track_id": self.open.player_track_id or "",
            "jersey_number": self.open.jersey_number,
            "jersey_conf": round(self.open.jersey_conf, 3) if self.open.jersey_number else "",
            "event_conf": round(self.open.event_conf, 3),
            "ball_speed_peak": round(self.open.ball_speed_peak, 1),
            "rally_id": self.open.rally_id,
            "touch_index": self.open.touch_index,
            "event_status": "confirmed",
        }
        row.update(self.open.extra)
        self.closed.append(row)
        self.open = None

    def _open_segment(
        self,
        name: str,
        frame_idx: int,
        t: float,
        tid: Optional[int],
        jersey: str,
        jconf: float,
        speed: float,
        event_conf: float = 0.8,
        extra: Optional[dict] = None,
        extend: bool = False,
    ):
        if extend and self.open and self.open.event == name and self.open.player_track_id == tid:
            self.open.ball_speed_peak = max(self.open.ball_speed_peak, speed)
            if extra:
                self.open.extra.update(extra)
            return
        self._close_open(frame_idx, t)
        self.open = OpenSegment(
            event=name,
            frame_start=frame_idx,
            time_start=t,
            player_track_id=tid,
            jersey_number=jersey or "",
            jersey_conf=jconf,
            ball_speed_peak=speed,
            event_conf=event_conf,
            rally_id=self.rally_id,
            touch_index=self.touch_index,
            extra=extra or {},
        )

    def _end_rally(self, frame_idx: int, t: float, speed: float, reason: str):
        if not self.rally_active:
            return
        self._close_open(frame_idx, t)
        self.touch_index += 1
        self._open_segment("Point", frame_idx, t, None, "", 0.0, speed,
                            event_conf=1.0, extra={"detail": reason})
        self._close_open(frame_idx, t)
        self.rally_active = False
        self.touch_index = 0
        self._contact_frames = 0
        self._contact_tid = None

    def _classify_touch(
        self,
        speed: float,
        vx: float,
        vy: float,
        ball_y: float,
        player_y: float,
        frame_h: float,
        is_rally_start: bool,
    ) -> Tuple[str, float]:
        """
        Classify a ball-contact event and return (event_name, confidence).

        Speed and vy are in pixels/second at the video frame rate.
        Tuned for 1080p / 25 fps footage; scales reasonably with resolution.

        Touch hierarchy (first matching rule wins):
          1. Rally start  → Serve (from back, fast) or Reception (front, slower)
          2. Near net + strong deflection → Block
          3. Steep downward spike          → Attack
          4. Gentle upward arc near net    → Set
          5. Low-to-ground defensive save  → Dig
          6. Fallback by touch count
        """
        net = self.net_y if self.net_y is not None else frame_h * 0.42

        # Normalise ball position to [0,1] (0=top, 1=bottom)
        ball_rel = ball_y / max(frame_h, 1)
        # Proximity to net (fraction of frame height)
        net_dist = abs(ball_y - net) / max(frame_h, 1)
        near_net = net_dist < 0.15

        # ── 1. Rally start ──────────────────────────────────────────────────
        if is_rally_start:
            # Serve: player typically deep (bottom 35 %), ball moving fast or
            # sharply upward (vy negative = upward in image coords).
            if ball_rel > 0.62 or speed > 150 or vy < -60:
                return "Serve", 0.85
            return "Reception", 0.75

        # ── 2. Block ────────────────────────────────────────────────────────
        # Ball very close to net, strong deflection in either vertical direction.
        if near_net and (abs(vy) > 60 or speed > 130):
            return "Block", 0.80

        # ── 3. Attack ───────────────────────────────────────────────────────
        # Ball moving sharply downward (vy positive in image coords) OR high speed
        # while ball is at or past the net level.
        if vy > 60 and speed > 100:
            return "Attack", 0.82
        if speed > 200 and ball_rel > 0.35:
            return "Attack", 0.75

        # ── 4. Set ──────────────────────────────────────────────────────────
        # Gentle, controlled upward or lateral touch near the net.
        if near_net and vy < 0 and speed < 160:
            return "Set", 0.78
        if abs(vy) < 60 and speed < 140 and net_dist < 0.25:
            return "Set", 0.72

        # ── 5. Dig ──────────────────────────────────────────────────────────
        # Ball down in the court (bottom 45 %), not fast, defensive posture.
        if ball_rel > 0.55 and speed < 200:
            return "Dig", 0.74
        if ball_rel > 0.45 and speed < 120:
            return "Dig", 0.70

        # ── 6. Fallback by touch index ──────────────────────────────────────
        if self.touch_index <= 1:
            return "Reception", 0.60
        if self.touch_index == 2:
            return "Set", 0.55
        return "Attack" if speed > 80 else "Dig", 0.55

    def _nearest_player(
        self,
        bx: float,
        by: float,
        player_boxes: Sequence[Tuple[int, int, int, int, int, str, float]],
    ) -> Optional[Tuple[int, str, float, float, float]]:
        best = None
        best_d = 1e18
        for (x1, y1, x2, y2, tid, jersey, jconf) in player_boxes:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            pad = self.cfg.contact_dist_px
            if not (x1 - pad <= bx <= x2 + pad and y1 - pad <= by <= y2 + pad):
                continue
            d = (cx - bx) ** 2 + (cy - by) ** 2
            if d < best_d:
                best_d = d
                best = (tid, jersey, jconf, cy, float(d) ** 0.5)
        return best

    def update(
        self,
        frame_idx: int,
        ball: Optional[Tuple[float, float, float]],
        player_boxes: Sequence[Tuple[int, int, int, int, int, str, float]],
        frame_h: int,
    ) -> Optional[str]:
        t = frame_idx / self.fps
        vx, vy, speed = 0.0, 0.0, 0.0

        if ball is not None and ball[2] >= self.cfg.min_ball_conf_contact:
            bx, by, bconf = ball
            self.ball_hist.append((t, bx, by, bconf))
            self.last_ball_t = t
            self.last_valid_ball_t = t
            if len(self.ball_hist) >= 2:
                (_t0, x0, y0, _c0) = self.ball_hist[-2]
                (t1, x1, y1, _c1) = self.ball_hist[-1]
                dt = max(1e-3, t1 - _t0)
                vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
                speed = (vx * vx + vy * vy) ** 0.5

        # Rally end: no trustworthy ball for rally_gap_sec
        if (
            self.rally_active
            and self.last_valid_ball_t is not None
            and (t - self.last_valid_ball_t) > self.cfg.rally_gap_sec
        ):
            self._end_rally(frame_idx, t, speed, "ball_lost")
            return "Point"

        if ball is None or ball[2] < self.cfg.min_ball_conf_contact:
            self._contact_frames = 0
            self._contact_tid = None
            return None

        bx, by, bconf = ball

        # FIX: the old check `speed < min_ball_speed and touch_index > 0` caused
        # ALL after-first touches to be skipped whenever speed < 250. Now min_ball_speed
        # is 30 and the gate only applies mid-rally after touch 2+ to avoid double-
        # counting a stationary or very slow ball as a contact.
        if speed < self.cfg.min_ball_speed and self.rally_active and self.touch_index > 1:
            return None

        nearest = self._nearest_player(bx, by, player_boxes)
        if nearest is None:
            self._contact_frames = 0
            self._contact_tid = None
            return None

        tid, jersey, jconf, player_y, dist = nearest
        if dist > self.cfg.contact_dist_px:
            self._contact_frames = 0
            self._contact_tid = None
            return None

        if self._contact_tid == tid:
            self._contact_frames += 1
        else:
            self._contact_tid = tid
            self._contact_frames = 1
            self._contact_jersey = jersey
            self._contact_jconf = jconf

        if self._contact_frames < self.cfg.contact_min_frames:
            return None
        if not self._cooldown_ok(t):
            return None

        self.last_contact_t = t
        is_start = not self.rally_active

        if is_start:
            self.rally_id += 1
            self.touch_index = 0
            self.rally_active = True
            self._open_segment(
                "Rally", frame_idx, t, tid, jersey, jconf, speed,
                event_conf=1.0, extra={"detail": "rally_start"},
            )
            self._close_open(frame_idx, t)

        self.touch_index += 1
        ev, ev_conf = self._classify_touch(speed, vx, vy, by, player_y, float(frame_h), is_start)
        self._last_event_name = ev
        self._open_segment(
            ev, frame_idx, t, tid, jersey, jconf, speed,
            event_conf=ev_conf,
            extra={"ball_conf": round(bconf, 3), "vx": round(vx, 1), "vy": round(vy, 1)},
        )
        return ev

    def finalize(self, frame_idx: int):
        t = frame_idx / self.fps
        if self.rally_active:
            self._end_rally(frame_idx, t, 0.0, "video_end")
        else:
            self._close_open(frame_idx, t)

    def backfill_jerseys(self, jersey_cache: Dict[int, str], jersey_conf_cache: Dict[int, float]):
        """
        Post-process jersey join.

        After the full video scan jersey_cache has the final (best) jersey for
        every track.  Re-apply it to every closed event row so that events
        detected before the OCR voted-in a number now carry the correct jersey.
        """
        for row in self.closed:
            tid = row.get("player_track_id")
            if not tid or row.get("jersey_number"):
                continue
            try:
                tid_int = int(tid)
            except (TypeError, ValueError):
                continue
            if tid_int in jersey_cache:
                row["jersey_number"] = jersey_cache[tid_int]
                row["jersey_conf"] = round(jersey_conf_cache.get(tid_int, 0.0), 3)


class VolleyballAnalyticsPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        if self.cfg.device == 0 and not torch.cuda.is_available():
            self.cfg.device = "cpu"
        if self.cfg.crops_dir is None:
            self.cfg.crops_dir = self.cfg.project_root / "o" / "p" / "jersey_crops"
        self.player_model, self.player_classes = self._load_player_model()
        self.ball_model, self.ball_classes = self._load_ball_model()
        self.jersey = JerseyReader(
            use_gpu=not self.cfg.ocr_on_cpu and torch.cuda.is_available(),
            vote_min=self.cfg.jersey_vote_min,
            crops_dir=self.cfg.crops_dir if self.cfg.save_jersey_crops else None,
            max_jersey_number=self.cfg.max_jersey_number,
            blur_threshold=self.cfg.ocr_blur_threshold,
            brightness_min=self.cfg.ocr_brightness_min,
            brightness_max=self.cfg.ocr_brightness_max,
            min_bbox_area=self.cfg.ocr_min_bbox_area,
        )
        self.jersey_regions = JerseyRegionDetector(
            self.cfg.jersey_detector_weights, self.cfg.device, self.cfg.image_size
        )

    def _load_player_model(self) -> Tuple[YOLO, Optional[List[int]]]:
        finetuned = self.cfg.finetuned_player or (
            self.cfg.project_root / "models" / "volleyball_player_best.pt"
        )
        if finetuned.exists() and self.cfg.prefer_finetuned_player:
            return YOLO(str(finetuned)), None
        return YOLO(self.cfg.player_weights), [COCO_PERSON]

    def _load_ball_model(self) -> Tuple[YOLO, Optional[List[int]]]:
        finetuned = self.cfg.finetuned_ball or (
            self.cfg.project_root / "models" / "volleyball_ball_best.pt"
        )
        if finetuned.exists() and self.cfg.prefer_finetuned_ball:
            return YOLO(str(finetuned)), None
        ball_w = self.cfg.ball_weights or self.cfg.player_weights
        return YOLO(ball_w), [COCO_SPORTS_BALL]

    def detect_ball_yolo(self, frame: np.ndarray, court: CourtROI) -> Optional[Tuple[float, float, float]]:
        classes = self.ball_classes

        def predict(conf: float):
            return self.ball_model.predict(
                frame, conf=conf, classes=classes, verbose=False,
                device=self.cfg.device,
                imgsz=max(self.cfg.image_size, self.cfg.ball_image_size),
            )[0]

        r = predict(self.cfg.conf_ball)
        if r.boxes is None or len(r.boxes) == 0:
            r = predict(self.cfg.conf_ball_fallback)
        if r.boxes is None or len(r.boxes) == 0:
            return None
        best = None
        frame_area = frame.shape[0] * frame.shape[1]
        for box in r.boxes:
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            if area > frame_area * 0.015:
                continue
            if not court.contains(cx, cy):
                continue
            if best is None or conf > best[2]:
                best = (cx, cy, conf)
        return best

    @staticmethod
    def detect_ball_motion(
        frame: np.ndarray, prev_gray: Optional[np.ndarray], court: CourtROI,
    ) -> Tuple[Optional[Tuple[float, float, float]], np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.GaussianBlur(gray, (5, 5), 0)
        motion = None
        if prev_gray is not None:
            diff = cv2.absdiff(gray_b, prev_gray)
            _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        circles = cv2.HoughCircles(
            gray_b, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=100, param2=28, minRadius=3, maxRadius=18,
        )
        best = None
        if circles is not None:
            for c in circles[0]:
                cx, cy, _r = int(c[0]), int(c[1]), int(c[2])
                if not court.contains(cx, cy):
                    continue
                score = 0.30
                if motion is not None and 0 <= cy < motion.shape[0] and 0 <= cx < motion.shape[1]:
                    if motion[cy, cx] > 0:
                        score += 0.40
                if best is None or score > best[2]:
                    best = (float(cx), float(cy), score)
        return best, gray_b


def finalize_playable_mp4(temp_video: Path, final_mp4: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    final_mp4 = Path(final_mp4)
    temp_video = Path(temp_video)
    if not temp_video.exists() or temp_video.stat().st_size < 1000:
        raise RuntimeError(f"Temp video missing/empty: {temp_video}")
    if final_mp4.suffix.lower() == ".avi":
        shutil.move(str(temp_video), str(final_mp4))
        return final_mp4
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-i", str(temp_video),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23",
            "-movflags", "+faststart", "-an", str(final_mp4),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and final_mp4.exists():
            temp_video.unlink(missing_ok=True)
            return final_mp4
    avi = final_mp4.with_suffix(".avi")
    shutil.move(str(temp_video), str(avi))
    return avi


# ──────────────────────────────────────────────────────────────────────────────
# Single-CSV output spec (exactly these 11 columns, one file per video)
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_CSV_COLUMNS = [
    "event_id",
    "video_name",
    "rally_id",
    "event_type",
    "time_start_sec",
    "time_end_sec",
    "timestamp",
    "player_track_id",
    "jersey_number",
    "jersey_confidence",
    "event_confidence",
]


def _build_output_row(seg: dict, video_name: str) -> dict:
    """Map an internal closed-segment dict to the clean 11-column output row."""
    return {
        "event_id": seg["event_id"],
        "video_name": video_name,
        "rally_id": seg["rally_id"],
        "event_type": seg["event"],
        "time_start_sec": seg["time_start_sec"],
        "time_end_sec": seg["time_end_sec"],
        "timestamp": seg["timestamp"],
        "player_track_id": seg.get("player_track_id", ""),
        "jersey_number": seg.get("jersey_number", ""),
        "jersey_confidence": seg.get("jersey_conf", ""),
        "event_confidence": seg.get("event_conf", ""),
    }


def process_video(
    pipeline: VolleyballAnalyticsPipeline,
    video_path: Path,
    out_path: Path,
    max_seconds: Optional[float] = None,
    split_label: str = "custom",
) -> dict:
    cfg = pipeline.cfg
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(max_seconds * fps) if max_seconds else (total or 10**9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(".writing.avi")
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

    court = CourtROI(w, h, cfg)
    engine = RallyEventEngine(fps=fps, cfg=cfg)
    engine.net_y = h * cfg.net_y_ratio
    ball_tracker = BallKalmanTracker(max_jump_px=cfg.ball_max_jump_px)
    last_boxes: List[Tuple[int, int, int, int, int, str, float, float]] = []
    jersey_cache: Dict[int, str] = {}
    jersey_conf_cache: Dict[int, float] = {}
    last_ocr_frame: Dict[int, int] = defaultdict(lambda: -(10**9))
    current_event = ""
    frame_idx = 0
    t0 = time.time()
    prev_gray: Optional[np.ndarray] = None
    yolo_ball_miss = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = frame_idx / fps
        player_boxes_evt: List[Tuple[int, int, int, int, int, str, float]] = []

        if frame_idx % cfg.player_infer_stride == 0:
            track_kwargs = dict(
                source=frame, persist=True, conf=cfg.conf_player,
                tracker=cfg.tracker, verbose=False, device=cfg.device, imgsz=cfg.image_size,
            )
            if pipeline.player_classes is not None:
                track_kwargs["classes"] = pipeline.player_classes
            results = pipeline.player_model.track(**track_kwargs)
            r0 = results[0]
            new_boxes = []
            if r0.boxes is not None and len(r0.boxes):
                ids = r0.boxes.id
                for i, box in enumerate(r0.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if (x2 - x1) < 18 or (y2 - y1) < 36:
                        continue
                    tid = int(ids[i].item()) if ids is not None else i
                    det_conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                    jersey = jersey_cache.get(tid, "")
                    jconf = jersey_conf_cache.get(tid, 0.0)

                    # ── v3: Best-frame selection + confirmed-track skip ────────
                    # 1. If already confirmed and skip flag is on, don't waste time on OCR.
                    run_ocr = False
                    if cfg.ocr_skip_confirmed and pipeline.jersey.is_confirmed(tid):
                        # Track already locked — just read from cache, no OCR needed.
                        pass
                    else:
                        crop = frame[y1:y2, x1:x2]
                        quality = pipeline.jersey.frame_quality_score(crop)
                        frames_since_ocr = frame_idx - last_ocr_frame[tid]
                        ocr_every_frames = max(1, int(cfg.ocr_interval_sec * fps))
                        ocr_min_frames = max(1, int(cfg.ocr_min_interval_sec * fps))

                        if quality == 0.0:
                            # Frame is definitely bad (too small/blurry/dark) — skip OCR
                            run_ocr = False
                        elif quality >= 0.6 and frames_since_ocr >= ocr_min_frames:
                            # High-quality frame: run OCR more aggressively
                            run_ocr = True
                        elif frames_since_ocr >= ocr_every_frames:
                            # Normal interval elapsed: run OCR if quality is at least marginal
                            run_ocr = quality > 0.0

                        if run_ocr:
                            last_ocr_frame[tid] = frame_idx
                            number_regions = pipeline.jersey_regions.crops(crop)
                            ocr_crop = number_regions[0] if number_regions else crop
                            num, conf = pipeline.jersey.update_track(tid, ocr_crop, frame_idx)
                            if num and conf >= jconf:
                                jersey_cache[tid] = num
                                jersey_conf_cache[tid] = conf
                                jersey, jconf = num, conf
                    # ── end v3 OCR block ─────────────────────────────────────

                    player_boxes_evt.append((x1, y1, x2, y2, tid, jersey, jconf))
                    new_boxes.append((x1, y1, x2, y2, tid, jersey, jconf, det_conf))
            last_boxes = new_boxes
        else:
            player_boxes_evt = [(b[0], b[1], b[2], b[3], b[4], b[5], b[6]) for b in last_boxes]

        ball_det = None
        ball_yolo = None
        ball_accepted = False
        if frame_idx % cfg.infer_stride == 0:
            ball_yolo = pipeline.detect_ball_yolo(frame, court)
            if ball_yolo:
                yolo_ball_miss = 0
                ball_accepted = ball_tracker.update(ball_yolo[0], ball_yolo[1], ball_yolo[2], t_sec)
                if ball_accepted and ball_tracker.is_confirmed(cfg.ball_confirm_frames):
                    ball_det = (ball_tracker.x, ball_tracker.y, ball_tracker.conf)
            else:
                yolo_ball_miss += 1
            if cfg.use_motion_fallback and ball_yolo is None and yolo_ball_miss >= 4:
                motion_ball, prev_gray = pipeline.detect_ball_motion(frame, prev_gray, court)
                if motion_ball is not None:
                    ball_accepted = ball_tracker.update(
                        motion_ball[0], motion_ball[1], motion_ball[2] * 0.75, t_sec
                    )
                    if ball_accepted and ball_tracker.is_confirmed(cfg.ball_confirm_frames):
                        ball_det = (ball_tracker.x, ball_tracker.y, ball_tracker.conf)
            elif prev_gray is None or frame_idx % cfg.infer_stride == 0:
                _, prev_gray = pipeline.detect_ball_motion(frame, prev_gray, court)

        ball = ball_det
        if ball is None and not ball_accepted:
            pred = ball_tracker.predict()
            if pred and pred[2] >= cfg.min_ball_conf_contact:
                ball = pred

        emitted = engine.update(frame_idx, ball, player_boxes_evt, h)
        if emitted:
            current_event = emitted

        # ── Visualisation ────────────────────────────────────────────────
        vis = frame.copy()
        cv2.rectangle(vis, (court.x1, court.y1), (court.x2, court.y2), (80, 80, 80), 1)
        for (x1, y1, x2, y2, tid, jersey, jconf, _dc) in last_boxes:
            color = (0, 220, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"ID{tid}"
            if jersey:
                label += f" #{jersey}({jconf:.2f})"
            cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        if ball is not None:
            bx, by = float(ball[0]), float(ball[1])
            if np.isfinite(bx) and np.isfinite(by) and abs(bx) < 1e5 and abs(by) < 1e5:
                cv2.circle(vis, (int(round(bx)), int(round(by))), 10, (0, 0, 255), 2)
        cv2.line(vis, (0, int(h * cfg.net_y_ratio)), (w, int(h * cfg.net_y_ratio)), (255, 128, 0), 1)
        cv2.rectangle(vis, (0, 0), (w, 44), (0, 0, 0), -1)
        hud = (f"Event: {current_event or '-'} | Players: {len(last_boxes)} | "
               f"Jerseys: {len(jersey_cache)} | {frame_idx}/{max_frames}")
        cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(vis)
        frame_idx += 1
        if frame_idx % 400 == 0:
            print(f"  frame {frame_idx}/{max_frames} ({time.time()-t0:.0f}s)")

    cap.release()
    writer.release()
    engine.finalize(frame_idx)

    # ── Post-process: backfill jerseys from the final cache ───────────────
    # By now OCR has had the entire video to vote; update every event row that
    # was recorded before the jersey was confirmed.
    engine.backfill_jerseys(jersey_cache, jersey_conf_cache)

    playable = finalize_playable_mp4(temp_path, out_path)

    # ── Build the single output CSV (11 columns) ──────────────────────────
    video_name = video_path.name
    output_rows = [_build_output_row(seg, video_name) for seg in engine.closed]

    out_csv_dir = cfg.project_root / "o" / "p"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    out_csv = out_csv_dir / f"{stem}_events.csv"
    pd.DataFrame(output_rows, columns=OUTPUT_CSV_COLUMNS).to_csv(out_csv, index=False)

    # Also save jerseys JSON for reference
    jersey_path = cfg.project_root / "o" / "p" / f"{stem}_jerseys.json"
    jersey_path.write_text(
        json.dumps({str(k): v for k, v in jersey_cache.items()}, indent=2),
        encoding="utf-8",
    )

    return {
        "video": video_path.name,
        "out_video": str(playable),
        "frames": frame_idx,
        "n_events": len(engine.closed),
        "out_csv": str(out_csv),
        "jerseys": dict(jersey_cache),
        "event_rows": engine.closed,
        "output_rows": output_rows,
    }
