# -*- coding: utf-8 -*-
"""Jump Log - athlete tracker, capture-FPS-aware processing, Python 3.11.
Dependencies: streamlit==1.40.1, mediapipe==0.10.21,
opencv-python-headless==4.11.0.86, numpy==1.26.4, imageio-ffmpeg, pandas.
"""
import hashlib
import uuid
import sqlite3
import subprocess
import tempfile
import shutil
import urllib.request
from types import SimpleNamespace
from datetime import datetime, date
from pathlib import Path

import cv2
import imageio_ffmpeg
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pandas as pd
import streamlit as st

G = 9.81
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
DB_PATH = DATA_DIR / "jump_history.sqlite3"
FOOT_IDS = ((29, 31), (30, 32))  # L/R: heel, toe


def height_cm(seconds):
    return G * seconds * seconds / 8.0 * 100.0


def smooth_series(values, fps):
    """Fill only short missing intervals; median filter suppresses 1-frame errors."""
    a = np.asarray(values, dtype=float)
    n = len(a)
    if n == 0:
        return a
    good = np.isfinite(a)
    if good.sum() < 4:
        return np.full(n, np.nan)
    idx = np.arange(n)
    out = np.interp(idx, idx[good], a[good])
    max_gap = max(2, int(round(fps * 0.065)))
    bad = ~good
    starts = np.flatnonzero(bad & ~np.r_[False, bad[:-1]])
    ends = np.flatnonzero(bad & ~np.r_[bad[1:], False]) + 1
    for start, end in zip(starts, ends):
        if end - start > max_gap:
            out[start:end] = np.nan
    window = min(7, max(3, int(round(fps * 0.035)) | 1))
    r = window // 2
    result = np.full(n, np.nan)
    for i in range(n):
        part = out[max(0, i - r):min(n, i + r + 1)]
        good_part = part[np.isfinite(part)]
        if len(good_part) >= (len(part) + 1) // 2:
            result[i] = float(np.median(good_part))
    return result


def pose_features(pose_result, frame_height):
    """Find left/right approximate sole position, pelvis and full-body scale."""
    if not pose_result.pose_landmarks:
        return (np.nan,) * 4
    pts = pose_result.pose_landmarks.landmark

    def sole(indices):
        heel, toe = (pts[j] for j in indices)
        if min(heel.visibility, toe.visibility) < 0.55:
            return np.nan
        return max(heel.y, toe.y) * frame_height

    hips = [pts[23], pts[24]]
    if min(p.visibility for p in hips) > 0.55:
        hip_y = (hips[0].y + hips[1].y) * 0.5 * frame_height
    else:
        hip_y = np.nan
    shoulders = [pts[11], pts[12]]
    ankles = [pts[27], pts[28]]
    if min(p.visibility for p in shoulders + ankles) > 0.55:
        top = (shoulders[0].y + shoulders[1].y) * 0.5
        bottom = (ankles[0].y + ankles[1].y) * 0.5
        scale = max(1.0, (bottom - top) * frame_height)
    else:
        scale = np.nan
    return sole(FOOT_IDS[0]), sole(FOOT_IDS[1]), hip_y, scale


def segments(mask, gap_frames):
    """Contiguous true regions with short false gaps bridged."""
    m = np.asarray(mask, dtype=bool).copy()
    n = len(m)
    i = 0
    while i < n:
        if m[i]:
            i += 1
            continue
        start = i
        while i < n and not m[i]:
            i += 1
        if start > 0 and i < n and (i - start) <= gap_frames:
            m[start:i] = True
    starts = np.flatnonzero(m & ~np.r_[False, m[:-1]])
    ends = np.flatnonzero(m & ~np.r_[m[1:], False]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def detect_events(left_raw, right_raw, hip_raw, scale_raw, fps, times):
    """Conservative automatic takeoff/landing detector: never force a result.

    Sole clearance on BOTH feet defines the flight core. Hip motion validates
    it. Low-clearance crossings refine frame boundaries. Human review remains
    required, since pose landmarks do not measure actual foot-ground contact.
    """
    n = len(left_raw)
    out = {
        "takeoff": None, "landing": None, "quality": "判定不可",
        "reason": "", "left": np.full(n, np.nan),
        "right": np.full(n, np.nan), "both": np.full(n, np.nan),
        "hip": np.full(n, np.nan), "ground_left": np.nan,
        "ground_right": np.nan, "body_scale": np.nan,
        "coverage": 0.0, "candidates": 0,
    }
    if n < 15 or fps < 1:
        out["reason"] = "動画が短すぎるため判定できません。"
        return out
    left = smooth_series(left_raw, fps)
    right = smooth_series(right_raw, fps)
    hip = smooth_series(hip_raw, fps)
    good = np.isfinite(left) & np.isfinite(right) & np.isfinite(hip)
    out["coverage"] = float(np.mean(good))
    if out["coverage"] < 0.60:
        out["reason"] = "両足と腰が明瞭に見えるフレームが不足しています。"
        return out
    valid_scale = np.asarray(scale_raw, dtype=float)
    valid_scale = valid_scale[np.isfinite(valid_scale) & (valid_scale > 20)]
    scale = float(np.median(valid_scale)) if len(valid_scale) else 200.0
    out["body_scale"] = scale

    # Prefer resting feet in first and last 20% of the recording.
    edge = min(max(5, int(round(n * 0.20))), max(5, int(round(fps * 1.0))))

    def estimate_floor(values):
        sample = np.r_[values[:edge], values[-edge:]]
        sample = sample[np.isfinite(sample)]
        if len(sample) < 5:
            return np.nan
        return float(np.percentile(sample, 78))

    gl = estimate_floor(left)
    gr = estimate_floor(right)
    if not np.isfinite(gl) or not np.isfinite(gr):
        out["reason"] = "床の基準位置を推定できませんでした。"
        return out
    out["ground_left"], out["ground_right"] = gl, gr
    lc, rc = gl - left, gr - right
    both = np.minimum(lc, rc)   # Both feet must be in the air.
    both[~good] = np.nan
    out["left"], out["right"], out["both"], out["hip"] = lc, rc, both, hip

    # Thresholds adapt to body size and noise. Boundary is intentionally
    # much closer to the floor than the reliable middle of the flight.
    contacts = np.r_[both[:edge], both[-edge:]]
    contacts = contacts[np.isfinite(contacts)]
    noise = float(1.4826 * np.median(np.abs(contacts - np.median(contacts)))) if len(contacts) >= 8 else 2.0
    core_level = max(4.0, scale * 0.018, noise * 2.2)
    edge_level = max(1.5, scale * 0.004, noise * 0.75)
    if edge_level >= core_level:
        edge_level = core_level * 0.45

    core = np.isfinite(both) & (both > core_level)
    ranges = segments(core, max(1, int(round(fps * 0.035))))
    candidates = []
    for start, end in ranges:
        if end - start < max(2, round(fps * 0.070)):
            continue
        # Extend to earliest/later near-floor crossing; not just center of arc.
        takeoff = start
        limit_left = max(1, start - int(round(fps * 0.24)))
        while (takeoff > limit_left and np.isfinite(both[takeoff - 1])
               and both[takeoff - 1] > edge_level):
            takeoff -= 1
        landing = end
        limit_right = min(n - 2, end + int(round(fps * 0.24)))
        while (landing < limit_right and np.isfinite(both[landing])
               and both[landing] > edge_level):
            landing += 1
        if takeoff < 2 or landing > n - 3:
            continue
        flight = float(times[landing] - times[takeoff])
        if not 0.16 <= flight <= 1.30:
            continue
        valid_section = float(np.mean(good[takeoff:landing + 1]))
        if valid_section < 0.78:
            continue
        # A genuine jump should raise the pelvis during the airborne phase.
        pre = hip[max(0, takeoff - max(3, round(fps * 0.12))):takeoff]
        air = hip[takeoff:landing]
        pre = pre[np.isfinite(pre)]
        air = air[np.isfinite(air)]
        if len(pre) < 2 or len(air) < 3:
            continue
        hip_rise = float(np.median(pre) - np.percentile(air, 12))
        if hip_rise < max(5.0, 0.04 * scale):
            continue
        peak_clearance = float(np.nanmax(both[start:end]))
        # Require contact-like samples on both sides, not a mid-video crop.
        before = both[max(0, takeoff - round(fps * 0.18)):takeoff]
        after = both[landing + 1:min(n, landing + 1 + round(fps * 0.18))]
        if (np.sum(np.isfinite(before) & (before < core_level)) < 2
                or np.sum(np.isfinite(after) & (after < core_level)) < 2):
            continue
        quality_score = (
            0.30 * min(1.0, peak_clearance / (core_level * 3.0))
            + 0.35 * min(1.0, hip_rise / (scale * 0.18))
            + 0.35 * valid_section
        )
        candidates.append((quality_score, peak_clearance, takeoff, landing))

    out["candidates"] = len(candidates)
    if not candidates:
        out["reason"] = (
            "離地・着地を安全に自動判定できませんでした。"
            "撮影条件を確認し、下のフレーム確認から手動指定してください。"
        )
        return out
    best = max(candidates, key=lambda item: (item[0], item[1]))
    out["takeoff"], out["landing"] = int(best[2]), int(best[3])
    out["quality"] = "要確認" if best[0] < 0.83 or out["coverage"] < 0.90 else "比較的安定"
    out["reason"] = "自動推定です。離地・着地のフレームを必ず確認してください。"
    return out


def to_preview_jpeg(frame, max_side=400):
    h, w = frame.shape[:2]
    factor = min(1.0, max_side / max(h, w))
    if factor < 1:
        frame = cv2.resize(frame, (int(w * factor), int(h * factor)))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return buffer.tobytes() if ok else b""



# ================================================
# High-accuracy pose model: MediaPipe Tasks API
# The legacy MediaPipe Pose tries to download Heavy into
# site-packages (read-only on Streamlit Community Cloud).
# Tasks instead accepts an explicit, writable model path.
# ================================================
HEAVY_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)


def get_heavy_model_path():
    """Download the official Heavy model to a writable temporary directory.

    It is reused by subsequent analyses while the app's server is running.
    """
    model_path = Path(tempfile.gettempdir()) / "jump_log_pose_models" / "pose_landmarker_heavy.task"
    if model_path.is_file() and model_path.stat().st_size > 20_000_000:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_name(model_path.name + "." + uuid.uuid4().hex + ".part")
    try:
        with urllib.request.urlopen(HEAVY_TASK_URL, timeout=90) as response:
            with temp_path.open("wb") as dest:
                shutil.copyfileobj(response, dest)
        if temp_path.stat().st_size < 20_000_000:
            raise RuntimeError("Heavyモデルのダウンロードが不完全です。")
        temp_path.replace(model_path)
        return model_path
    except Exception as exc:
        raise RuntimeError(
            "高精度モデルの取得に失敗しました。通信状態を確認して再度お試しください。"
            f" 詳細: {exc}"
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)


class PoseProcessor:
    """Adapt the Tasks Heavy model to the existing legacy Pose result format."""

    def __init__(self, complexity):
        self.is_heavy = (complexity == 2)
        self._last_timestamp_ms = -1
        self._legacy = None
        self._task = None
        if self.is_heavy:
            from mediapipe.tasks import python as tasks_python
            from mediapipe.tasks.python import vision

            options = vision.PoseLandmarkerOptions(
                base_options=tasks_python.BaseOptions(
                    model_asset_path=str(get_heavy_model_path())
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.55,
                min_pose_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self._task = vision.PoseLandmarker.create_from_options(options)
        else:
            self._legacy = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._task is not None:
            self._task.close()
        if self._legacy is not None:
            self._legacy.close()

    def process(self, rgb, frame_index, fps):
        if not self.is_heavy:
            return self._legacy.process(rgb)

        # Video mode requires strictly increasing frame timestamps in ms.
        timestamp_ms = max(
            self._last_timestamp_ms + 1,
            int(round(frame_index * 1000.0 / fps)),
        )
        self._last_timestamp_ms = timestamp_ms
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb).copy(),
        )
        task_result = self._task.detect_for_video(mp_image, timestamp_ms)
        if not task_result.pose_landmarks:
            return SimpleNamespace(pose_landmarks=None)

        # The existing pose_features()/draw_landmarks() expect protobufs.
        landmarks = landmark_pb2.NormalizedLandmarkList()
        for src in task_result.pose_landmarks[0]:
            dst = landmarks.landmark.add()
            dst.x = float(src.x)
            dst.y = float(src.y)
            dst.z = float(src.z)
            if src.visibility is not None:
                dst.visibility = float(src.visibility)
            if src.presence is not None:
                dst.presence = float(src.presence)
        return SimpleNamespace(pose_landmarks=landmarks)


def analyze_video(video_bytes, suffix, model_complexity, display_progress, capture_fps_override=None):
    """Run every frame with MediaPipe; output review JPGs and H264 MP4."""
    cap = None
    writer = None
    with tempfile.TemporaryDirectory(prefix="vertical_jump_lab_") as tmp:
        base = Path(tmp)
        raw_input = base / ("input" + suffix)
        raw_output = base / "skeleton_raw.mp4"
        h264_output = base / "skeleton_h264.mp4"
        raw_input.write_bytes(video_bytes)
        try:
            cap = cv2.VideoCapture(str(raw_input))
            if not cap.isOpened():
                raise RuntimeError("動画を開けません。MP4（H.264）形式に変換して試してください。")
            # OpenCV reports the *playback* FPS stored in the file, not
            # necessarily the camera's capture FPS (e.g. iPhone slow-motion).
            source_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if capture_fps_override is not None:
                fps = float(capture_fps_override)
            else:
                fps = source_fps
            if not np.isfinite(fps) or fps <= 1 or fps > 500:
                raise RuntimeError("動画のFPSを取得できません。スロー動画の場合は撮影時FPSを手動指定してください。")
            source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if source_w <= 0 or source_h <= 0:
                raise RuntimeError("動画の解像度を取得できませんでした。")
            if total > 0 and total / fps > 60:
                raise RuntimeError("解析動画は60秒以下にしてください。ジャンプ1回を5〜10秒に切り出すことを推奨します。")
            # Keep image processing reasonably fast without lowering FPS.
            max_side = 960
            ratio = min(1.0, max_side / max(source_w, source_h))
            w = max(2, int(round(source_w * ratio / 2) * 2))
            h = max(2, int(round(source_h * ratio / 2) * 2))
            writer = cv2.VideoWriter(
                str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError("骨格付き動画の保存を開始できませんでした。")
            props = [[], [], [], []]
            timestamps_ms = []
            frames_jpeg = []
            landmark_count = 0
            draw = mp.solutions.drawing_utils
            pose_api = mp.solutions.pose
            with PoseProcessor(model_complexity) as pose:
                index = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame.shape[1] != w or frame.shape[0] != h:
                        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    result = pose.process(rgb, index, fps)
                    rgb.flags.writeable = True
                    features = pose_features(result, h)
                    for column, val in zip(props, features):
                        column.append(val)
                    timestamps_ms.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
                    if result.pose_landmarks:
                        landmark_count += 1
                        draw.draw_landmarks(
                            frame, result.pose_landmarks, pose_api.POSE_CONNECTIONS,
                            draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=2),
                            draw.DrawingSpec(color=(245, 245, 245), thickness=2, circle_radius=2),
                        )
                    writer.write(frame)
                    frames_jpeg.append(to_preview_jpeg(frame))
                    index += 1
                    if index % 8 == 0 and total > 0:
                        display_progress.progress(min(index / total * 0.88, 0.88))
                    if index > 30000:
                        raise RuntimeError("動画のフレーム数が多すぎます。短く切り出してください。")
            cap.release()
            cap = None
            writer.release()
            writer = None
            n = len(frames_jpeg)
            if n < 10:
                raise RuntimeError("動画のフレームが不足しています。")
            # Prefer actual timestamps if available and consistent; otherwise CFR.
            pos = np.asarray(timestamps_ms, dtype=float) / 1000.0
            dt = np.diff(pos)
            # A deliberate high-FPS override means the file is a uniformly
            # slowed-down recording; playback PTS must not be used for physics.
            # Do not override an ordinary 30fps video: no frames are created.
            if capture_fps_override is not None:
                times = np.arange(n, dtype=float) / fps
                time_source = f"撮影時FPS {fps:g}で再計算（手動指定）"
            elif (np.all(np.isfinite(pos)) and len(dt)
                    and np.all(dt > 0) and 0.5 / fps <= float(np.median(dt)) <= 2.0 / fps):
                times = pos - pos[0]
                time_source = "動画タイムスタンプ"
            else:
                times = np.arange(n, dtype=float) / fps
                time_source = "FPS換算（近似）"
            display_progress.progress(0.90)
            prediction = detect_events(*props, fps, times)
            command = [
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(raw_output), "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-an", str(h264_output),
            ]
            converted = subprocess.run(command, capture_output=True, text=True)
            if converted.returncode != 0 or not h264_output.exists():
                raise RuntimeError("H.264変換に失敗しました: " + converted.stderr[-900:])
            display_progress.progress(1.0)
            return {
                "fps": fps, "source_fps": source_fps,
                "fps_override": capture_fps_override, "frames": n, "times": times,
                "time_source": time_source, "pose_coverage": landmark_count / n,
                "prediction": prediction, "preview_frames": frames_jpeg,
                "video_h264": h264_output.read_bytes(), "resolution": (w, h),
                "duration": float(times[-1]) if n else 0.0,
            }
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()


# ================================================
# Local athlete profiles & permanent history
# ================================================


def open_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH), timeout=15)
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=15000")
    return db


def init_db():
    db = open_db()
    try:
        db.execute("""
        CREATE TABLE IF NOT EXISTS athletes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL COLLATE NOCASE UNIQUE,
            created_at TEXT NOT NULL
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id INTEGER NOT NULL REFERENCES athletes(id),
            measured_date TEXT NOT NULL,
            saved_at TEXT NOT NULL,
            video_filename TEXT NOT NULL,
            saved_video_name TEXT,
            sport TEXT NOT NULL,
            jump_type TEXT NOT NULL,
            weight_kg REAL,
            notes TEXT NOT NULL,
            fps REAL NOT NULL,
            takeoff_frame INTEGER NOT NULL,
            landing_frame INTEGER NOT NULL,
            flight_seconds REAL NOT NULL,
            height_cm REAL NOT NULL,
            detection TEXT NOT NULL,
            time_source TEXT NOT NULL
        )""")
        count = db.execute("SELECT COUNT(*) FROM athletes").fetchone()[0]
        if count == 0:
            db.execute(
                "INSERT INTO athletes (name,created_at) VALUES (?,?)",
                ("自分", datetime.now().isoformat(timespec="seconds"))
            )
        db.commit()
    finally:
        db.close()


def get_athletes():
    db = open_db()
    try:
        return db.execute("SELECT id,name FROM athletes ORDER BY id").fetchall()
    finally:
        db.close()


def add_athlete(name):
    name = name.strip()
    if not 1 <= len(name) <= 40:
        raise ValueError("名前は1〜40文字で入力してください。")
    db = open_db()
    try:
        cur = db.execute(
            "INSERT INTO athletes (name,created_at) VALUES (?,?)",
            (name, datetime.now().isoformat(timespec="seconds"))
        )
        db.commit()
        return cur.lastrowid
    finally:
        db.close()


def get_measurements(athlete_id):
    db = open_db()
    try:
        rows = db.execute("""
            SELECT id, measured_date, saved_at, video_filename,
                   saved_video_name, sport, jump_type, weight_kg, notes,
                   fps, takeoff_frame, landing_frame, flight_seconds,
                   height_cm, detection, time_source
            FROM measurements WHERE athlete_id = ?
            ORDER BY measured_date DESC, id DESC
        """, (int(athlete_id),)).fetchall()
    finally:
        db.close()
    names = ["ID", "測定日", "保存日時", "元動画", "保存動画", "競技",
             "ジャンプ種目", "体重(kg)", "メモ", "FPS", "離地フレーム",
             "着地フレーム", "滞空時間(秒)", "跳躍高(cm)", "判定方法", "時間の根拠"]
    return pd.DataFrame(rows, columns=names)


def save_record(athlete_id, day, sport, jump_type, weight, notes,
                upload_name, video_data, result, takeoff, landing, detection):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_name = f"{uuid.uuid4().hex}.mp4"
    path = VIDEO_DIR / video_name
    path.write_bytes(video_data)
    flight = float(result["times"][landing] - result["times"][takeoff])
    try:
        db = open_db()
        try:
            db.execute("""
                INSERT INTO measurements (
                    athlete_id, measured_date, saved_at, video_filename,
                    saved_video_name, sport, jump_type, weight_kg, notes,
                    fps, takeoff_frame, landing_frame, flight_seconds,
                    height_cm, detection, time_source
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                int(athlete_id), day.isoformat(),
                datetime.now().isoformat(timespec="seconds"),
                upload_name, video_name, sport, jump_type,
                None if weight <= 0 else float(weight), notes.strip(),
                float(result["fps"]), int(takeoff), int(landing), flight,
                float(height_cm(flight)), detection, result["time_source"]
            ))
            db.commit()
        finally:
            db.close()
    except Exception:
        path.unlink(missing_ok=True)
        raise


def safe_video_path(video_name):
    if not video_name or not isinstance(video_name, str):
        return None
    # Names are generated by this application, never trust arbitrary paths.
    if len(video_name) != 36 or not video_name.endswith(".mp4"):
        return None
    if any(c not in "0123456789abcdef" for c in video_name[:-4]):
        return None
    return VIDEO_DIR / video_name


def delete_record(rec_id, athlete_id):
    db = open_db()
    try:
        found = db.execute(
            "SELECT saved_video_name FROM measurements WHERE id=? AND athlete_id=?",
            (int(rec_id), int(athlete_id))
        ).fetchone()
        if found is None:
            return False
        db.execute("DELETE FROM measurements WHERE id=? AND athlete_id=?",
                   (int(rec_id), int(athlete_id)))
        db.commit()
    finally:
        db.close()
    p = safe_video_path(found[0])
    if p:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    return True


def display_athlete_dashboard(history):
    st.subheader("パフォーマンス")
    if history.empty:
        st.info("最初の測定を保存すると、自己ベストや成長グラフが表示されます。")
        return
    jump_types = sorted(history["ジャンプ種目"].dropna().unique().tolist())
    chosen = st.selectbox("比較するジャンプ種目", ["すべて"] + jump_types, key="dashboard_jump_type")
    if chosen != "すべて":
        history = history[history["ジャンプ種目"] == chosen].copy()
    elif len(jump_types) > 1:
        st.caption("異なるジャンプ種目の記録が混在しています。同じ種目で絞り込むと成長を比較しやすくなります。")
    ordered = history.sort_values(["測定日", "ID"])
    last = ordered.iloc[-1]
    best = float(history["跳躍高(cm)"].max())
    latest = float(last["跳躍高(cm)"])
    avg = float(history["跳躍高(cm)"].mean())
    a, b, c, d = st.columns(4)
    a.metric("自己ベスト", f"{best:.1f} cm")
    b.metric("最新記録", f"{latest:.1f} cm")
    c.metric("平均", f"{avg:.1f} cm")
    d.metric("測定回数", f"{len(history)} 回")
    if len(history) >= 2:
        first = float(ordered.iloc[0]["跳躍高(cm)"])
        st.caption(f"初回の保存記録との差：{latest - first:+.1f} cm（測定条件が同じ場合の参考値）")
    grouped = (history.groupby("測定日", as_index=False)["跳躍高(cm)"]
               .max().sort_values("測定日"))
    grouped["測定日"] = pd.to_datetime(grouped["測定日"])
    st.markdown("**日ごとの最高記録**")
    st.line_chart(grouped.set_index("測定日")["跳躍高(cm)"], use_container_width=True)
    st.caption("同じ日に複数回測った場合は、その日の最高値を表示します。")


def result_warning(flight, history):
    if not (0.16 <= flight <= 1.30):
        st.error("一般的なジャンプ計測の目安範囲から外れています。離地・着地を再確認してください。")
    if len(history) >= 3:
        median = float(history["跳躍高(cm)"].median())
        jump = height_cm(flight)
        if median > 0 and abs(jump - median) / median > 0.35:
            st.warning(
                f"過去の記録（中央値 {median:.1f} cm）と大きな差があります。"
                "動画の離地・着地フレームや撮影条件を再確認してください。"
            )


def invalidate_review():
    """Changing a proposed event frame requires confirmation again."""
    st.session_state["review_confirmed"] = False


def frame_preview(label, result, frame_number):
    idx = int(frame_number)
    st.image(result["preview_frames"][idx],
             caption=f"{label}：{idx} フレーム / {result['times'][idx]:.3f} 秒",
             use_column_width=True)


def render_measure_tab(athlete_id, history):
    st.subheader("新しく測定する")
    st.write("動画を選ぶだけで骨格を解析し、離地と着地の候補を提示します。映像を確認してから記録を保存できます。")
    video = st.file_uploader("ジャンプ動画をアップロード", type=["mp4", "mov", "avi"], key="athlete_video")
    if video is None:
        st.info("**おすすめ：** 真横から全身・床を撮影し、ジャンプ前後に静止した場面を入れた5〜10秒の動画。60fps以上、可能なら120fpsで撮影してください。")
        return
    video_bytes = video.getvalue()
    video_hash = hashlib.sha256(video_bytes).hexdigest()
    if st.session_state.get("analysis_owner") != athlete_id:
        st.session_state["analysis"] = None
        st.session_state["active_video_hash"] = None
        st.session_state["analysis_owner"] = athlete_id
    if st.session_state.get("active_video_hash") != video_hash:
        st.session_state["active_video_hash"] = video_hash
        st.session_state["analysis"] = None
        st.session_state["last_saved_measurement"] = None
        for k in ("takeoff_input", "landing_input", "review_confirmed"):
            st.session_state.pop(k, None)
    st.caption(f"{video.name}（{len(video_bytes)/1048576:.1f} MB）")
    if Path(video.name).suffix.lower() == ".mp4":
        with st.expander("元動画を確認", expanded=False):
            st.video(video_bytes, format="video/mp4")
    with st.expander("AI解析の設定", expanded=False):
        model = st.selectbox("骨格モデル", [2, 1, 0], index=1,
                             format_func=lambda x: {2: "高精度 / Heavy（時間がかかる）", 1: "標準 / Full", 0: "高速 / Lite"}[x])
        st.caption("骨格の検出精度を高めても、離地・着地判定の正確さが保証されるわけではありません。")
        fps_modes = {
            "自動（動画ファイルに記録されたFPSを使用）": None,
            "120fpsで撮影したスロー動画（実時間に補正）": 120,
            "60fpsで撮影したスロー動画（実時間に補正）": 60,
            "240fpsで撮影したスロー動画（実時間に補正）": 240,
        }
        fps_mode = st.selectbox(
            "時間計算に使用するFPS",
            list(fps_modes), index=0, key="fps_mode",
            help="通常の120fpsファイルは『自動』で120fpsのまま解析します。"
        )
        fps_override = fps_modes[fps_mode]
        if fps_override is not None:
            st.warning(
                "この補正は、元動画が本当に指定FPSで撮影され、全フレームが残っている"
                "スローモーション動画の場合に限り使用してください。普通の30fps動画を"
                "120に指定しても画質や時間分解能は向上せず、測定値が不正確になります。"
            )
        else:
            st.caption(
                "撮影時の120fpsがファイルにも記録されていれば、そのまま120fpsで解析します。"
                "iPhoneのスローモーション動画は30fps再生として保存される場合があるため、"
                "その場合だけ撮影時FPSを指定してください。"
            )
    if st.button("AIでジャンプを解析する", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        status.info("骨格検出 → 離地・着地候補の推定 → 再生用動画の生成")
        try:
            result = analyze_video(
                video_bytes, Path(video.name).suffix.lower(), model, progress,
                capture_fps_override=fps_override,
            )
            st.session_state["analysis"] = result
            prediction = result["prediction"]
            if prediction["takeoff"] is not None:
                start, end = prediction["takeoff"], prediction["landing"]
            else:
                start = max(0, result["frames"] // 3)
                end = min(result["frames"] - 1, result["frames"] * 2 // 3)
            st.session_state["takeoff_input"] = int(start)
            st.session_state["landing_input"] = int(end)
            st.session_state["review_confirmed"] = False
            st.session_state["last_saved_measurement"] = None
            status.empty()
            progress.empty()
            st.rerun()
        except Exception as e:
            status.error("動画解析に失敗しました。入力形式・FPS・MediaPipeの設定を確認してください。")
            st.error(str(e))
            progress.empty()
    result = st.session_state.get("analysis")
    if result is None:
        return
    if result.get("fps_override") != fps_override:
        st.info("FPS設定を変更しました。新しい設定を適用するには『AIでジャンプを解析する』をもう一度押してください。")
    n, fps = result["frames"], result["fps"]
    pred = result["prediction"]
    st.divider()
    st.markdown("### 解析結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("解析FPS", f"{fps:.2f}")
    c2.metric("フレーム数", str(n))
    c3.metric("骨格検出率", f"{result['pose_coverage']*100:.0f}%")
    detected_file_fps = result.get("source_fps", fps)
    st.caption(f"元ファイルの再生FPS：{detected_file_fps:.2f} / 解析・出力FPS：{fps:.2f}")
    if result.get("fps_override") is not None:
        st.warning(
            "撮影時FPSを手動で補正した計測値です。動画にそのFPS分の実フレームが"
            "記録されていることを確認してください。"
        )
    if fps < 60:
        st.warning("60fps未満の動画はフレーム誤差が大きいため、結果の解釈に注意してください。")
    if pred["takeoff"] is None:
        st.warning("自動検出に失敗しました。開始・終了フレームを目視で選択してください。現在の数字は仮置きです。")
        st.caption(pred["reason"])
    else:
        candidate_seconds = float(result["times"][pred["landing"]] - result["times"][pred["takeoff"]])
        st.info(f"AIの参考値：{height_cm(candidate_seconds):.1f} cm ｜ 滞空 {candidate_seconds:.3f} 秒 ｜ {pred['quality']}")
    st.video(result["video_h264"], format="video/mp4")
    st.download_button("骨格付き動画をダウンロード", result["video_h264"],
                       file_name="jump_with_skeleton.mp4", mime="video/mp4")
    st.markdown("### 離地・着地を確認")
    st.write("離地＝両足が床から完全に離れた最初のフレーム。着地＝どちらかの足が床に最初に触れたフレーム。")
    col_left, col_right = st.columns(2)
    with col_left:
        takeoff = st.number_input("離地フレーム", 0, n - 1, step=1, key="takeoff_input", on_change=invalidate_review)
        frame_preview("離地候補", result, takeoff)
    with col_right:
        landing = st.number_input("着地フレーム", 0, n - 1, step=1, key="landing_input", on_change=invalidate_review)
        frame_preview("着地候補", result, landing)
    if int(landing) <= int(takeoff):
        st.error("着地フレームが離地フレームより前になっています。")
        return
    flight = float(result["times"][int(landing)] - result["times"][int(takeoff)])
    if flight <= 0:
        st.error("滞空時間を計算できません。フレーム時刻を確認してください。")
        return
    jump = height_cm(flight)
    st.markdown("### 今回の測定値")
    k1, k2, k3 = st.columns(3)
    k1.metric("推定跳躍高", f"{jump:.1f} cm")
    k2.metric("滞空時間", f"{flight:.3f} 秒")
    k3.metric("判定フレーム差", f"{int(landing) - int(takeoff)}")
    result_warning(flight, history)
    if pred["takeoff"] is None:
        st.warning("自動検出できていないため、この値は仮設定したフレームから計算した値です。必ず修正してください。")
    st.caption("計算式：h = 9.81 × t² / 8。離地と着地の身体重心の高さが同じ場合の近似です。")
    st.caption(f"時刻計算：{result['time_source']}。平均FPSと実際の撮影間隔が違う動画では誤差が生じます。")
    with st.expander("解析データを見る（検証用）"):
        px = pred["both"]
        if np.isfinite(px).any():
            chart = pd.DataFrame({"推定した両足の床からの高さ（px）": px}, index=result["times"])
            chart.index.name = "秒"
            st.line_chart(chart)
        csv_data = pd.DataFrame({
            "frame": np.arange(n), "time_s": result["times"],
            "left_clearance_px": pred["left"], "right_clearance_px": pred["right"],
            "both_clearance_px": pred["both"], "hip_y_px": pred["hip"]
        }).to_csv(index=False).encode("utf-8-sig")
        st.download_button("解析CSVを保存", csv_data, file_name="jump_landmarks.csv", mime="text/csv")
    st.divider()
    st.markdown("### 記録を保存する")
    st.write("撮影した日付・競技・ジャンプの種類を登録できます。保存後は『測定履歴』から動画を見返せます。")
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        day = st.date_input("測定日", value=date.today(), key="measurement_day")
        sport = st.selectbox("競技", ["野球", "バスケットボール", "バレーボール", "陸上", "サッカー", "その他"], key="sport_selection")
        weight = st.number_input("体重（kg・任意）", min_value=0.0, max_value=250.0,
                                 value=0.0, step=0.5, key="weight_entry",
                                 help="入力しない場合は0のままで保存できます。")
    with form_col2:
        jump_type = st.selectbox("ジャンプ種目", ["CMJ（反動あり）", "SJ（反動なし）", "腕振りあり", "その他"], key="jump_type_selection")
        notes = st.text_area("メモ（任意）", placeholder="例：練習前 / ウォームアップ後 / 疲労感あり", key="measurement_notes")
    confirmed = st.checkbox("離地・着地の画像を確認し、このフレームで記録する", key="review_confirmed")
    already_saved = st.session_state.get("last_saved_measurement")
    if already_saved is not None:
        st.success("記録を保存しました。『測定履歴』で確認できます。別のジャンプ動画をアップロードすると次の測定ができます。")
    elif st.button("この結果を保存", type="primary", use_container_width=True, disabled=not confirmed):
        detection = ("AI候補を画像確認" if pred["takeoff"] is not None and
                     int(takeoff) == pred["takeoff"] and int(landing) == pred["landing"]
                     else "フレーム手動調整")
        try:
            save_record(athlete_id, day, sport, jump_type, weight, notes,
                        video.name, result["video_h264"], result,
                        int(takeoff), int(landing), detection)
            st.session_state["last_saved_measurement"] = video_hash
            st.rerun()
        except Exception as e:
            st.error(f"保存に失敗しました：{e}")


def render_history_tab(athlete_id, history):
    st.subheader("測定履歴")
    if history.empty:
        st.info("まだ記録がありません。『新規測定』からジャンプ動画を登録してください。")
        return
    c1, c2 = st.columns(2)
    with c1:
        all_sports = ["すべて"] + sorted(history["競技"].dropna().unique().tolist())
        sport_filter = st.selectbox("競技で絞り込む", all_sports, key="filter_sport")
    with c2:
        all_types = ["すべて"] + sorted(history["ジャンプ種目"].dropna().unique().tolist())
        type_filter = st.selectbox("ジャンプ種目で絞り込む", all_types, key="filter_type")
    filt = history.copy()
    if sport_filter != "すべて":
        filt = filt[filt["競技"] == sport_filter]
    if type_filter != "すべて":
        filt = filt[filt["ジャンプ種目"] == type_filter]
    if filt.empty:
        st.info("指定した条件に合う測定記録がありません。")
        return
    view = filt[["ID", "測定日", "競技", "ジャンプ種目", "跳躍高(cm)", "滞空時間(秒)", "メモ"]].copy()
    view["跳躍高(cm)"] = view["跳躍高(cm)"].round(1)
    view["滞空時間(秒)"] = view["滞空時間(秒)"].round(3)
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("表示中の記録をCSVで保存", filt.drop(columns=["保存動画"]).to_csv(index=False).encode("utf-8-sig"),
                       file_name="jump_history.csv", mime="text/csv")
    options = filt["ID"].astype(int).tolist()
    lookup = {int(row["ID"]): row for _, row in filt.iterrows()}
    selected = st.selectbox("詳しく見る記録", options,
                            format_func=lambda x: f"{lookup[x]['測定日']}｜{lookup[x]['跳躍高(cm)']:.1f} cm｜{lookup[x]['ジャンプ種目']}")
    item = lookup[selected]
    c1, c2 = st.columns(2)
    c1.metric("跳躍高", f"{item['跳躍高(cm)']:.1f} cm")
    c2.metric("滞空時間", f"{item['滞空時間(秒)']:.3f} 秒")
    st.write(f"**記録日：** {item['測定日']}　**競技：** {item['競技']}　**ジャンプ種目：** {item['ジャンプ種目']}")
    if item["メモ"]:
        st.write(f"**メモ：** {item['メモ']}")
    st.caption(f"判定：{item['判定方法']}｜離地 {item['離地フレーム']} フレーム、着地 {item['着地フレーム']} フレーム")
    vid_path = safe_video_path(item["保存動画"])
    if vid_path is not None and vid_path.is_file():
        st.video(vid_path.read_bytes(), format="video/mp4")
        st.download_button("この動画をダウンロード", vid_path.read_bytes(),
                           file_name=f"jump_{item['測定日']}_{selected}.mp4", mime="video/mp4")
    else:
        st.warning("動画ファイルが見つかりません。保存フォルダが移動・削除された可能性があります。")
    with st.expander("この記録を削除"):
        confirm_delete = st.checkbox("この記録と保存済み動画を削除することを確認しました", key=f"confirm_delete_{selected}")
        if st.button("記録を削除", key=f"delete_{selected}", disabled=not confirm_delete):
            if delete_record(selected, athlete_id):
                st.success("記録を削除しました。")
                st.rerun()
            else:
                st.error("削除する記録が見つかりませんでした。")


def render_howto_tab():
    st.subheader("撮影のコツ")
    st.markdown("""
1. スマートフォンを固定し、身体の**真横**から撮影する。
2. 足先から頭まで全身を画面に入れ、**両足と床が見える**状態にする。
3. ジャンプの前後はそれぞれ0.5〜1秒ほど静止する。
4. **同じ撮影位置・同じジャンプ種目**で記録を比較する。
5. 60fps以上、可能なら120fpsの動画を用いる。

**動画の長さ：** ジャンプ1回を含む5〜10秒が目安です。
""")
    st.warning("推定値は測定器と同等の精度を保証するものではありません。骨格推定は床との接触を直接検知できないため、離地・着地画像を確認してから保存してください。")
    st.write("**計算式：** 跳躍高(cm) = 9.81 × 滞空時間² ÷ 8 × 100")
    st.caption("離地・着地時の重心の高さが等しいと仮定した滞空時間法です。膝の曲げ方などにより実際の跳躍高と差が出る場合があります。")
    st.divider()
    st.subheader("データの保存場所")
    st.code(str(DATA_DIR))
    st.write("測定履歴と動画は、このPC内の `data` フォルダに保存されます。バックアップするときはフォルダごとコピーしてください。")
    st.caption("この試作版にはログイン機能・クラウド同期・オンライン共有はありません。他のPCやスマートフォンと履歴は自動で同期されません。")


def main():
    st.set_page_config(page_title="Vertical Jump | Jump Log", page_icon="🏃", layout="wide")
    init_db()
    st.title("🏃 Jump Log")
    st.caption("垂直跳びを測って、記録して、成長を振り返る。")
    with st.sidebar:
        st.markdown("### 選手プロフィール")
        athletes = get_athletes()
        athlete_ids = [x[0] for x in athletes]
        athlete_names = {row[0]: row[1] for row in athletes}
        choice = st.session_state.get("selected_athlete_id")
        if choice not in athlete_ids:
            choice = athlete_ids[0]
        selected = st.selectbox("記録を見る選手", athlete_ids, index=athlete_ids.index(choice),
                                format_func=lambda key: athlete_names[key], key="athlete_selection")
        st.session_state["selected_athlete_id"] = selected
        with st.expander("選手を追加する"):
            name = st.text_input("選手名", max_chars=40, key="new_athlete_name")
            if st.button("選手を登録", key="add_athlete"):
                try:
                    new_id = add_athlete(name)
                    st.session_state["created_athlete"] = name.strip()
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("同じ名前の選手が既に登録されています。")
                except ValueError as e:
                    st.error(str(e))
        if st.session_state.pop("created_athlete", None):
            st.success("選手を登録しました。上の一覧から選択してください。")
        st.divider()
        st.caption("データはこのPC内に保存されます。")
    history = get_measurements(selected)
    # Streamlit tabs execute *all* tab contents on every rerun.  This includes
    # video processing/rendering that is not visible on the selected tab.
    # Conditional navigation keeps screens independent and substantially
    # reduces unintended side effects and memory pressure.
    pages = ["🏠 ホーム", "📹 新規測定", "📒 測定履歴", "ℹ️ 使い方"]
    chosen_page = st.radio(
        "画面を選択", pages,
        key="jump_log_current_page",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    try:
        if chosen_page == pages[0]:
            st.markdown(f"### {athlete_names[selected]} の記録")
            display_athlete_dashboard(history)
            st.divider()
            st.markdown("**使い方**")
            st.write("① 動画を撮影 → ② AIで測定 → ③ 離地と着地を確認 → ④ 日付・メモ付きで保存 → ⑤ 過去の結果と比較")
            if history.empty:
                st.info("まず『新規測定』から最初のジャンプを登録しましょう。")
        elif chosen_page == pages[1]:
            render_measure_tab(selected, history)
        elif chosen_page == pages[2]:
            render_history_tab(selected, history)
        else:
            render_howto_tab()
    except Exception as exc:
        st.error(f"『{chosen_page}』画面の表示中にエラーが発生しました。")
        st.exception(exc)
        st.info("ここに表示された赤いエラーと詳細をコピーすると、原因を特定できます。測定履歴は削除されていません。")
    st.divider()
    st.caption("Jump Log | 個人記録向けのローカル試作版。自動判定結果は目視確認が必要です。")


if __name__ == "__main__":
    main()
