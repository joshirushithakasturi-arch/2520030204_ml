from pathlib import Path
from collections import Counter
from datetime import datetime
import uuid
import json
import os

import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Model is loaded lazily so the website can display setup errors clearly.
model = None
model_load_error = None
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}


def get_model():
    global model, model_load_error
    if model is not None:
        return model
    if model_load_error:
        raise RuntimeError(model_load_error)
    try:
        from ultralytics import YOLO
        # On first run, Ultralytics downloads the small pretrained weights if absent.
        model = YOLO("yolov8n.pt")
        return model
    except Exception as exc:
        model_load_error = (
            "Could not load the YOLO model. Check your internet connection on first run "
            "and install the packages from requirements.txt. Details: " + str(exc)
        )
        raise RuntimeError(model_load_error)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_density(avg_visible, peak_visible):
    """Demo thresholds only; calibrate for a particular camera before real deployment."""
    score = max(avg_visible, peak_visible * 0.65)
    if score < 4:
        return "Light", "Traffic appears relatively clear in the analyzed frames.", "low"
    if score < 10:
        return "Moderate", "Several vehicles are visible. Continue monitoring for queue growth.", "medium"
    return "Heavy", "Many vehicles are visible. Review the annotated video for queueing or slow movement.", "high"


def analyze_video(video_path, output_path, analysis_speed="balanced"):
    """Analyze a road clip and return detection, congestion, road-space, and risk indicators.

    All movement/occupancy thresholds are provisional image-space heuristics. They are
    intended for a student prototype and must not be treated as calibrated traffic metrics.
    """
    detector = get_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("The uploaded file could not be opened as a video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("The video does not contain readable frames.")

    # Fast mode samples fewer frames. Balanced mode samples up to about 8 FPS.
    target_sample_fps = 4 if analysis_speed == "fast" else 8
    sample_every = max(1, int(round(fps / target_sample_fps)))
    writer_fps = max(1.0, fps / sample_every)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), writer_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError("Could not create the annotated output video. Try another video format.")

    seen_ids = set()
    class_counts = Counter()
    samples = []
    processed_frames = 0
    peak_visible = 0
    frame_index = 0
    track_history = {}
    motion_samples = []
    roi_counts = []
    frame_diagonal = max(1.0, (width ** 2 + height ** 2) ** 0.5)

    # A fixed image-space analysis area, divided into three vertical regions.
    # It is not a true road mask; users should review the camera framing.
    area_x1, area_x2 = int(width * 0.05), int(width * 0.95)
    area_y1, area_y2 = int(height * 0.18), int(height * 0.95)
    area_width = max(1, area_x2 - area_x1)
    area_height = max(1, area_y2 - area_y1)
    region_width = area_width / 3.0
    region_names = ["Left", "Centre", "Right"]
    region_accumulator = {
        name: {"visible_sum": 0, "occupancy_sum": 0.0, "speed_sum": 0.0,
               "speed_count": 0, "slow_count": 0, "frame_count": 0}
        for name in region_names
    }
    per_frame_region_counts = {name: [] for name in region_names}
    per_frame_region_occupancy = {name: [] for name in region_names}
    per_frame_region_speeds = {name: [] for name in region_names}
    near_miss_events = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % sample_every != 0:
                frame_index += 1
                continue

            result = detector.track(frame, persist=True, classes=list(VEHICLE_CLASSES.keys()),
                                    conf=0.25, verbose=False, tracker="bytetrack.yaml")[0]
            annotated = result.plot()
            visible_now = 0
            vehicles_in_area = 0
            elapsed_video = frame_index / fps
            frame_regions = {name: {"count": 0, "box_area": 0.0, "speeds": []} for name in region_names}
            current_tracks = []

            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0].item())
                    name = VEHICLE_CLASSES.get(cls_id, "Vehicle")
                    visible_now += 1
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    in_area = area_x1 <= cx <= area_x2 and area_y1 <= cy <= area_y2
                    if in_area:
                        vehicles_in_area += 1
                        region_index = min(2, max(0, int((cx - area_x1) / region_width)))
                        region_name = region_names[region_index]
                        frame_regions[region_name]["count"] += 1
                        # Clip the vehicle box to its region before estimating pixel occupancy.
                        rx1 = area_x1 + region_index * region_width
                        rx2 = area_x1 + (region_index + 1) * region_width
                        clipped_w = max(0.0, min(x2, rx2) - max(x1, rx1))
                        clipped_h = max(0.0, min(y2, area_y2) - max(y1, area_y1))
                        frame_regions[region_name]["box_area"] += clipped_w * clipped_h

                    track_id = None
                    if result.boxes.id is not None:
                        track_id = int(result.boxes.id[i].item())
                    velocity = None
                    normalized_speed = None
                    if track_id is not None:
                        key = f"{cls_id}:{track_id}"
                        if key not in seen_ids:
                            seen_ids.add(key)
                            class_counts[name] += 1
                        previous = track_history.get(key)
                        if previous is not None:
                            delta_t = elapsed_video - previous["seconds"]
                            if delta_t > 0:
                                vx = (cx - previous["x"]) / frame_diagonal / delta_t
                                vy = (cy - previous["y"]) / frame_diagonal / delta_t
                                normalized_speed = (vx * vx + vy * vy) ** 0.5
                                velocity = (vx, vy)
                                motion_samples.append({"seconds": elapsed_video,
                                                       "normalized_speed": normalized_speed,
                                                       "in_area": in_area})
                                if in_area:
                                    region_index = min(2, max(0, int((cx - area_x1) / region_width)))
                                    frame_regions[region_names[region_index]]["speeds"].append(normalized_speed)
                        track_history[key] = {"seconds": elapsed_video, "x": cx, "y": cy,
                                              "vx": velocity[0] if velocity else 0.0,
                                              "vy": velocity[1] if velocity else 0.0,
                                              "class_id": cls_id}
                        current_tracks.append({"key": key, "track_id": track_id, "class_name": name,
                                               "x": cx / frame_diagonal, "y": cy / frame_diagonal,
                                               "vx": velocity[0] if velocity else None,
                                               "vy": velocity[1] if velocity else None})
                    else:
                        # If IDs are unavailable, this is a detection count, not a unique count.
                        class_counts[name] += 1

            # Potential near-miss radar: closest-approach heuristic for tracked pairs.
            # Coordinates/speeds are normalized image-space values, not real-world metres.
            valid_tracks = [t for t in current_tracks if t["vx"] is not None and t["vy"] is not None]
            for a_idx in range(len(valid_tracks)):
                a = valid_tracks[a_idx]
                for b in valid_tracks[a_idx + 1:]:
                    rx, ry = b["x"] - a["x"], b["y"] - a["y"]
                    rvx, rvy = b["vx"] - a["vx"], b["vy"] - a["vy"]
                    rel_speed_sq = rvx * rvx + rvy * rvy
                    current_distance = (rx * rx + ry * ry) ** 0.5
                    if rel_speed_sq < 1e-7 or current_distance > 0.09:
                        continue
                    t_cpa = -(rx * rvx + ry * rvy) / rel_speed_sq
                    if not (0.0 < t_cpa <= 1.5):
                        continue
                    closest_x, closest_y = rx + rvx * t_cpa, ry + rvy * t_cpa
                    closest_distance = (closest_x * closest_x + closest_y * closest_y) ** 0.5
                    if closest_distance > 0.025:
                        continue
                    pair_key = "|".join(sorted([a["key"], b["key"]]))
                    # Keep the strongest observed estimate for a pair to avoid one event per frame.
                    previous_event = near_miss_events.get(pair_key)
                    if previous_event is None or closest_distance < previous_event["closest_distance_norm"]:
                        risk = "High" if closest_distance < 0.012 and t_cpa < 0.8 else "Moderate"
                        near_miss_events[pair_key] = {
                            "time_seconds": round(elapsed_video, 1),
                            "vehicle_a": a["class_name"], "vehicle_b": b["class_name"],
                            "time_to_closest_seconds": round(t_cpa, 2),
                            "current_distance_norm": round(current_distance, 4),
                            "closest_distance_norm": round(closest_distance, 4),
                            "risk": risk,
                            "note": "Image-space trajectory proximity; review the clip manually."
                        }
                    # Mark candidate pair on the annotated video for review.
                    ax, ay = int(a["x"] * frame_diagonal), int(a["y"] * frame_diagonal)
                    bx, by = int(b["x"] * frame_diagonal), int(b["y"] * frame_diagonal)
                    cv2.line(annotated, (ax, ay), (bx, by), (0, 60, 255), 2, cv2.LINE_AA)
                    cv2.putText(annotated, "POTENTIAL CONFLICT", (max(8, min(ax, bx)), max(24, min(ay, by) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 60, 255), 2, cv2.LINE_AA)

            # Accumulate three region-level metrics for this sampled frame.
            for region_index, region_name in enumerate(region_names):
                region_x1 = int(area_x1 + region_index * region_width)
                region_x2 = int(area_x1 + (region_index + 1) * region_width)
                region_area = max(1, (region_x2 - region_x1) * area_height)
                occupancy = min(100.0, frame_regions[region_name]["box_area"] / region_area * 100.0)
                speeds = frame_regions[region_name]["speeds"]
                avg_speed = sum(speeds) / len(speeds) if speeds else None
                acc = region_accumulator[region_name]
                acc["visible_sum"] += frame_regions[region_name]["count"]
                acc["occupancy_sum"] += occupancy
                acc["frame_count"] += 1
                if avg_speed is not None:
                    acc["speed_sum"] += avg_speed
                    acc["speed_count"] += 1
                    if avg_speed < 0.025:
                        acc["slow_count"] += 1
                per_frame_region_counts[region_name].append(frame_regions[region_name]["count"])
                per_frame_region_occupancy[region_name].append(occupancy)
                if avg_speed is not None:
                    per_frame_region_speeds[region_name].append(avg_speed)

            # Draw analysis area and region separators to make the measured areas visible.
            cv2.rectangle(annotated, (area_x1, area_y1), (area_x2, area_y2), (70, 190, 255), 1)
            for region_index, region_name in enumerate(region_names):
                label_x = int(area_x1 + region_index * region_width + 8)
                cv2.putText(annotated, region_name.upper(), (label_x, area_y1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 190, 255), 2, cv2.LINE_AA)
                if region_index > 0:
                    divider_x = int(area_x1 + region_index * region_width)
                    cv2.line(annotated, (divider_x, area_y1), (divider_x, area_y2), (70, 190, 255), 1, cv2.LINE_AA)

            processed_frames += 1
            peak_visible = max(peak_visible, visible_now)
            samples.append({"seconds": round(elapsed_video, 1), "visible": visible_now, "roi_visible": vehicles_in_area})
            roi_counts.append({"seconds": elapsed_video, "visible": vehicles_in_area})
            cv2.putText(annotated, f"RoadPulse AI | visible: {visible_now}", (18, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(annotated)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if processed_frames == 0:
        raise ValueError("No readable frames were found in this video.")

    avg_visible = sum(s["visible"] for s in samples) / len(samples)
    density, explanation, severity = classify_density(avg_visible, peak_visible)

    # Congestion Detective: compare first and last thirds, and the same central region.
    window = max(1, len(samples) // 3)
    early_samples = samples[:window]
    late_samples = samples[-window:]
    early_avg = sum(s["visible"] for s in early_samples) / len(early_samples)
    late_avg = sum(s["visible"] for s in late_samples) / len(late_samples)
    trend_delta = late_avg - early_avg
    vehicle_count_increasing = len(samples) >= 6 and trend_delta >= 1.5
    slow_cutoff = 0.025
    slow_candidates = [m for m in motion_samples if m["normalized_speed"] < slow_cutoff]
    slow_share = len(slow_candidates) / len(motion_samples) if motion_samples else None
    slow_movement = slow_share is not None and len(motion_samples) >= 5 and slow_share >= 0.55
    early_roi = roi_counts[:window]
    late_roi = roi_counts[-window:]
    early_roi_avg = sum(s["visible"] for s in early_roi) / len(early_roi)
    late_roi_avg = sum(s["visible"] for s in late_roi) / len(late_roi)
    roi_delta = late_roi_avg - early_roi_avg
    queue_growing = len(samples) >= 6 and roi_delta >= max(1.5, early_roi_avg * 0.25)
    signals_detected = sum([vehicle_count_increasing, slow_movement, queue_growing])
    possible_bottleneck = signals_detected >= 2
    bottleneck_evidence = [
        f"Vehicle presence {'increased' if vehicle_count_increasing else 'did not show a strong increase'} (first-third average {early_avg:.1f}, last-third average {late_avg:.1f}).",
        ("Slow movement could not be estimated reliably because there were not enough tracked vehicle movements." if slow_share is None
         else f"{slow_share * 100:.0f}% of measured tracked movements were below the provisional slow-motion threshold."),
        f"Central image region average changed from {early_roi_avg:.1f} to {late_roi_avg:.1f} visible vehicles per sampled frame."
    ]
    bottleneck_summary = ("Possible developing traffic bottleneck — review the highlighted evidence and video."
                          if possible_bottleneck else
                          "The available signals do not provide enough combined evidence to flag a developing bottleneck.")
    if vehicle_count_increasing:
        alert = "Vehicle presence increased during the clip; review the motion and queue indicators for possible congestion build-up."
        alert_type = "warning"
    elif len(samples) >= 6 and trend_delta <= -1.5:
        alert = "Vehicle presence decreased toward the end of the clip."
        alert_type = "good"
    else:
        alert = "No strong increase in visible vehicle count was detected in this clip."
        alert_type = "neutral"

    # Road-Space Intelligence: bounding-box area is a proxy for occupancy in each region.
    # The region covers image area, not a calibrated road-surface segmentation.
    region_results = []
    for region_name in region_names:
        acc = region_accumulator[region_name]
        frame_count = max(1, acc["frame_count"])
        avg_count = acc["visible_sum"] / frame_count
        avg_occupancy = acc["occupancy_sum"] / frame_count
        avg_speed = acc["speed_sum"] / acc["speed_count"] if acc["speed_count"] else None
        slow_frame_share = acc["slow_count"] / acc["speed_count"] * 100 if acc["speed_count"] else None
        region_results.append({
            "region": region_name,
            "average_visible_vehicles": round(avg_count, 2),
            "estimated_occupancy_percent": round(avg_occupancy, 2),
            "average_normalized_movement": round(avg_speed, 5) if avg_speed is not None else None,
            "slow_movement_share_percent": round(slow_frame_share, 1) if slow_frame_share is not None else None,
            "frames_measured": acc["frame_count"]
        })
    total_region_count = sum(r["average_visible_vehicles"] for r in region_results)
    for region in region_results:
        region["vehicle_distribution_percent"] = round(region["average_visible_vehicles"] / total_region_count * 100, 1) if total_region_count > 0 else 0.0
    available_speeds = [r["average_normalized_movement"] for r in region_results if r["average_normalized_movement"] is not None]
    uneven_movement = False
    movement_spread_percent = None
    if len(available_speeds) >= 2:
        max_speed, min_speed = max(available_speeds), min(available_speeds)
        movement_spread_percent = round((max_speed - min_speed) / max(max_speed, 1e-6) * 100, 1)
        uneven_movement = movement_spread_percent >= 50
    avg_occupancy = sum(r["estimated_occupancy_percent"] for r in region_results) / len(region_results)
    busiest_region = max(region_results, key=lambda r: r["average_visible_vehicles"])
    road_space_summary = (
        f"The {busiest_region['region'].lower()} region has the highest average visible count. "
        + ("Movement differs noticeably across regions." if uneven_movement else "No strong cross-region movement imbalance was detected or there was insufficient movement data.")
    )

    # Near-Miss Risk Radar summary: these are image-space candidate conflicts, not collision predictions.
    events = sorted(near_miss_events.values(), key=lambda e: (0 if e["risk"] == "High" else 1, e["time_seconds"]))
    high_events = sum(1 for e in events if e["risk"] == "High")
    moderate_events = sum(1 for e in events if e["risk"] == "Moderate")
    risk_level = "High" if high_events else "Moderate" if moderate_events else "No candidate detected"
    near_miss_summary = (
        f"Flagged {len(events)} potential close-trajectory interaction(s) for manual review."
        if events else
        "No candidate near-miss pattern met the current image-space thresholds in the sampled frames. This does not prove the road was risk-free."
    )

    duration = total_frames / fps if total_frames > 0 else frame_index / fps
    return {
        "total_unique_vehicles": int(sum(class_counts.values())),
        "vehicle_counts": {k: int(class_counts.get(k, 0)) for k in ["Car", "Motorcycle", "Bus", "Truck"]},
        "density": density,
        "severity": severity,
        "explanation": explanation,
        "alert": alert,
        "alert_type": alert_type,
        "bottleneck_analysis": {
            "vehicle_count_increasing": bool(vehicle_count_increasing),
            "slow_movement": bool(slow_movement),
            "queue_growing_same_section": bool(queue_growing),
            "possible_bottleneck": bool(possible_bottleneck),
            "signals_detected": int(signals_detected),
            "early_average_visible": round(early_avg, 2),
            "late_average_visible": round(late_avg, 2),
            "early_roi_average": round(early_roi_avg, 2),
            "late_roi_average": round(late_roi_avg, 2),
            "slow_movement_share_percent": round(slow_share * 100, 1) if slow_share is not None else None,
            "evidence": bottleneck_evidence,
            "summary": bottleneck_summary,
            "limitations": "Heuristic signals only. The region is a fixed image area, and speed is estimated in image coordinates. Camera perspective, camera movement, occlusion, and tracking errors can affect results."
        },
        "road_space_analysis": {
            "analysis_area_occupancy_percent": round(avg_occupancy, 2),
            "regions": region_results,
            "uneven_movement": bool(uneven_movement),
            "movement_spread_percent": movement_spread_percent,
            "summary": road_space_summary,
            "limitations": "Occupancy is estimated from detected vehicle bounding-box area inside three fixed image regions, not from a segmented or calibrated road surface. Normalized movement is not km/h. Check that the road is framed inside the analysis region."
        },
        "near_miss_radar": {
            "candidate_count": len(events),
            "high_count": high_events,
            "moderate_count": moderate_events,
            "risk_level": risk_level,
            "events": events[:10],
            "summary": near_miss_summary,
            "limitations": "This is a preliminary closest-approach heuristic using 2D image-space tracks. It is not a collision probability, does not understand right-of-way, and can produce false positives/negatives due to perspective, tracking errors, occlusion, and sparse sampling. Every flag needs human review."
        },
        "average_visible": round(avg_visible, 2),
        "peak_visible": int(peak_visible),
        "duration_seconds": round(duration, 1),
        "processed_frames": int(processed_frames),
        "timeline": samples[::max(1, len(samples)//30)],
        "method_note": "Traffic levels and road-space metrics use provisional image-space heuristics, not calibrated road capacity. Vehicle counts and motion estimates can be affected by occlusion, camera angle, lighting, camera movement, and tracking errors.",
        "video_url": f"/results/{output_path.name}"
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "Please choose a road video first."}), 400
    file = request.files["video"]
    if not file or not file.filename:
        return jsonify({"error": "Please choose a road video first."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use MP4, AVI, MOV, MKV, or WEBM."}), 400

    safe_name = secure_filename(file.filename)
    token = uuid.uuid4().hex[:10]
    input_path = UPLOAD_DIR / f"{token}_{safe_name}"
    output_path = RESULT_DIR / f"analyzed_{token}.mp4"
    file.save(input_path)
    try:
        data = analyze_video(input_path, output_path, request.form.get("analysis_speed", "balanced"))
        data["filename"] = safe_name
        return jsonify(data)
    except ValueError as exc:
        output_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        return jsonify({"error": str(exc)}), 500
    finally:
        input_path.unlink(missing_ok=True)


@app.route("/results/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULT_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    print("RoadPulse AI is starting at http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
