# RoadPulse AI — Traffic Video Intelligence

A student-project prototype that accepts an uploaded road video, uses a pretrained YOLOv8 detector plus ByteTrack tracking to detect/track common vehicle classes, produces an annotated video, summarizes unique tracked vehicles, and shows provisional traffic-level and vehicle-activity signals.

## Important scope and limitations

- This is a **video-analysis prototype**, not a validated traffic-control or reliable forecasting system.
- Traffic level is estimated using provisional visible-vehicle thresholds. Calibrate these thresholds for the camera/road and validate against manually labelled footage before making real-world claims.
- A rise/fall signal compares visible vehicle counts in the beginning and ending portions of the clip. It is **not** a trained congestion forecast.
- Detection can miss or misclassify vehicles because of camera angle, occlusion, lighting, resolution, and tracking errors.
- The provided CSV appears to contain one CCTV identifier (`cctv052`) and limited conditions. It is not itself a collection of usable video files and cannot train a general vehicle detector. Keep it for exploratory historical-label work; the video detector is a separate pretrained model.
- On first run, Ultralytics downloads `yolov8n.pt`; an internet connection is needed. Subsequent runs use the cached weights. You may need to accept the model's license/terms for your use case.
- Uploaded source files are deleted after processing. Annotated outputs remain in `results/` until you remove them.

## Requirements

- Python 3.10–3.12 recommended
- VS Code
- Internet for installing packages and the first model-weight download
- A short road video (`.mp4`, `.avi`, `.mov`, `.mkv`, or `.webm`), up to 200 MB

## Run on Windows in VS Code

Open this folder in VS Code, then choose **Terminal → New Terminal**.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

If PowerShell blocks activation, run this in the same terminal and retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Open http://127.0.0.1:5000 in your browser. Upload a short video and select **Run AI analysis**. The first analysis may take longer while model weights download.

## Run on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

## Project structure

```text
RoadPulse-AI/
├── app.py                 # Flask API + video processing
├── requirements.txt
├── README.md
├── templates/index.html   # Dashboard UI
├── static/style.css       # Responsive styling
├── static/app.js          # Upload + results interaction
├── uploads/               # Temporary uploads (deleted after analysis)
└── results/               # Annotated MP4 results
```

## Suggested next improvements

1. Label a representative set of frames and validate detector/count accuracy.
2. Calibrate density thresholds by road ROI, lane count, and camera viewpoint.
3. Add line-crossing counts, average speed estimates, queue-length estimation, and privacy-preserving retention rules.
4. Collect multiple cameras and longer time series before attempting forecasting.

## Run EDA on the provided traffic dataset

The project includes `data/traffic_dataset.csv` and `eda.py`. To explore the CSV metadata, activate the same virtual environment and run:

```powershell
python eda.py
```

The script creates an `eda_output/` folder containing `eda_report.txt`, CSV summaries, and PNG charts. It explores the dataset's metadata and labels; it does not analyze video pixels or validate the separate YOLO detector.


## Added intelligence features

### 1. Congestion Detective

After video analysis, the dashboard displays three heuristic signals: (1) whether visible vehicle counts increase from the first to last third of the clip, (2) whether tracked vehicle motion is slow in normalized image coordinates, and (3) whether detections in the same fixed central image region increase over the clip. When at least two signals are detected, the app flags a **possible developing traffic bottleneck** and lists the evidence.

### 2. Road-Space Intelligence

The image is divided into left, centre, and right regions inside a fixed analysis area. The dashboard estimates vehicle bounding-box occupancy, average visible detections, distribution of detections, and normalized movement per region. It can flag uneven movement when the measured regional movement indices differ noticeably.

**Important:** occupancy is the fraction of region pixels covered by detected vehicle bounding boxes. This is only an image-space proxy; it is not a calibrated road-surface occupancy measurement. Make sure the road is framed inside the analysis area. Perspective, overlapping boxes, and non-road background affect the result.

### 3. Near-Miss Risk Radar

The system compares tracked vehicle trajectories using a closest-approach heuristic. It flags candidate interactions when tracked paths are close and approaching within a short projected time window. Candidate timestamps, vehicle classes, and heuristic risk levels are shown in the dashboard; detected candidate pairs are marked in the annotated output video.

These are **potential conflict indicators**, not collision predictions or probabilities. The model does not understand right-of-way and uses 2D image coordinates rather than calibrated road coordinates. False positives and missed events are possible. Every event needs human review; do not use this prototype for live traffic control or enforcement.

### Shared limitations

All new indicators are preliminary image-space heuristics, not proof of a real-world cause. Camera perspective, camera movement, occlusion, lighting, sparse sampling, and tracking errors can affect the result. Calibrate thresholds and validate against manually reviewed clips before making operational claims. Balanced mode samples up to about 8 frames/second; Fast mode samples fewer frames for quicker preview and can miss short events.
