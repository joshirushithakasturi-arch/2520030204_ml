const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('videoInput');
const dropzone = document.getElementById('dropzone');
const fileTitle = document.getElementById('fileTitle');
const fileSubtitle = document.getElementById('fileSubtitle');
const previewWrap = document.getElementById('previewWrap');
const videoPreview = document.getElementById('videoPreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const progressWrap = document.getElementById('progressWrap');
const errorBox = document.getElementById('errorBox');
const outputPanel = document.getElementById('outputPanel');
let previewUrl = null;

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  fileTitle.textContent = file.name;
  fileSubtitle.textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MB · Ready to analyze`;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  videoPreview.src = previewUrl;
  previewWrap.classList.remove('hidden');
  errorBox.classList.add('hidden');
});
['dragenter', 'dragover'].forEach(eventName => dropzone.addEventListener(eventName, e => {
  e.preventDefault(); dropzone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach(eventName => dropzone.addEventListener(eventName, e => {
  e.preventDefault(); dropzone.classList.remove('dragover');
}));
dropzone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer(); transfer.items.add(file); fileInput.files = transfer.files;
  fileInput.dispatchEvent(new Event('change'));
});

form.addEventListener('submit', async e => {
  e.preventDefault();
  errorBox.classList.add('hidden');
  const file = fileInput.files[0];
  if (!file) return showError('Please choose a road video first.');
  if (file.size > 200 * 1024 * 1024) return showError('This file is larger than 200 MB. Please choose a shorter or smaller video.');
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(ext)) return showError('Unsupported video format. Use MP4, AVI, MOV, MKV, or WEBM.');

  const body = new FormData(); body.append('video', file);
  body.append('camera_name', document.getElementById('cameraName').value.trim());
  body.append('analysis_speed', document.getElementById('sampleRate').value);
  analyzeBtn.disabled = true;
  analyzeBtn.querySelector('span:first-child').textContent = 'Analyzing video…';
  progressWrap.classList.remove('hidden');
  outputPanel.classList.add('hidden');
  document.getElementById('progressText').textContent = 'Loading AI detector and analyzing sampled frames…';
  try {
    const response = await fetch('/analyze', { method: 'POST', body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed. Please try another video.');
    renderResults(data);
  } catch (err) {
    showError(err.message || 'Could not analyze this video.');
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.querySelector('span:first-child').textContent = 'Run AI analysis';
    progressWrap.classList.add('hidden');
  }
});

function showError(message) { errorBox.textContent = message; errorBox.classList.remove('hidden'); }
function renderResults(data) {
  document.getElementById('metricVehicles').textContent = data.total_unique_vehicles;
  document.getElementById('metricDensity').textContent = data.density;
  document.getElementById('metricPeak').textContent = data.peak_visible;
  document.getElementById('metricDuration').textContent = `${data.duration_seconds}s`;
  document.getElementById('countCar').textContent = data.vehicle_counts.Car;
  document.getElementById('countMoto').textContent = data.vehicle_counts.Motorcycle;
  document.getElementById('countBus').textContent = data.vehicle_counts.Bus;
  document.getElementById('countTruck').textContent = data.vehicle_counts.Truck;
  const alertCard = document.getElementById('alertCard');
  alertCard.className = `alert-card ${data.alert_type === 'warning' ? 'warning' : data.alert_type === 'good' ? 'good' : ''}`;
  document.getElementById('alertTitle').textContent = data.alert_type === 'warning' ? 'Congestion build-up signal' : data.alert_type === 'good' ? 'Traffic easing signal' : 'Traffic activity signal';
  document.getElementById('alertText').textContent = data.alert;
  document.getElementById('resultHeading').textContent = data.filename || 'Video analysis results';
  document.getElementById('resultSubheading').textContent = document.getElementById('cameraName').value.trim() || 'Annotated output and traffic signals from your clip.';
  const videoUrl = `${data.video_url}?t=${Date.now()}`;
  const resultVideo = document.getElementById('resultVideo'); resultVideo.src = videoUrl; resultVideo.load();
  const download = document.getElementById('downloadVideo'); download.href = data.video_url; download.download = `roadpulse-${Date.now()}.mp4`;
  const badge = document.getElementById('densityBadge'); badge.className = `density-badge ${data.severity === 'medium' ? 'medium' : data.severity === 'high' ? 'heavy' : ''}`;
  document.getElementById('resultDensity').textContent = `${data.density} traffic`;
  document.getElementById('resultExplanation').textContent = data.explanation;
  document.getElementById('resultAverage').textContent = data.average_visible;
  document.getElementById('resultPeak').textContent = data.peak_visible;
  document.getElementById('resultFrames').textContent = data.processed_frames;
  document.getElementById('resultTrend').textContent = data.alert_type === 'warning' ? 'Increasing' : data.alert_type === 'good' ? 'Decreasing' : 'No strong change';
  document.getElementById('methodNote').textContent = data.method_note;
  renderBottleneckAnalysis(data.bottleneck_analysis || {});
  renderRoadSpace(data.road_space_analysis || {});
  renderNearMissRadar(data.near_miss_radar || {});
  drawTimeline(data.timeline || []);
  outputPanel.classList.remove('hidden');
  outputPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function drawTimeline(points) {
  const chart = document.getElementById('timelineChart'); chart.innerHTML = '';
  if (!points.length) { chart.innerHTML = '<span class="chart-empty">No timeline data available</span>'; return; }
  const max = Math.max(1, ...points.map(p => p.visible));
  points.forEach(point => {
    const bar = document.createElement('div'); bar.className = 'bar';
    bar.style.height = `${Math.max(4, (point.visible / max) * 100)}%`;
    bar.title = `${point.seconds}s: ${point.visible} visible vehicles`;
    chart.appendChild(bar);
  });
}


function renderBottleneckAnalysis(analysis) {
  const signals = [
    {
      card: 'signalIncreasing', status: 'signalIncreasing', text: 'signalIncreasingText',
      detected: Boolean(analysis.vehicle_count_increasing),
      detail: `First-third average: ${analysis.early_average_visible ?? '—'} vehicles; last-third average: ${analysis.late_average_visible ?? '—'} vehicles.`
    },
    {
      card: 'signalSlow', status: 'signalSlow', text: 'signalSlowText',
      detected: Boolean(analysis.slow_movement),
      detail: analysis.slow_movement_share_percent == null
        ? 'Not enough reliable tracked movement samples to estimate speed.'
        : `${analysis.slow_movement_share_percent}% of measured tracked movements were below the provisional slow-motion threshold.`
    },
    {
      card: 'signalQueue', status: 'signalQueue', text: 'signalQueueText',
      detected: Boolean(analysis.queue_growing_same_section),
      detail: `Central region average: ${analysis.early_roi_average ?? '—'} early → ${analysis.late_roi_average ?? '—'} late vehicles per sampled frame.`
    }
  ];

  signals.forEach(signal => {
    const card = document.getElementById(signal.card);
    const status = card.querySelector('.signal-status');
    card.classList.toggle('detected', signal.detected);
    card.classList.toggle('not-detected', !signal.detected);
    status.textContent = signal.detected ? '✓' : '–';
    document.getElementById(signal.text).textContent = signal.detail;
  });

  const verdict = document.getElementById('bottleneckVerdict');
  verdict.classList.toggle('possible', Boolean(analysis.possible_bottleneck));
  document.getElementById('bottleneckTitle').textContent = analysis.possible_bottleneck
    ? `Possible cause: developing traffic bottleneck (${analysis.signals_detected || 0}/3 signals)`
    : `Bottleneck not confirmed (${analysis.signals_detected || 0}/3 signals)`;
  document.getElementById('bottleneckSummary').textContent = analysis.summary || 'No bottleneck assessment is available.';

  const list = document.getElementById('bottleneckEvidence');
  list.innerHTML = '';
  (analysis.evidence || []).forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    list.appendChild(li);
  });
  if (!list.children.length) {
    const li = document.createElement('li');
    li.textContent = 'No evidence details were returned for this clip.';
    list.appendChild(li);
  }
  document.getElementById('bottleneckLimitations').textContent = analysis.limitations ||
    'These are heuristic indicators, not proof of the cause of congestion.';
}


function renderRoadSpace(analysis) {
  document.getElementById('overallOccupancy').textContent = analysis.analysis_area_occupancy_percent == null
    ? '—' : `${analysis.analysis_area_occupancy_percent}%`;
  document.getElementById('movementBalance').textContent = analysis.uneven_movement === true
    ? 'Uneven' : analysis.uneven_movement === false ? 'No strong imbalance' : '—';
  document.getElementById('roadSpaceSummary').textContent = analysis.summary || 'No road-space summary is available.';
  document.getElementById('roadSpaceLimitations').textContent = analysis.limitations || 'Occupancy is an image-space estimate.';
  const grid = document.getElementById('regionGrid');
  grid.innerHTML = '';
  (analysis.regions || []).forEach(region => {
    const card = document.createElement('div');
    card.className = 'region-card';
    const title = document.createElement('span'); title.textContent = region.region.toUpperCase();
    const count = document.createElement('strong'); count.textContent = `${region.estimated_occupancy_percent}%`;
    const detail = document.createElement('small');
    const speed = region.average_normalized_movement == null ? 'Movement: insufficient samples' : `Motion index: ${region.average_normalized_movement}`;
    detail.textContent = `${region.average_visible_vehicles} visible/frame · ${region.vehicle_distribution_percent}% of regional detections · ${speed}`;
    const meter = document.createElement('div'); meter.className = 'occupancy-meter';
    const fill = document.createElement('span'); fill.style.width = `${Math.max(0, Math.min(100, region.estimated_occupancy_percent || 0))}%`;
    meter.appendChild(fill); card.append(title, count, meter, detail); grid.appendChild(card);
  });
  if (!grid.children.length) grid.innerHTML = '<p class="empty-note">No regional measurements were returned.</p>';
}

function renderNearMissRadar(analysis) {
  const banner = document.getElementById('riskBanner');
  banner.classList.remove('risk-high', 'risk-moderate', 'risk-none');
  const level = analysis.risk_level || 'No result';
  banner.classList.add(level === 'High' ? 'risk-high' : level === 'Moderate' ? 'risk-moderate' : 'risk-none');
  document.getElementById('riskLevel').textContent = `${level} risk indicator`;
  document.getElementById('riskSummary').textContent = analysis.summary || 'No risk summary is available.';
  document.getElementById('riskCandidateCount').textContent = analysis.candidate_count ?? '—';
  document.getElementById('riskHighCount').textContent = analysis.high_count ?? '—';
  document.getElementById('riskModerateCount').textContent = analysis.moderate_count ?? '—';
  document.getElementById('nearMissLimitations').textContent = analysis.limitations || 'Potential conflict flags require human review.';
  const list = document.getElementById('nearMissEvents');
  list.innerHTML = '';
  (analysis.events || []).forEach((event, index) => {
    const row = document.createElement('div'); row.className = 'event-row';
    const left = document.createElement('div');
    const eventSeconds = Math.max(0, Math.floor(event.time_seconds || 0));
    const stamp = `${String(Math.floor(eventSeconds / 60)).padStart(2, '0')}:${String(eventSeconds % 60).padStart(2, '0')}`;
    const title = document.createElement('strong'); title.textContent = `${stamp} · ${event.vehicle_a} + ${event.vehicle_b}`;
    const detail = document.createElement('small'); detail.textContent = `Closest approach estimate: ${event.time_to_closest_seconds}s ahead · Review clip manually`;
    left.append(title, detail);
    const badge = document.createElement('span'); badge.className = `event-badge ${event.risk === 'High' ? 'high' : 'moderate'}`; badge.textContent = event.risk;
    row.append(left, badge); list.appendChild(row);
  });
  if (!list.children.length) list.innerHTML = '<p class="empty-note">No candidate interactions met the current thresholds. This does not prove the video is risk-free.</p>';
}
