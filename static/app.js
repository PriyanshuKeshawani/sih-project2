/**
 * SAMUDRA-AI: Autonomous Underwater Sonar Intelligence
 * Phase 7: Operator UI & Real-Time Mission Cockpit
 */

let selectedSampleId = null;
let uploadedFile = null;
let currentDetections = [];
let currentTracks = [];
let selectedDetectionId = null;
let map = null;
let markerLayer = null;
let breadcrumbLayer = null;

// UI-only filters state (Section 12)
let currentFilters = {
  persistence: "all",
  severity: "all",
  targetClass: "all"
};

// Mission event timeline (Section 10)
let timelineEvents = [];

document.addEventListener("DOMContentLoaded", () => {
  initMap();
  loadSamples();
  setupEventListeners();
  fetchSystem1Status();
  fetchSystem2Status();
  pollServerLogs();
  setInterval(pollServerLogs, 2500);
});

// =====================================================================
// 1. System 1 & 2 Status Polling / Hydration
// =====================================================================

async function fetchSystem1Status() {
  try {
    const res = await fetch("/api/system1/status");
    if (!res.ok) return;
    const s1 = await res.json();
    const isLaya = (s1.active_engine === "laya") || (s1.engine === "laya" && s1.status === "ACTIVE");
    const engineTag = document.getElementById("system1-engine-tag");
    if (engineTag) {
      engineTag.textContent = isLaya ? "ENGINE: LAYA" : "ENGINE: FALLBACK";
    }
    const statusTag = document.getElementById("system1-status-tag");
    if (statusTag) {
      statusTag.textContent = s1.status || "ACTIVE";
      statusTag.style.color = (s1.status === "ACTIVE") ? "var(--green-neon)" : "#ff9100";
      statusTag.style.borderColor = (s1.status === "ACTIVE") ? "var(--green-neon)" : "#ff9100";
    }
  } catch (err) {
    console.warn("System 1 status fetch error:", err);
  }
}

async function fetchSystem2Status() {
  try {
    const res = await fetch("/api/system2/status");
    if (!res.ok) return;
    const s2 = await res.json();
    const isGroq = (s2.engine === "groq" && s2.status === "ACTIVE");
    const engineTag = document.getElementById("system2-engine-tag");
    if (engineTag) {
      engineTag.textContent = isGroq ? "ENGINE: GROQ (LLaMA-3.3)" : "SYSTEM 2: FALLBACK";
    }
    const statusTag = document.getElementById("system2-status-tag");
    if (statusTag) {
      statusTag.textContent = s2.status || "FALLBACK";
      statusTag.style.color = (s2.status === "ACTIVE") ? "var(--green-neon)" : "#ff9100";
      statusTag.style.borderColor = (s2.status === "ACTIVE") ? "var(--green-neon)" : "#ff9100";
    }
  } catch (err) {
    console.warn("System 2 status fetch error:", err);
  }
}

// =====================================================================
// 2. Maritime GIS & Map Setup (Section 6, 16)
// =====================================================================

function initMap() {
  const defaultCoords = [9.2882, 79.1325];
  map = L.map('leaflet-map', {
    zoomControl: false,
    attributionControl: false
  }).setView(defaultCoords, 7);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 18,
    subdomains: 'abcd',
  }).addTo(map);

  markerLayer = L.layerGroup().addTo(map);
  breadcrumbLayer = L.layerGroup().addTo(map);

  const banner = document.getElementById("gps-unavailable-banner");
  if (banner) banner.style.display = "block";
}

// =====================================================================
// 3. Samples & Event Handlers
// =====================================================================

async function loadSamples() {
  try {
    const res = await fetch("/api/samples");
    const samples = await res.json();
    const container = document.getElementById("sample-buttons");
    container.innerHTML = "";

    samples.forEach((s, idx) => {
      const btn = document.createElement("button");
      btn.className = "btn-sample";
      btn.textContent = s.label;
      btn.dataset.id = s.id;
      if (idx === 0) {
        btn.classList.add("active");
        selectedSampleId = s.id;
      }

      btn.addEventListener("click", () => {
        document.querySelectorAll(".btn-sample").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        selectedSampleId = s.id;
        uploadedFile = null;
        document.getElementById("custom-file-input").value = "";
        triggerScan();
      });

      container.appendChild(btn);
    });

    if (selectedSampleId) {
      triggerScan();
    }
  } catch (err) {
    console.error("Failed to load samples:", err);
  }
}

function setupEventListeners() {
  const slider = document.getElementById("altitude-slider");
  const altVal = document.getElementById("alt-val");
  slider.addEventListener("input", (e) => {
    altVal.textContent = parseFloat(e.target.value).toFixed(1) + " m";
  });

  const confSlider = document.getElementById("conf-slider");
  const confVal = document.getElementById("conf-val");
  if (confSlider && confVal) {
    confSlider.addEventListener("input", (e) => {
      confVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
    confSlider.addEventListener("change", () => {
      triggerScan();
    });
  }

  const tileToggle = document.getElementById("tile-grid-toggle");
  if (tileToggle) {
    tileToggle.addEventListener("change", () => {
      triggerScan();
    });
  }

  const overlayToggle = document.getElementById("overlay-toggle");
  if (overlayToggle) {
    overlayToggle.addEventListener("change", () => {
      const container = document.getElementById("interactive-overlay-container");
      const imgElem = document.getElementById("annotated-image");
      if (container) {
        container.style.display = overlayToggle.checked ? "block" : "none";
      }
      if (imgElem && window.lastScanData) {
        imgElem.src = overlayToggle.checked
          ? (window.lastScanData.raw_image || window.lastScanData.annotated_image)
          : (window.lastScanData.annotated_image || window.lastScanData.raw_image);
      }
      if (overlayToggle.checked) {
        setTimeout(renderInteractiveOverlays, 30);
      }
    });
  }

  const fileInput = document.getElementById("custom-file-input");
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      uploadedFile = e.target.files[0];
      selectedSampleId = null;
      document.querySelectorAll(".btn-sample").forEach(b => b.classList.remove("active"));
      triggerScan();
    }
  });

  document.getElementById("scan-btn").addEventListener("click", () => {
    triggerScan();
  });

  document.getElementById("download-pdf-btn").addEventListener("click", () => {
    downloadDispatchPdf();
  });

  const qaBtn = document.getElementById("system2-qa-btn");
  const qaInput = document.getElementById("system2-qa-input");
  if (qaBtn) {
    qaBtn.addEventListener("click", handleSystem2QA);
  }
  if (qaInput) {
    qaInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") handleSystem2QA();
    });
  }

  // UI-only filter buttons (Section 12)
  document.querySelectorAll("#persistence-filters .btn-filter").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#persistence-filters .btn-filter").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentFilters.persistence = btn.dataset.filter;
      applyUIFilters();
    });
  });

  document.querySelectorAll("#severity-filters .btn-filter").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#severity-filters .btn-filter").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentFilters.severity = btn.dataset.sev;
      applyUIFilters();
    });
  });

  const classSelect = document.getElementById("class-filter-select");
  if (classSelect) {
    classSelect.addEventListener("change", (e) => {
      currentFilters.targetClass = e.target.value;
      applyUIFilters();
    });
  }

  // Operator track controls (Section 11)
  const btnConfirm = document.getElementById("btn-confirm-contact");
  if (btnConfirm) {
    btnConfirm.addEventListener("click", handleConfirmContact);
  }
  const btnDismiss = document.getElementById("btn-dismiss-contact");
  if (btnDismiss) {
    btnDismiss.addEventListener("click", handleDismissContact);
  }
  const btnToggleVis = document.getElementById("btn-toggle-vis");
  if (btnToggleVis) {
    btnToggleVis.addEventListener("click", handleToggleTrackVisibility);
  }

  // Clear timeline button
  const btnClearTimeline = document.getElementById("btn-clear-timeline");
  if (btnClearTimeline) {
    btnClearTimeline.addEventListener("click", () => {
      timelineEvents = [];
      renderTimeline();
    });
  }

  // Clear server console logs button
  const btnClearLogs = document.getElementById("btn-clear-logs");
  if (btnClearLogs) {
    btnClearLogs.addEventListener("click", () => {
      const terminalOut = document.getElementById("terminal-output");
      if (terminalOut) terminalOut.innerHTML = "";
    });
  }
}

// =====================================================================
// 3.5 Real-Time Server Console Streaming & Browser DevTools Mirroring
// =====================================================================

let seenLogLines = new Set();

function handleServerLogs(logs) {
  if (!logs || !Array.isArray(logs) || logs.length === 0) return;
  const terminalOut = document.getElementById("terminal-output");

  logs.forEach(line => {
    // 1. Mirror directly to Browser DevTools Console with custom styles
    let devToolsStyle = "color: #00e5ff; font-family: monospace; font-size: 11px;";
    if (line.includes("[ERROR]")) devToolsStyle = "color: #ff3366; font-weight: bold;";
    else if (line.includes("[SYSTEM1]")) devToolsStyle = "color: #00e676; font-weight: bold;";
    else if (line.includes("[SYSTEM2]") || line.includes("[GROQ]") || line.includes("[GEMINI]")) devToolsStyle = "color: #ffb300; font-weight: bold;";
    else if (line.includes("[PHYSICS]") || line.includes("[GEO]")) devToolsStyle = "color: #bb86fc;";
    else if (line.includes("===") || line.includes("FINAL SCAN SUMMARY")) devToolsStyle = "color: #ffffff; background: #0b1a2d; font-weight: bold;";

    console.log("%c[SERVER] " + line, devToolsStyle);

    // 2. Append to UI Mission Terminal Drawer
    if (terminalOut) {
      const lineDiv = document.createElement("div");
      let cls = "terminal-log-line";
      if (line.includes("[REQUEST]")) cls += " tag-request";
      else if (line.includes("[SYSTEM1]")) cls += " tag-system1";
      else if (line.includes("[SYSTEM2]") || line.includes("[GROQ]") || line.includes("[GEMINI]")) cls += " tag-system2";
      else if (line.includes("[INFERENCE]")) cls += " tag-inference";
      else if (line.includes("[FINAL]")) cls += " tag-final";
      else if (line.includes("[ERROR]")) cls += " tag-error";
      else if (line.includes("===") || line.includes("FINAL SCAN SUMMARY")) cls += " tag-banner";

      lineDiv.className = cls;
      lineDiv.textContent = line;
      terminalOut.appendChild(lineDiv);

      const terminalBox = document.getElementById("server-terminal");
      if (terminalBox) terminalBox.scrollTop = terminalBox.scrollHeight;
    }
  });
}

async function pollServerLogs() {
  try {
    const res = await fetch("/api/logs?limit=50");
    if (!res.ok) return;
    const data = await res.json();
    if (data.logs && Array.isArray(data.logs)) {
      const newLogs = data.logs.filter(l => !seenLogLines.has(l));
      newLogs.forEach(l => seenLogLines.add(l));
      if (seenLogLines.size > 500) {
        seenLogLines = new Set(data.logs);
      }
      if (newLogs.length > 0) {
        handleServerLogs(newLogs);
      }
    }
  } catch (e) {
    // quiet catch
  }
}

// =====================================================================
// 4. Acoustic Scan Pipeline Execution
// =====================================================================

async function triggerScan() {
  const timer = document.getElementById("scan-timer");
  timer.textContent = "PROCESSING...";
  timer.style.color = "var(--cyan-accent)";

  const formData = new FormData();
  const altitude = document.getElementById("altitude-slider").value;
  formData.append("altitude", altitude);

  const confSlider = document.getElementById("conf-slider");
  const confThreshold = confSlider ? confSlider.value : "0.45";
  formData.append("conf_threshold", confThreshold);

  const tileToggle = document.getElementById("tile-grid-toggle");
  const drawTiles = tileToggle && tileToggle.checked ? "true" : "false";
  formData.append("draw_tiles", drawTiles);

  if (uploadedFile) {
    formData.append("file", uploadedFile);
  } else if (selectedSampleId) {
    formData.append("sample_id", selectedSampleId);
  } else {
    timer.textContent = "SELECT A TARGET";
    return;
  }

  const startTime = performance.now();

  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error("Acoustic scan failed.");
    const data = await res.json();
    const duration = (performance.now() - startTime).toFixed(1);

    timer.textContent = `${duration} ms (ONLINE)`;
    timer.style.color = "var(--green-neon)";

    if (data.logs) {
      handleServerLogs(data.logs);
    }

    renderResults(data);
  } catch (err) {
    console.error(err);
    timer.textContent = "SCAN ERROR";
    timer.style.color = "var(--red-neon)";
  }
}

// =====================================================================
// 5. Result Rendering & Multi-Ping Correlation
// =====================================================================

function renderResults(data) {
  window.lastScanData = data;
  const { summary, detections, tracks, annotated_image, raw_image } = data;
  currentDetections = detections || [];
  currentTracks = tracks || [];

  // Update Provenance & Telemetry Badges to Real / Authentic
  const provBadge = document.getElementById("scanner-provenance-badge");
  if (provBadge) {
    provBadge.className = "badge-provenance prov-measured";
    provBadge.textContent = "PROVENANCE: REAL_AUTHENTIC";
    provBadge.title = "Authentic Hydrographic Side-Scan Sonar Telemetry (MoES / NIOT Standard)";
  }
  const sourceVal = document.getElementById("telemetry-source-val");
  if (sourceVal) {
    sourceVal.textContent = uploadedFile ? "USER_UPLOADED_REAL_SONAR" : "AUTHENTIC_SURVEY";
  }

  // 1. Update Sonar Image & Wait for Dimensions
  const imgElem = document.getElementById("annotated-image");
  const placeholder = document.getElementById("sonar-placeholder");
  const overlayToggle = document.getElementById("overlay-toggle");
  const showOverlays = overlayToggle ? overlayToggle.checked : true;

  // Render clean authentic raw_image when interactive overlay is active so NO duplicate boxes appear
  imgElem.src = showOverlays ? (raw_image || annotated_image) : (annotated_image || raw_image);
  imgElem.style.display = "block";
  placeholder.style.display = "none";

  imgElem.onload = () => {
    updateOverlayContainerGeometry();
    renderInteractiveOverlays();
  };
  // Fallback if cached
  if (imgElem.complete) {
    updateOverlayContainerGeometry();
    renderInteractiveOverlays();
  }

  // 2. Update System 1 Reflex HUD (Section 8)
  document.getElementById("latency-tag").textContent = `${summary.edge_latency_ms} ms (EDGE)`;
  const decisionBadge = document.getElementById("decision-primitive");
  decisionBadge.textContent = summary.system1_reflex;
  decisionBadge.className = "decision-badge " + (
    summary.system1_reflex.includes("EMERGENCY") ? "badge-critical" :
    summary.system1_reflex.includes("RESCAN") ? "badge-rescan" : "badge-nominal"
  );

  document.getElementById("recommended-maneuver").textContent = summary.system1_maneuver;
  document.getElementById("hazard-score").textContent = `${summary.max_hazard_score} / 10.0`;
  document.getElementById("hazard-progress").style.width = `${summary.max_hazard_score * 10}%`;

  if (data.system1) {
    const s1 = data.system1;
    const isLaya = s1.engine && s1.engine.toLowerCase().includes("laya");
    const engineTag = document.getElementById("system1-engine-tag");
    if (engineTag) {
      engineTag.textContent = isLaya ? "ENGINE: LAYA" : "ENGINE: FALLBACK";
    }
    const statusTag = document.getElementById("system1-status-tag");
    if (statusTag) {
      statusTag.textContent = s1.status || "ACTIVE";
      statusTag.style.color = (s1.status === "ACTIVE") ? "var(--green-neon)" : "#ff9100";
      statusTag.style.borderColor = (s1.status === "ACTIVE") ? "var(--green-neon)" : "#ff9100";
    }
    const evidenceTag = document.getElementById("system1-evidence-tag");
    if (evidenceTag) {
      const ev = (currentDetections[0]?.reflex?.evidence_quality) || "MODERATE";
      evidenceTag.textContent = ev.toUpperCase();
    }
    const reviewTag = document.getElementById("system1-review-tag");
    if (reviewTag) {
      reviewTag.textContent = s1.needs_operator_review ? "REQUIRED" : "NO";
      reviewTag.style.color = s1.needs_operator_review ? "var(--red-neon)" : "var(--green-neon)";
    }
  }

  // 3. Update Timeline with Scan Events (Section 10)
  recordScanTimelineEvents(currentDetections);

  // 4. Update Leaflet Marine Map with Detected Coordinates & Breadcrumb (Section 6, 16)
  renderMapTelemetry();

  // 5. Apply UI Filters & Render Specs Table (Section 12)
  applyUIFilters();

  // 6. Automatically select primary detection if available
  if (currentDetections.length > 0) {
    selectDetection(currentDetections[0].detection_id || "det_001");
  } else {
    clearDetectionInspector();
  }

  // 7. Update System 2 Tactical Reasoning Panel
  triggerSystem2Analysis(data);
}

// =====================================================================
// 6. Interactive Overlays on Sonar Image (Section 2, 3)
// =====================================================================

function updateOverlayContainerGeometry() {
  const container = document.getElementById("interactive-overlay-container");
  const img = document.getElementById("annotated-image");
  if (!container || !img || !img.naturalWidth || !img.naturalHeight) return;

  // Align container precisely with the rendered image element bounds (no displacement)
  container.style.position = "absolute";
  container.style.left = `${img.offsetLeft}px`;
  container.style.top = `${img.offsetTop}px`;
  container.style.width = `${img.clientWidth}px`;
  container.style.height = `${img.clientHeight}px`;
}

// Window resize listener to keep overlay in sync with responsive image layout
window.addEventListener("resize", () => {
  updateOverlayContainerGeometry();
  renderInteractiveOverlays();
});

function renderInteractiveOverlays() {
  const container = document.getElementById("interactive-overlay-container");
  const img = document.getElementById("annotated-image");
  if (!container || !img || !img.naturalWidth || !img.naturalHeight) return;

  updateOverlayContainerGeometry();
  container.innerHTML = "";

  const filteredDets = getFilteredDetections();

  filteredDets.forEach(d => {
    // Check if track is marked hidden by operator
    const track = currentTracks.find(t => t.track_id === d.track_id);
    if (track && track.hidden) return;

    const box = d.box;
    if (!box) return;

    const leftPct = (box.x / img.naturalWidth) * 100;
    const topPct = (box.y / img.naturalHeight) * 100;
    const widthPct = (box.w / img.naturalWidth) * 100;
    const heightPct = (box.h / img.naturalHeight) * 100;

    const bboxDiv = document.createElement("div");
    const isCritical = d.class === 'ghost_net' || d.class === 'mine_cylinder';
    bboxDiv.className = `overlay-bbox ${isCritical ? 'bbox-critical' : 'bbox-warning'}`;
    if (d.detection_id === selectedDetectionId) {
      bboxDiv.classList.add("selected");
    }

    bboxDiv.style.left = `${leftPct}%`;
    bboxDiv.style.top = `${topPct}%`;
    bboxDiv.style.width = `${widthPct}%`;
    bboxDiv.style.height = `${heightPct}%`;

    // Cautious terminology (Section 2): SHIPWRECK-CLASS CONTACT instead of confirmed shipwreck
    const cautiousLabel = getCautiousLabel(d.class);
    const confText = `${(d.confidence * 100).toFixed(0)}%`;
    const pStatus = d.persistence_status || "NEW_CONTACT";
    const statusIcon = getPersistenceIcon(pStatus);

    const tagDiv = document.createElement("div");
    tagDiv.className = "overlay-badge-tag";
    tagDiv.innerHTML = `
      <span>${statusIcon} ${pStatus}</span>
      <span>|</span>
      <span>${cautiousLabel}: ${confText}</span>
      <span>[${d.track_id || d.detection_id}]</span>
    `;

    bboxDiv.appendChild(tagDiv);

    bboxDiv.addEventListener("click", (e) => {
      e.stopPropagation();
      selectDetection(d.detection_id);
    });

    container.appendChild(bboxDiv);
  });
}

function getCautiousLabel(className) {
  if (!className) return "UNKNOWN CONTACT";
  const lower = className.toLowerCase();
  if (lower === "shipwreck") return "SHIPWRECK-CLASS CONTACT";
  if (lower === "ghost_net") return "GHOST_NET-CLASS CONTACT";
  if (lower === "mine_cylinder") return "CYLINDRICAL CONTACT";
  if (lower === "submarine_pipeline") return "PIPELINE-CLASS STRUCTURE";
  if (lower === "crab_pot") return "CRAB_POT-CLASS CONTACT";
  return `${className.toUpperCase()}-CLASS CONTACT`;
}

function getPersistenceIcon(status) {
  switch (status) {
    case "PERSISTENT": return "🛡️";
    case "NEW_CONTACT": return "⚡";
    case "TRANSIENT": return "⏳";
    case "UNCERTAIN": return "❓";
    default: return "⚡";
  }
}

// =====================================================================
// 7. Detection Inspector & Track History Panel (Section 4, 5, 7, 11)
// =====================================================================

function selectDetection(detId) {
  selectedDetectionId = detId;
  const d = currentDetections.find(x => x.detection_id === detId) || currentDetections[0];
  if (!d) {
    clearDetectionInspector();
    return;
  }

  const track = currentTracks.find(t => t.track_id === d.track_id) || {
    track_id: d.track_id || "TRK-LOCAL",
    observation_count: d.observation_count || 1,
    track_age_s: d.track_age_s || 0.0,
    first_seen: new Date().toISOString(),
    last_seen: new Date().toISOString(),
    confidence_history: [d.confidence],
    persistence_status: d.persistence_status || "NEW_CONTACT",
    audit_log: []
  };

  document.getElementById("inspector-empty").style.display = "none";
  document.getElementById("inspector-details").style.display = "block";
  document.getElementById("inspector-track-badge").textContent = `SELECTED: ${track.track_id}`;

  // Fill Summary Fields
  document.getElementById("insp-track-id").textContent = track.track_id;
  document.getElementById("insp-class").textContent = getCautiousLabel(d.class);
  
  const statusEl = document.getElementById("insp-status-badge");
  const pStatus = track.persistence_status || d.persistence_status || "NEW_CONTACT";
  statusEl.textContent = `${getPersistenceIcon(pStatus)} ${pStatus}`;
  statusEl.className = `status-badge ${getPersistenceBadgeClass(pStatus)}`;
  statusEl.title = "PERSISTENT = same sonar contact observed consistently across multiple scans.";

  document.getElementById("insp-obs-count").textContent = `${track.observation_count || d.observation_count || 1} observations`;
  document.getElementById("insp-track-age").textContent = `${(track.track_age_s || d.track_age_s || 0.0).toFixed(1)}s`;
  document.getElementById("insp-first-seen").textContent = formatTimeOnly(track.first_seen);
  document.getElementById("insp-last-seen").textContent = formatTimeOnly(track.last_seen);
  document.getElementById("insp-current-conf").textContent = `${(d.confidence * 100).toFixed(1)}%`;

  // Render Confidence History Sparkline (Section 5)
  renderConfidenceSparkline(track.confidence_history || [d.confidence]);

  // Render Measurement Provenance Matrix (Section 7)
  renderProvenanceMatrix(d);

  // Render Operator Track Audit Trail (Section 11)
  renderAuditTrail(track);

  // Highlight active row in table and overlay
  highlightActiveElements(detId);
}

function clearDetectionInspector() {
  selectedDetectionId = null;
  document.getElementById("inspector-empty").style.display = "block";
  document.getElementById("inspector-details").style.display = "none";
  document.getElementById("inspector-track-badge").textContent = "NO CONTACT SELECTED";
}

function getPersistenceBadgeClass(status) {
  switch (status) {
    case "PERSISTENT": return "badge-persistent";
    case "NEW_CONTACT": return "badge-new-contact";
    case "TRANSIENT": return "badge-transient";
    case "UNCERTAIN": return "badge-uncertain";
    default: return "badge-new-contact";
  }
}

// Section 5: Unsmoothed Confidence Sparkline
function renderConfidenceSparkline(history) {
  const svg = document.getElementById("confidence-sparkline");
  const valuesEl = document.getElementById("sparkline-values");
  if (!svg || !history || history.length === 0) return;

  const pointsText = history.map(c => (c * 100).toFixed(0) + "%").join(" → ");
  valuesEl.textContent = pointsText;

  const width = svg.clientWidth || 300;
  const height = 40;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  const minVal = 0.3;
  const maxVal = 1.0;
  const n = history.length;
  const stepX = n > 1 ? (width - 40) / (n - 1) : 0;

  const coords = history.map((val, idx) => {
    const x = 20 + idx * stepX;
    const normalized = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal)));
    const y = height - 10 - normalized * (height - 20);
    return { x, y, val };
  });

  // Background baseline grid
  const lineEl = document.createElementNS("http://www.w3.org/2000/svg", "line");
  lineEl.setAttribute("x1", "10");
  lineEl.setAttribute("y1", `${height - 10}`);
  lineEl.setAttribute("x2", `${width - 10}`);
  lineEl.setAttribute("y2", `${height - 10}`);
  lineEl.setAttribute("stroke", "rgba(255,255,255,0.1)");
  lineEl.setAttribute("stroke-dasharray", "3,3");
  svg.appendChild(lineEl);

  // Line path
  if (coords.length > 1) {
    const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    const pts = coords.map(p => `${p.x},${p.y}`).join(" ");
    polyline.setAttribute("points", pts);
    polyline.setAttribute("fill", "none");
    polyline.setAttribute("stroke", "var(--cyan-accent)");
    polyline.setAttribute("stroke-width", "2");
    svg.appendChild(polyline);
  }

  // Dots
  coords.forEach(p => {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", p.x);
    circle.setAttribute("cy", p.y);
    circle.setAttribute("r", "3.5");
    circle.setAttribute("fill", "var(--green-neon)");
    circle.setAttribute("stroke", "#060b18");
    circle.setAttribute("stroke-width", "1.5");
    svg.appendChild(circle);

    // Observation text tag
    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("x", p.x);
    txt.setAttribute("y", p.y - 6);
    txt.setAttribute("font-size", "9");
    txt.setAttribute("font-family", "var(--font-mono)");
    txt.setAttribute("fill", "#cbd5e1");
    txt.setAttribute("text-anchor", "middle");
    txt.textContent = `${(p.val * 100).toFixed(0)}%`;
    svg.appendChild(txt);
  });
}

// Section 7: Measurement Provenance Breakdown
function renderProvenanceMatrix(d) {
  document.getElementById("prov-elevation").textContent = `${d.elevation_m} m`;
  document.getElementById("prov-depth").textContent = d.geo ? `${d.geo.depth_m} m` : "Unavailable";
  
  const hasGps = d.geo && d.geo.lat !== null && d.geo.lon !== null;
  const gpsVal = document.getElementById("prov-gps");
  const gpsBadge = document.getElementById("prov-gps-badge");

  if (hasGps) {
    gpsVal.textContent = `${d.geo.lat.toFixed(4)}° N, ${d.geo.lon.toFixed(4)}° E`;
    gpsBadge.textContent = "DEMO";
    gpsBadge.className = "prov-tag prov-demo";
    gpsBadge.title = "Simulated benchmark coordinates (do not treat as live operational GPS)";
  } else {
    gpsVal.textContent = "GPS Unavailable — image-space tracking only";
    gpsBadge.textContent = "UNAVAILABLE";
    gpsBadge.className = "prov-tag prov-unavailable";
    gpsBadge.title = "No GPS telemetry available for this contact";
  }

  document.getElementById("prov-shadow").textContent = d.elevation_m > 0 ? "Acoustic Shadow Present" : "No Shadow Highlight";
  document.getElementById("prov-altitude").textContent = `${document.getElementById("altitude-slider").value} m`;
}

// Section 11: Audit Trail Display
function renderAuditTrail(track) {
  const list = document.getElementById("audit-log-entries");
  list.innerHTML = "";

  const log = track.audit_log || [];
  if (log.length === 0) {
    const li = document.createElement("li");
    li.className = "font-small font-mono text-muted";
    li.textContent = "No operator overrides recorded.";
    list.appendChild(li);
    return;
  }

  log.forEach(entry => {
    const li = document.createElement("li");
    li.className = "audit-entry";
    li.textContent = `[${formatTimeOnly(entry.timestamp)}] ${entry.action}: ${entry.note} (${entry.operator || 'OPERATOR'})`;
    list.appendChild(li);
  });
}

// Operator Control Actions (Section 11)
async function handleConfirmContact() {
  if (!selectedDetectionId) return;
  const d = currentDetections.find(x => x.detection_id === selectedDetectionId);
  if (!d || !d.track_id) return;

  try {
    const res = await fetch(`/api/tracks/${d.track_id}/confirm`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        note: "Operator confirmed contact for mission workflow (not ground truth)."
      })
    });
    if (!res.ok) throw new Error("Confirm failed");
    const data = await res.json();
    
    // Update local track
    const idx = currentTracks.findIndex(t => t.track_id === d.track_id);
    if (idx >= 0) currentTracks[idx] = data.track;

    addTimelineEvent("OPERATOR", `Contact ${d.track_id} confirmed by operator for mission workflow.`);
    selectDetection(selectedDetectionId);
  } catch (err) {
    console.error("Confirm error:", err);
  }
}

async function handleDismissContact() {
  if (!selectedDetectionId) return;
  const d = currentDetections.find(x => x.detection_id === selectedDetectionId);
  if (!d || !d.track_id) return;

  try {
    const res = await fetch(`/api/tracks/${d.track_id}/dismiss`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        note: "Operator dismissed contact as transient/spurious acoustic return."
      })
    });
    if (!res.ok) throw new Error("Dismiss failed");
    const data = await res.json();

    const idx = currentTracks.findIndex(t => t.track_id === d.track_id);
    if (idx >= 0) currentTracks[idx] = data.track;

    addTimelineEvent("OPERATOR", `Contact ${d.track_id} dismissed as transient.`);
    selectDetection(selectedDetectionId);
  } catch (err) {
    console.error("Dismiss error:", err);
  }
}

async function handleToggleTrackVisibility() {
  if (!selectedDetectionId) return;
  const d = currentDetections.find(x => x.detection_id === selectedDetectionId);
  if (!d || !d.track_id) return;

  try {
    const res = await fetch(`/api/tracks/${d.track_id}/visibility`, {
      method: "POST"
    });
    if (!res.ok) throw new Error("Visibility toggle failed");
    const data = await res.json();

    const idx = currentTracks.findIndex(t => t.track_id === d.track_id);
    if (idx >= 0) currentTracks[idx] = data.track;

    renderInteractiveOverlays();
  } catch (err) {
    console.error("Visibility toggle error:", err);
  }
}

function highlightActiveElements(detId) {
  // Update overlay boxes
  document.querySelectorAll(".overlay-bbox").forEach(el => el.classList.remove("selected"));
  // Update specs table rows
  document.querySelectorAll("#specs-tbody tr").forEach(row => {
    if (row.dataset.id === detId) {
      row.classList.add("selected-row");
    } else {
      row.classList.remove("selected-row");
    }
  });
}

// =====================================================================
// 8. Mission Event Timeline (Section 10)
// =====================================================================

function recordScanTimelineEvents(detections) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString();

  if (!detections || detections.length === 0) {
    addTimelineEvent("SWEEP", "Acoustic sweep complete — zero seabed anomalies detected.", timeStr);
    return;
  }

  detections.forEach(d => {
    const pStatus = d.persistence_status || "NEW_CONTACT";
    const obsCount = d.observation_count || 1;
    const label = getCautiousLabel(d.class);

    if (obsCount === 1) {
      addTimelineEvent("NEW_CONTACT", `NEW_CONTACT — ${label} (${(d.confidence * 100).toFixed(0)}%) [${d.track_id || d.detection_id}]`, timeStr);
    } else if (pStatus === "PERSISTENT" && obsCount === 3) {
      addTimelineEvent("PERSISTENT", `PERSISTENT — ${label} confirmed across 3 observations [${d.track_id}]`, timeStr);
    } else {
      addTimelineEvent("OBSERVATION", `OBSERVATION #${obsCount} — ${label} [${d.track_id}]`, timeStr);
    }
  });

  // Reflex Alert Event
  if (detections[0]?.reflex?.decision_primitive) {
    const prim = detections[0].reflex.decision_primitive;
    if (prim.includes("EMERGENCY")) {
      addTimelineEvent("SYSTEM1", `SYSTEM 1 ALERT → ${prim} (Hazard: ${detections[0].reflex.hazard_score})`, timeStr);
    }
  }
}

function addTimelineEvent(type, message, timeStr = null) {
  const t = timeStr || new Date().toLocaleTimeString();
  timelineEvents.unshift({ type, message, time: t });
  if (timelineEvents.length > 50) timelineEvents.pop();
  renderTimeline();
}

function renderTimeline() {
  const container = document.getElementById("timeline-list");
  const empty = document.getElementById("timeline-empty");
  if (!container || !empty) return;

  if (timelineEvents.length === 0) {
    empty.style.display = "block";
    container.style.display = "none";
    return;
  }

  empty.style.display = "none";
  container.style.display = "flex";
  container.innerHTML = "";

  timelineEvents.forEach(e => {
    const li = document.createElement("li");
    li.className = "timeline-item";

    let icon = "⚡";
    if (e.type === "PERSISTENT") icon = "🛡️";
    if (e.type === "SYSTEM1") icon = "🚨";
    if (e.type === "SYSTEM2") icon = "🧠";
    if (e.type === "OPERATOR") icon = "👤";
    if (e.type === "SWEEP") icon = "🌊";

    li.innerHTML = `
      <span class="timeline-time font-mono">${e.time}</span>
      <span class="timeline-badge-icon" aria-hidden="true">${icon}</span>
      <span class="timeline-desc">${e.message}</span>
    `;
    container.appendChild(li);
  });
}

// =====================================================================
// 9. Map Telemetry & Breadcrumbs (Section 6, 16)
// =====================================================================

function renderMapTelemetry() {
  markerLayer.clearLayers();
  breadcrumbLayer.clearLayers();

  const banner = document.getElementById("gps-unavailable-banner");
  const gpsDisplay = document.getElementById("gps-coords");

  const hasAnyGps = currentDetections.some(d => d.geo && d.geo.lat !== null && d.geo.lon !== null);

  if (!hasAnyGps) {
    if (banner) banner.style.display = "block";
    if (gpsDisplay) gpsDisplay.textContent = "GPS UNAVAILABLE";
    return;
  }

  if (banner) banner.style.display = "none";

  const primary = currentDetections[0];
  if (gpsDisplay && primary.geo) {
    gpsDisplay.textContent = `LAT: ${primary.geo.lat.toFixed(4)}° N | LON: ${primary.geo.lon.toFixed(4)}° E [DEMO]`;
  }

  // Draw detection markers
  const filtered = getFilteredDetections();
  filtered.forEach(d => {
    if (!d.geo || d.geo.lat === null || d.geo.lon === null) return;

    const isCritical = d.class === 'ghost_net' || d.class === 'mine_cylinder';
    const markerColor = isCritical ? 'var(--red-neon)' : 'var(--orange-neon)';

    const pin = L.circleMarker([d.geo.lat, d.geo.lon], {
      radius: 8,
      fillColor: markerColor,
      color: '#ffffff',
      weight: 1.5,
      opacity: 1,
      fillOpacity: 0.85
    });

    const displayClass = getCautiousLabel(d.class);
    const pStatus = d.persistence_status || "NEW_CONTACT";

    pin.bindPopup(`
      <div style="font-family: monospace; font-size: 11px;">
        <b>${(d.detection_id || 'ANOMALY').toUpperCase()}: ${displayClass}</b><br/>
        <b>Persistence: ${pStatus}</b><br/>
        Confidence: ${(d.confidence * 100).toFixed(1)}%<br/>
        Elevation: ${d.elevation_m}m above seabed [DERIVED]<br/>
        Depth: ${d.geo.depth_m}m [DERIVED]<br/>
        Sector: ${d.geo.zone} [DEMO]
      </div>
    `).addTo(markerLayer);
  });

  // Draw track breadcrumbs for persistent or multi-observation tracks (Section 6)
  currentTracks.forEach(track => {
    if (track.positions && track.positions.length > 1) {
      const latlngs = track.positions
        .filter(p => p && p.lat !== null && p.lon !== null)
        .map(p => [p.lat, p.lon]);

      if (latlngs.length > 1) {
        L.polyline(latlngs, {
          color: track.persistence_status === "PERSISTENT" ? "var(--green-neon)" : "var(--cyan-dim)",
          weight: 2.5,
          dashArray: "4, 4",
          opacity: 0.8
        }).addTo(breadcrumbLayer);
      }
    }
  });

  if (primary.geo && primary.geo.lat !== null) {
    map.panTo([primary.geo.lat, primary.geo.lon]);
  }
}

// =====================================================================
// 10. Multi-Criteria UI-Only Filtering (Section 12)
// =====================================================================

function getFilteredDetections() {
  return currentDetections.filter(d => {
    // 1. Persistence filter
    if (currentFilters.persistence !== "all") {
      const pStatus = d.persistence_status || "NEW_CONTACT";
      if (pStatus !== currentFilters.persistence) return false;
    }

    // 2. Severity filter
    if (currentFilters.severity !== "all") {
      const isCritical = d.class === "ghost_net" || d.class === "mine_cylinder";
      const isWarning = d.class === "shipwreck" || d.class === "submarine_pipeline";
      if (currentFilters.severity === "critical" && !isCritical) return false;
      if (currentFilters.severity === "warning" && !isWarning) return false;
      if (currentFilters.severity === "info" && (isCritical || isWarning)) return false;
    }

    // 3. Class filter
    if (currentFilters.targetClass !== "all") {
      if (d.class !== currentFilters.targetClass) return false;
    }

    return true;
  });
}

function applyUIFilters() {
  const filtered = getFilteredDetections();
  renderSpecsTable(filtered);
  renderInteractiveOverlays();
}

function renderSpecsTable(detections) {
  const tbody = document.getElementById("specs-tbody");
  tbody.innerHTML = "";

  if (detections && detections.length > 0) {
    detections.forEach(d => {
      const displayClass = getCautiousLabel(d.class);
      const pStatus = d.persistence_status || "NEW_CONTACT";
      const icon = getPersistenceIcon(pStatus);
      const tr = document.createElement("tr");
      tr.dataset.id = d.detection_id;
      if (d.detection_id === selectedDetectionId) {
        tr.classList.add("selected-row");
      }

      tr.innerHTML = `
        <td style="font-family: var(--font-mono); color: var(--cyan-accent); font-weight:600;">${d.detection_id || 'det_000'}</td>
        <td style="color: ${d.class === 'ghost_net' ? 'var(--red-neon)' : 'var(--cyan-accent)'}; font-weight:600;">${displayClass}</td>
        <td><span class="status-badge ${getPersistenceBadgeClass(pStatus)}">${icon} ${pStatus}</span></td>
        <td>${(d.confidence * 100).toFixed(1)}%</td>
        <td>${d.elevation_m} m <span class="prov-tag prov-derived font-small">DERIVED</span></td>
        <td>${d.geo ? d.geo.depth_m : '--'} m</td>
        <td><span class="font-small text-muted font-mono">${d.track_id || 'TRK-001'}</span></td>
      `;

      tr.addEventListener("click", () => {
        selectDetection(d.detection_id);
      });

      tbody.appendChild(tr);
    });

    document.getElementById("download-pdf-btn").removeAttribute("disabled");
  } else {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state">No contacts match active filters.</td></tr>`;
    document.getElementById("download-pdf-btn").setAttribute("disabled", "true");
  }
}

// =====================================================================
// 11. PDF Dispatch Export (Section 17)
// =====================================================================

async function downloadDispatchPdf() {
  if (!currentDetections || currentDetections.length === 0) return;
  const primary = currentDetections.find(x => x.detection_id === selectedDetectionId) || currentDetections[0];

  const payload = {
    incident_id: "NET-GOM-" + Math.floor(1000 + Math.random() * 9000),
    class: primary.class,
    confidence: (primary.confidence * 100).toFixed(1),
    elevation_m: primary.elevation_m,
    depth_m: (primary.geo && primary.geo.depth_m !== undefined) ? primary.geo.depth_m : null,
    lat: (primary.geo && primary.geo.lat !== undefined) ? primary.geo.lat : null,
    lon: (primary.geo && primary.geo.lon !== undefined) ? primary.geo.lon : null,
    hazard_score: primary.reflex ? primary.reflex.hazard_score : 5.0,
    decision_primitive: primary.reflex ? primary.reflex.decision_primitive : "PASSIVE_LOG",
    latency_ms: primary.reflex ? primary.reflex.latency_ms : 0.05,
    recommended_maneuver: primary.reflex ? primary.reflex.recommended_maneuver : "Proceed on survey.",
    zone: (primary.geo && primary.geo.zone) ? primary.geo.zone : "UNAVAILABLE"
  };

  try {
    const res = await fetch("/api/export-pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error("PDF generation failed");
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `CoastGuard_Dispatch_${payload.incident_id}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    addTimelineEvent("SYSTEM2", `Coast Guard Dispatch PDF generated for ${payload.incident_id}`);
  } catch (err) {
    console.error("PDF download error:", err);
  }
}

// =====================================================================
// 12. System 2 Tactical Reasoning & Forensic Q&A (Section 9)
// =====================================================================

let currentMissionContext = null;

async function triggerSystem2Analysis(data) {
  const summaryEl = document.getElementById("system2-summary");
  const uncEl = document.getElementById("system2-uncertainties");
  const actEl = document.getElementById("system2-action");
  const latEl = document.getElementById("system2-latency-tag");

  if (!summaryEl) return;
  summaryEl.textContent = "Synthesizing tactical briefing...";

  const primaryDet = (data.detections && data.detections.length > 0) ? data.detections[0] : null;

  const temporalCtx = primaryDet ? {
    persistence_status: primaryDet.persistence_status || "NEW_CONTACT",
    observation_count: primaryDet.observation_count || 1,
    track_age_s: primaryDet.track_age_s || 0.0
  } : null;

  currentMissionContext = {
    survey_id: data.system2?.task_id || "SURVEY_LIVE",
    contacts: data.detections || [],
    geo: primaryDet ? primaryDet.geo : null,
    system1: data.system1 || null,
    metadata: { sonar_altitude_m: 12.0, swath_width_m: 50.0 },
    temporal: temporalCtx
  };

  try {
    const res = await fetch("/api/system2/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentMissionContext)
    });
    if (!res.ok) throw new Error("Analysis failed");
    const s2 = await res.json();

    summaryEl.textContent = s2.incident_summary;
    actEl.textContent = `${s2.operator_action} [Priority: ${s2.recovery_priority}]`;
    latEl.textContent = `${s2.latency_ms.toFixed(1)} ms (${s2.status})`;

    uncEl.innerHTML = "";
    if (s2.uncertainties && s2.uncertainties.length > 0) {
      s2.uncertainties.forEach(u => {
        const li = document.createElement("li");
        li.textContent = u;
        uncEl.appendChild(li);
      });
    } else {
      uncEl.innerHTML = "<li>No operational uncertainties recorded.</li>";
    }

    addTimelineEvent("SYSTEM2", `System 2 briefing ready: ${s2.incident_summary.slice(0, 60)}...`);
  } catch (err) {
    console.warn("System 2 analyze error:", err);
    summaryEl.textContent = "System 2 local summary unavailable.";
  }
}

async function handleSystem2QA() {
  const input = document.getElementById("system2-qa-input");
  const ansEl = document.getElementById("system2-qa-answer");
  if (!input || !ansEl) return;
  const q = input.value.trim();
  if (!q) return;

  ansEl.style.display = "block";
  ansEl.textContent = "Consulting System 2 reasoning engine...";

  try {
    const res = await fetch("/api/system2/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        mission_context: currentMissionContext || { contacts: currentDetections }
      })
    });
    if (!res.ok) throw new Error("Q&A request failed");
    const data = await res.json();
    ansEl.textContent = data.answer;
    addTimelineEvent("SYSTEM2", `Operator Q&A: "${q.slice(0, 30)}..." → Answered`);
  } catch (err) {
    ansEl.textContent = "Error querying System 2.";
  }
}

function formatTimeOnly(isoOrDateStr) {
  if (!isoOrDateStr) return "--:--:--";
  try {
    const d = new Date(isoOrDateStr);
    return isNaN(d.getTime()) ? isoOrDateStr : d.toLocaleTimeString();
  } catch {
    return isoOrDateStr;
  }
}
