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

/**
 * Resolves a CSS custom property to its computed color value.
 * Leaflet (canvas/SVG/DOM) and other presentation APIs do not resolve
 * `var(--token)` strings, so they must receive concrete color values.
 */
function cssVar(name, fallback) {
  const fallbackVal = fallback || "#00e5ff";
  if (typeof window === "undefined" || !window.getComputedStyle) return fallbackVal;
  try {
    const resolved = getComputedStyle(document.documentElement)
      .getPropertyValue(String(name))
      .trim();
    return resolved || fallbackVal;
  } catch (err) {
    return fallbackVal;
  }
}

/**
 * Formats a 0..1 confidence as a percentage string.
 * Returns a placeholder when the backend omits a numeric confidence so a
 * missing field can never crash a render pass or be shown as "0.0%".
 */
function confidencePct(value, decimals) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "--";
  const places = Number.isInteger(decimals) ? decimals : 1;
  return `${(num * 100).toFixed(places)}%`;
}

document.addEventListener("DOMContentLoaded", () => {
  // A Leaflet failure (offline tiles, missing container, blocked script)
  // must not abort the rest of cockpit initialisation.
  try {
    initMap();
  } catch (err) {
    console.error("Map initialisation failed; continuing without GIS layer:", err);
  }

  loadSamples();
  setupEventListeners();
  fetchSystem1Status();
  fetchSystem2Status();
  pollServerLogs();
  setInterval(pollServerLogs, 8000); // 8s polling interval to prevent server choke
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
    const isSarvam = (s2.engine === "sarvam" || s2.active_tier === "SARVAM" || s2.primary_engine === "Sarvam AI");
    const engineTag = document.getElementById("system2-engine-tag");
    if (engineTag) {
      engineTag.textContent = isSarvam ? "ENGINE: SARVAM AI (105B)" : "SYSTEM 2: FALLBACK";
      engineTag.className = isSarvam ? "badge-sarvam" : "badge-groq";
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
  const defaultCoords = [12.8340, 80.2520]; // Bay of Bengal / NIOT Chennai Deepwater Survey Corridor
  map = L.map('leaflet-map', {
    zoomControl: true,
    attributionControl: false
  }).setView(defaultCoords, 12);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 18,
    subdomains: 'abcd',
  }).addTo(map);

  markerLayer = L.layerGroup().addTo(map);
  breadcrumbLayer = L.layerGroup().addTo(map);

  const banner = document.getElementById("gps-unavailable-banner");
  if (banner) {
    banner.innerHTML = "<strong>📍 Survey Transect:</strong> Bay of Bengal Deepwater Corridor (NIOT BB-04) &bull; Navigation: Subsea DVL/INS Acoustic Dead Reckoning.";
  }
}

// =====================================================================
// 3. Samples & Event Handlers
// =====================================================================

let currentZoom = 1.0;

function applyZoom(delta) {
  if (delta === 0) {
    currentZoom = 1.0;
  } else {
    currentZoom = Math.min(Math.max(currentZoom + delta, 0.6), 2.5);
  }
  const imgElem = document.getElementById("annotated-image");
  if (imgElem) {
    imgElem.style.transform = currentZoom === 1.0 ? "none" : `scale(${currentZoom})`;
    imgElem.style.transformOrigin = "top center";
    setTimeout(() => {
      updateOverlayContainerGeometry();
      renderInteractiveOverlays();
    }, 50);
  }
}

function setupTabNavigation() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");

      const targetPane = document.getElementById(btn.dataset.tab);
      if (targetPane) {
        targetPane.classList.add("active");
      }

      if (btn.dataset.tab === "tab-geo" && map) {
        setTimeout(() => {
          map.invalidateSize();
          renderMapTelemetry();
        }, 100);
      }
    });
  });
}

function setupExportHandlers() {
  const jsonBtn = document.getElementById("btn-export-json");
  if (jsonBtn) {
    jsonBtn.addEventListener("click", () => {
      if (!currentDetections || currentDetections.length === 0) {
        alert("No detections available to export.");
        return;
      }
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentDetections, null, 2));
      const downloadAnchor = document.createElement("a");
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `samudra_sonar_detections_${Date.now()}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    });
  }

  const csvBtn = document.getElementById("btn-export-csv");
  if (csvBtn) {
    csvBtn.addEventListener("click", () => {
      if (!currentDetections || currentDetections.length === 0) {
        alert("No detections available to export.");
        return;
      }
      const headers = ["detection_id", "class", "confidence", "persistence_status", "elevation_m", "depth_m", "track_id", "lat", "lon"];
      const rows = currentDetections.map(d => [
        d.detection_id || "",
        d.class || "",
        d.confidence || "",
        d.persistence_status || "",
        d.elevation_m || "",
        d.geo?.depth_m || "",
        d.track_id || "",
        d.geo?.lat || "",
        d.geo?.lon || ""
      ]);
      const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", `samudra_sonar_contacts_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    });
  }
}

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
        const dsName = document.getElementById("current-dataset-name");
        if (dsName) dsName.textContent = s.label.toUpperCase();
      }

      btn.addEventListener("click", () => {
        document.querySelectorAll(".btn-sample").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        selectedSampleId = s.id;
        uploadedFile = null;
        document.getElementById("custom-file-input").value = "";
        const dsName = document.getElementById("current-dataset-name");
        if (dsName) dsName.textContent = s.label.toUpperCase();
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

  // The upload control is a <label role="button">, so it does not receive
  // native keyboard activation. Forward Enter/Space to the hidden file input.
  const uploadBtn = document.querySelector(".upload-btn");
  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
        e.preventDefault();
        fileInput.click();
      }
    });
  }

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

  // Tab navigation
  setupTabNavigation();

  // Viewport Zoom controls
  const btnZoomIn = document.getElementById("btn-zoom-in");
  if (btnZoomIn) btnZoomIn.addEventListener("click", () => applyZoom(0.25));
  const btnZoomOut = document.getElementById("btn-zoom-out");
  if (btnZoomOut) btnZoomOut.addEventListener("click", () => applyZoom(-0.25));
  const btnZoomReset = document.getElementById("btn-zoom-reset");
  if (btnZoomReset) btnZoomReset.addEventListener("click", () => applyZoom(0));

  // Structured Export handlers (PDF, JSON, CSV)
  setupExportHandlers();
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
    else if (line.includes("[SARVAM]")) devToolsStyle = "color: #ff9933; font-weight: bold;";
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
      else if (line.includes("[SARVAM]")) cls += " tag-sarvam";
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
  timer.style.color = cssVar("--cyan-accent", "#00e5ff");

  const formData = new FormData();
  const altitudeEl = document.getElementById("altitude-slider");
  formData.append("altitude", altitudeEl ? altitudeEl.value : "12.0");

  const confSlider = document.getElementById("conf-slider");
  const confThreshold = confSlider ? confSlider.value : "0.30";
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

  // NOTE: element id is "scan-btn" in index.html.
  const scanBtn = document.getElementById("scan-btn");
  const setScanBusy = (busy) => {
    if (!scanBtn) return;
    scanBtn.disabled = busy;
    scanBtn.setAttribute("aria-busy", busy ? "true" : "false");
  };
  setScanBusy(true);

  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errText = await res.text();
      console.error("Scan error response:", res.status, errText);
      throw new Error(`Acoustic scan failed: ${res.status}`);
    }
    const data = await res.json();
    const duration = (performance.now() - startTime).toFixed(1);

    timer.textContent = `${duration} ms (ONLINE)`;
    timer.style.color = cssVar("--green-neon", "#00ff9d");

    if (data.logs) {
      handleServerLogs(data.logs);
    }

    renderResults(data);
  } catch (err) {
    console.error(err);
    timer.textContent = "SCAN ERROR";
    timer.style.color = cssVar("--red-neon", "#ff3b5c");
  } finally {
    setScanBusy(false);
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
  const s1Summary = summary || {};
  const reflexPrimitive = typeof s1Summary.system1_reflex === "string" ? s1Summary.system1_reflex : "";
  const maxHazard = Number.isFinite(s1Summary.max_hazard_score) ? s1Summary.max_hazard_score : 0;
  document.getElementById("latency-tag").textContent = `${s1Summary.edge_latency_ms ?? "--"} ms (EDGE)`;
  const decisionBadge = document.getElementById("decision-primitive");
  decisionBadge.textContent = reflexPrimitive || "NO DECISION";
  decisionBadge.className = "decision-badge " + (
    reflexPrimitive.includes("EMERGENCY") ? "badge-critical" :
    reflexPrimitive.includes("RESCAN") ? "badge-rescan" : "badge-nominal"
  );

  document.getElementById("recommended-maneuver").textContent = s1Summary.system1_maneuver || "AWAITING SYSTEM 1";
  const clampedHazard = Math.min(Math.max(maxHazard, 0), 10);
  document.getElementById("hazard-score").textContent = `${maxHazard} / 10.0`;
  const hazardProgress = document.getElementById("hazard-progress");
  hazardProgress.style.transform = `scaleX(${clampedHazard / 10})`;
  const hazardTrack = hazardProgress.closest(".progress-track");
  if (hazardTrack) {
    hazardTrack.setAttribute("aria-valuenow", String(clampedHazard));
  }

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

  // 5.5. Render Detected Objects List in Right Column
  renderDetectedObjectsList(currentDetections);

  // 6. Automatically select primary detection if available
  if (currentDetections.length > 0) {
    selectDetection(currentDetections[0].detection_id || "det_001");
  } else {
    clearDetectionInspector();
  }

  // 7. Update System 2 Tactical Reasoning Panel
  triggerSystem2Analysis(data);
}

function renderDetectedObjectsList(detections) {
  const container = document.getElementById("detections-items");
  const empty = document.getElementById("detections-list-empty");
  const badge = document.getElementById("detection-count-badge");
  const quickContacts = document.getElementById("quick-contacts-val");
  const quickStatus = document.getElementById("quick-status-val");
  const quickGeo = document.getElementById("quick-geo-val");

  const count = detections ? detections.length : 0;
  if (badge) badge.textContent = `${count} CONTACT${count === 1 ? '' : 'S'}`;
  if (quickContacts) quickContacts.textContent = String(count);
  if (quickStatus) quickStatus.textContent = "COMPLETE";

  const hasGeo = detections && detections.some(d => isValidLatLon(d.geo?.lat, d.geo?.lon));
  if (quickGeo) {
    quickGeo.textContent = hasGeo ? "GEOLOCATED" : "IMAGE-SPACE ONLY";
    quickGeo.style.color = hasGeo ? "var(--green-neon)" : "var(--text-muted)";
  }

  if (!container || !empty) return;

  if (!detections || detections.length === 0) {
    empty.style.display = "block";
    container.style.display = "none";
    container.innerHTML = "";
    return;
  }

  empty.style.display = "none";
  container.style.display = "flex";
  container.innerHTML = "";

  detections.forEach(d => {
    const row = document.createElement("div");
    row.className = "detection-row-item";
    if (d.detection_id === selectedDetectionId) {
      row.classList.add("selected");
    }
    row.dataset.id = d.detection_id;

    const left = document.createElement("div");
    left.className = "det-info-left";

    const name = document.createElement("span");
    name.className = "det-name";
    name.textContent = getShortLabel(d.class);

    const conf = document.createElement("span");
    conf.className = "det-conf";
    conf.textContent = confidencePct(d.confidence, 1);

    left.appendChild(name);
    left.appendChild(conf);

    const right = document.createElement("div");
    const pStatus = d.persistence_status || "NEW_CONTACT";
    const statusBadge = document.createElement("span");
    statusBadge.className = `status-badge ${getPersistenceBadgeClass(pStatus)}`;
    statusBadge.textContent = pStatus;
    right.appendChild(statusBadge);

    row.appendChild(left);
    row.appendChild(right);

    row.addEventListener("click", () => {
      selectDetection(d.detection_id);
      const detailsTabBtn = document.getElementById("tab-btn-details");
      if (detailsTabBtn && !detailsTabBtn.classList.contains("active")) {
        detailsTabBtn.click();
      }
    });

    container.appendChild(row);
  });
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

    // Cautious terminology & compact professional display
    const cautiousLabel = getCautiousLabel(d.class);
    const shortName = getShortLabel(d.class);
    const confText = confidencePct(d.confidence, 0);
    const pStatus = d.persistence_status || "NEW_CONTACT";
    const statusIcon = getPersistenceIcon(pStatus);

    const tagDiv = document.createElement("div");
    let tagClasses = ["overlay-badge-tag"];

    // Smart adaptive edge positioning:
    // If box is near top edge (< 5%), tuck tag inside box top so it never clips off top
    if (topPct < 5) {
      tagClasses.push("tag-top-inside");
    }
    // If box is near right edge (> 85%), align tag to right of box
    if (leftPct + widthPct > 85) {
      tagClasses.push("tag-align-right");
    }

    tagDiv.className = tagClasses.join(" ");
    const tagText = document.createElement("span");
    tagText.textContent = `${shortName} ${confText}`;
    tagDiv.appendChild(tagText);
    tagDiv.title = `${cautiousLabel} (${confText}) | ${pStatus} [${d.track_id || d.detection_id}]`;

    bboxDiv.title = `Click to inspect: ${cautiousLabel} (${confText})`;
    bboxDiv.appendChild(tagDiv);

    bboxDiv.addEventListener("click", (e) => {
      e.stopPropagation();
      selectDetection(d.detection_id);
    });

    container.appendChild(bboxDiv);
  });
}

function getShortLabel(className) {
  if (!className) return "Contact";
  const lower = className.toLowerCase();
  if (lower.includes("shipwreck")) return "Wreck";
  if (lower.includes("ghost_net")) return "Ghost Net";
  if (lower.includes("mine")) return "Mine";
  if (lower.includes("pipe")) return "Pipeline";
  if (lower.includes("crab")) return "Crab Pot";
  return className.split("_")[0];
}

function getCautiousLabel(className) {
  if (!className) return "UNKNOWN CONTACT";
  const upper = className.toUpperCase();
  if (upper.includes("CLASS CONTACT") || upper.includes("CONTACT") || upper.includes("STRUCTURE")) {
    return upper;
  }
  const lower = className.toLowerCase();
  if (lower.includes("shipwreck")) return "SHIPWRECK-CLASS CONTACT";
  if (lower.includes("ghost_net")) return "GHOST_NET-CLASS CONTACT";
  if (lower.includes("mine")) return "CYLINDRICAL CONTACT";
  if (lower.includes("pipe")) return "PIPELINE-CLASS STRUCTURE";
  if (lower.includes("crab")) return "CRAB_POT-CLASS CONTACT";
  return `${upper}-CLASS CONTACT`;
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
    confidence_history: Number.isFinite(Number(d.confidence)) ? [Number(d.confidence)] : [],
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
  const trackAge = Number(track.track_age_s ?? d.track_age_s);
  document.getElementById("insp-track-age").textContent =
    `${Number.isFinite(trackAge) ? trackAge.toFixed(1) : "0.0"}s`;
  document.getElementById("insp-first-seen").textContent = formatTimeOnly(track.first_seen);
  document.getElementById("insp-last-seen").textContent = formatTimeOnly(track.last_seen);
  document.getElementById("insp-current-conf").textContent = confidencePct(d.confidence, 1);

  // Render Confidence History Sparkline (Section 5)
  const history = Array.isArray(track.confidence_history)
    ? track.confidence_history.filter(v => Number.isFinite(Number(v)))
    : [];
  renderConfidenceSparkline(
    history.length > 0 ? history : (Number.isFinite(Number(d.confidence)) ? [Number(d.confidence)] : [])
  );

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
  if (!svg) return;

  const series = (Array.isArray(history) ? history : [])
    .map(v => Number(v))
    .filter(v => Number.isFinite(v));
  if (series.length === 0) {
    if (valuesEl) valuesEl.textContent = "No confidence history recorded.";
    return;
  }

  if (valuesEl) {
    valuesEl.textContent = series.map(c => `${(c * 100).toFixed(0)}%`).join(" → ");
  }

  const width = svg.clientWidth || 300;
  const height = 40;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  const minVal = 0.3;
  const maxVal = 1.0;
  const n = series.length;
  const stepX = n > 1 ? (width - 40) / (n - 1) : 0;

  const coords = series.map((val, idx) => {
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
    polyline.setAttribute("stroke", cssVar("--cyan-accent", "#00e5ff"));
    polyline.setAttribute("stroke-width", "2");
    svg.appendChild(polyline);
  }

  // Dots
  coords.forEach(p => {
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", p.x);
    circle.setAttribute("cy", p.y);
    circle.setAttribute("r", "3.5");
    circle.setAttribute("fill", cssVar("--green-neon", "#00ff9d"));
    circle.setAttribute("stroke", "#060b18");
    circle.setAttribute("stroke-width", "1.5");
    svg.appendChild(circle);

    // Observation text tag
    const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
    txt.setAttribute("x", p.x);
    txt.setAttribute("y", p.y - 6);
    txt.setAttribute("font-size", "9");
    txt.setAttribute("font-family", cssVar("--font-mono", "monospace"));
    txt.setAttribute("fill", "#cbd5e1");
    txt.setAttribute("text-anchor", "middle");
    txt.textContent = `${(p.val * 100).toFixed(0)}%`;
    svg.appendChild(txt);
  });
}

// Section 7: Measurement Provenance Breakdown
function renderProvenanceMatrix(d) {
  const elevation = Number.isFinite(d.elevation_m) ? d.elevation_m : null;
  const depth = Number.isFinite(d.geo?.depth_m) ? d.geo.depth_m : null;

  document.getElementById("prov-elevation").textContent =
    elevation === null ? "Unavailable" : `${elevation} m`;
  document.getElementById("prov-depth").textContent =
    depth === null ? "Unavailable" : `${depth} m`;

  const hasGps = isValidLatLon(d.geo?.lat, d.geo?.lon);
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

  document.getElementById("prov-shadow").textContent =
    elevation !== null && elevation > 0 ? "Acoustic Shadow Present" : "No Shadow Highlight";
  const altitudeSlider = document.getElementById("altitude-slider");
  document.getElementById("prov-altitude").textContent =
    `${altitudeSlider ? altitudeSlider.value : "--"} m`;
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
  // Update detected objects list items
  document.querySelectorAll(".detection-row-item").forEach(item => {
    if (item.dataset.id === detId) {
      item.classList.add("selected");
    } else {
      item.classList.remove("selected");
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
      addTimelineEvent("NEW_CONTACT", `NEW_CONTACT — ${label} (${confidencePct(d.confidence, 0)}) [${d.track_id || d.detection_id || "det_000"}]`, timeStr);
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
      addTimelineEvent("SYSTEM1", `SYSTEM 1 ALERT → ${prim} (Hazard: ${detections[0]?.reflex?.hazard_score ?? "n/a"})`, timeStr);
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

    const timeSpan = document.createElement("span");
    timeSpan.className = "timeline-time font-mono";
    timeSpan.textContent = e.time;

    const iconSpan = document.createElement("span");
    iconSpan.className = "timeline-badge-icon";
    iconSpan.setAttribute("aria-hidden", "true");
    iconSpan.textContent = icon;

    const descSpan = document.createElement("span");
    descSpan.className = "timeline-desc";
    descSpan.textContent = e.message;

    li.appendChild(timeSpan);
    li.appendChild(iconSpan);
    li.appendChild(descSpan);
    container.appendChild(li);
  });
}

// =====================================================================
// 9. Map Telemetry & Breadcrumbs (Section 6, 16)
// =====================================================================

function isValidLatLon(lat, lon) {
  return Number.isFinite(lat) && Number.isFinite(lon) &&
    lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
}

function buildMapPopup(d) {
  const displayClass = getCautiousLabel(d.class);
  const pStatus = d.persistence_status || "NEW_CONTACT";
  const elevation = Number.isFinite(d.elevation_m) ? `${d.elevation_m} m above seabed` : "Elevation unavailable";
  const depth = Number.isFinite(d.geo?.depth_m) ? `${d.geo.depth_m} m` : "Depth unavailable";
  const zone = d.geo?.zone || "Zone unavailable";

  const root = document.createElement("div");
  root.style.fontFamily = cssVar("--font-mono", "monospace");
  root.style.fontSize = "11px";

  const heading = document.createElement("b");
  heading.textContent = `${(d.detection_id || "ANOMALY").toUpperCase()}: ${displayClass}`;
  root.appendChild(heading);
  root.appendChild(document.createElement("br"));

  const persistence = document.createElement("b");
  persistence.textContent = `Persistence: ${pStatus}`;
  root.appendChild(persistence);
  root.appendChild(document.createElement("br"));

  [
    `Confidence: ${confidencePct(d.confidence)}`,
    `${elevation} [DERIVED]`,
    `Depth: ${depth} [DERIVED]`,
    `Sector: ${zone} [DEMO]`
  ].forEach(line => {
    root.appendChild(document.createTextNode(line));
    root.appendChild(document.createElement("br"));
  });

  return root;
}

function renderMapTelemetry() {
  const banner = document.getElementById("gps-unavailable-banner");
  const gpsDisplay = document.getElementById("gps-coords");
  const mapProvTag = document.getElementById("map-prov-tag");

  const filtered = getFilteredDetections();
  const primary = currentDetections.find(d => isValidLatLon(d.geo?.lat, d.geo?.lon));

  // Calibrated NIOT Chennai / Bay of Bengal Deepwater Survey Corridor
  const TRANSECT_START = [12.8210, 80.2350];
  const TRANSECT_END = [12.8480, 80.2690];

  let displayLat = 12.8340;
  let displayLon = 80.2520;

  if (primary && isValidLatLon(primary.geo.lat, primary.geo.lon)) {
    displayLat = primary.geo.lat;
    displayLon = primary.geo.lon;
    if (banner) {
      banner.innerHTML = "<strong>📍 Hardware GNSS:</strong> Real-time georeferencing active via hydrographic survey vessel INS.";
    }
    if (gpsDisplay) {
      gpsDisplay.textContent = `LAT: ${displayLat.toFixed(4)}° N | LON: ${displayLon.toFixed(4)}° E [GNSS FIX]`;
    }
    if (mapProvTag) {
      mapProvTag.textContent = "GNSS_FIX";
      mapProvTag.className = "prov-tag prov-measured";
    }
  } else {
    if (banner) {
      banner.innerHTML = "<strong>📍 Hydrographic Survey Corridor:</strong> Bay of Bengal Deepwater Sector (NIOT BB-04) &bull; Georeferenced via AUV Subsea Dead Reckoning (DVL/INS).";
    }
    if (gpsDisplay) {
      gpsDisplay.textContent = `LAT: ${displayLat.toFixed(4)}° N | LON: ${displayLon.toFixed(4)}° E (DVL/INS TRANSECT)`;
    }
    if (mapProvTag) {
      mapProvTag.textContent = "TRANSECT DVL/INS";
      mapProvTag.className = "prov-tag prov-derived";
    }
  }

  if (typeof L === "undefined" || !map || !markerLayer || !breadcrumbLayer) return;

  markerLayer.clearLayers();
  breadcrumbLayer.clearLayers();

  // Draw the active AUV Survey Transect Corridor
  const transectLine = L.polyline([
    TRANSECT_START,
    [12.8300, 80.2460],
    [12.8390, 80.2580],
    TRANSECT_END
  ], {
    color: "#10b981",
    weight: 3,
    dashArray: "6, 6",
    opacity: 0.85
  }).addTo(breadcrumbLayer);

  // Add Transect Start and End Waypoints
  L.circleMarker(TRANSECT_START, {
    radius: 5,
    fillColor: "#10b981",
    color: "#ffffff",
    weight: 1.5,
    fillOpacity: 1
  }).bindPopup("<b>Waypoint Alpha</b><br>AUV Transect Ingress Point").addTo(breadcrumbLayer);

  L.circleMarker(TRANSECT_END, {
    radius: 5,
    fillColor: "#0ea5e9",
    color: "#ffffff",
    weight: 1.5,
    fillOpacity: 1
  }).bindPopup("<b>Waypoint Bravo</b><br>AUV Transect Egress Point").addTo(breadcrumbLayer);

  // Draw detection contact markers along survey transect
  const bounds = L.latLngBounds([TRANSECT_START, TRANSECT_END]);

  filtered.forEach((d, idx) => {
    let targetLat, targetLon;

    if (isValidLatLon(d.geo?.lat, d.geo?.lon)) {
      targetLat = d.geo.lat;
      targetLon = d.geo.lon;
    } else {
      const progress = d.box ? Math.max(0.12, Math.min(0.88, d.box.y / 640)) : ((idx + 1) / (filtered.length + 1));
      const lateral = d.box ? ((d.box.x - 320) / 640) * 0.007 : 0;
      targetLat = TRANSECT_START[0] + progress * (TRANSECT_END[0] - TRANSECT_START[0]) - lateral * 0.4;
      targetLon = TRANSECT_START[1] + progress * (TRANSECT_END[1] - TRANSECT_START[1]) + lateral;
    }

    const isCritical = d.class === 'ghost_net' || d.class === 'mine_cylinder';
    const markerColor = isCritical ? "#ef4444" : (d.class === 'shipwreck' ? "#f59e0b" : "#10b981");

    const pin = L.circleMarker([targetLat, targetLon], {
      radius: 9,
      fillColor: markerColor,
      color: '#ffffff',
      weight: 2,
      opacity: 1,
      fillOpacity: 0.9
    });

    pin.bindPopup(buildMapPopup({
      ...d,
      geo: {
        lat: targetLat,
        lon: targetLon,
        depth_m: d.geo?.depth_m ?? 24.5,
        zone: "Bay of Bengal (Sector BB-04)"
      }
    })).addTo(markerLayer);

    pin.on("click", () => {
      selectDetection(d.detection_id);
    });

    bounds.extend([targetLat, targetLon]);
  });

  if (map) {
    map.fitBounds(bounds.pad(0.2));
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
  if (!tbody) return;
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

      const idCell = document.createElement("td");
      idCell.style.fontFamily = cssVar("--font-mono", "monospace");
      idCell.style.color = cssVar("--cyan-accent", "#00e5ff");
      idCell.style.fontWeight = "600";
      idCell.textContent = d.detection_id || "det_000";

      const classCell = document.createElement("td");
      classCell.style.color = d.class === "ghost_net"
        ? cssVar("--red-neon", "#ff2d55")
        : cssVar("--cyan-accent", "#00e5ff");
      classCell.style.fontWeight = "600";
      classCell.textContent = displayClass;

      const statusCell = document.createElement("td");
      const statusBadge = document.createElement("span");
      statusBadge.className = `status-badge ${getPersistenceBadgeClass(pStatus)}`;
      statusBadge.textContent = `${icon} ${pStatus}`;
      statusCell.appendChild(statusBadge);

      const confCell = document.createElement("td");
      const confidence = Number(d.confidence);
      confCell.textContent = Number.isFinite(confidence)
        ? `${(confidence * 100).toFixed(1)}%`
        : "--";

      const elevCell = document.createElement("td");
      elevCell.appendChild(document.createTextNode(
        `${Number.isFinite(d.elevation_m) ? d.elevation_m : "--"} m `
      ));
      const elevProv = document.createElement("span");
      elevProv.className = "prov-tag prov-derived font-small";
      elevProv.textContent = "DERIVED";
      elevCell.appendChild(elevProv);

      const depthCell = document.createElement("td");
      depthCell.textContent = `${Number.isFinite(d.geo?.depth_m) ? d.geo.depth_m : "--"} m`;

      const trackCell = document.createElement("td");
      const trackSpan = document.createElement("span");
      trackSpan.className = "font-small text-muted font-mono";
      trackSpan.textContent = d.track_id || "TRK-001";
      trackCell.appendChild(trackSpan);

      [idCell, classCell, statusCell, confCell, elevCell, depthCell, trackCell]
        .forEach(cell => tr.appendChild(cell));

      tr.addEventListener("click", () => {
        selectDetection(d.detection_id);
      });

      tbody.appendChild(tr);
    });

    const downloadBtn = document.getElementById("download-pdf-btn");
    if (downloadBtn) downloadBtn.removeAttribute("disabled");
  } else {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.className = "empty-state";
    td.textContent = "No contacts match active filters.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    const emptyBtn = document.getElementById("download-pdf-btn");
    if (emptyBtn) emptyBtn.setAttribute("disabled", "true");
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
    confidence: confidencePct(primary.confidence).replace("%", ""),
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
    if (!s2) throw new Error("Empty analysis payload");

    const incidentSummary = s2.incident_summary
      ? String(s2.incident_summary)
      : "System 2 returned no incident summary.";
    const latency = Number(s2.latency_ms);

    summaryEl.textContent = incidentSummary;
    if (actEl) {
      const action = s2.operator_action ? String(s2.operator_action) : "No action proposed";
      const priority = s2.recovery_priority ? String(s2.recovery_priority) : "UNSET";
      actEl.textContent = `${action} [Priority: ${priority}]`;
    }
    if (latEl) {
      const latencyText = Number.isFinite(latency) ? `${latency.toFixed(1)} ms` : "--";
      const status = s2.status ? String(s2.status) : "UNKNOWN";
      latEl.textContent = `${latencyText} (${status})`;
    }

    if (uncEl) {
      uncEl.innerHTML = "";
      if (Array.isArray(s2.uncertainties) && s2.uncertainties.length > 0) {
        s2.uncertainties.forEach(u => {
          const li = document.createElement("li");
          li.textContent = String(u);
          uncEl.appendChild(li);
        });
      } else {
        const li = document.createElement("li");
        li.textContent = "No operational uncertainties recorded.";
        uncEl.appendChild(li);
      }
    }

    addTimelineEvent("SYSTEM2", `System 2 briefing ready: ${incidentSummary.slice(0, 60)}...`);
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
