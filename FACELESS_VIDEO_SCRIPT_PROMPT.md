# 🎬 FACELESS SIH PROJECT VIDEO — MASTER PROMPT

> **How to use:** Apna PPT ChatGPT me attach karo, phir neeche ka poora block ek message me paste kar do.
> Default duration 4 min hai. Chaaho to top line change kar do (3 / 4 / 5 min).

---
---

# ==== BEGIN PASTE HERE ====

You are an expert **faceless YouTube scriptwriter, hackathon mentor, motion-graphics director, and technical video producer**.

Your task: create a professional, engaging **FACELESS** YouTube video script for a Smart India Hackathon (SIH) project, using **ONLY** the information available in the attached PPT and the GROUND TRUTH block below.

**CRITICAL — THE PRESENTER NEVER APPEARS ON CAMERA.**
There is no face, no talking head, no presenter on stage, no webcam footage. The entire video is **AI voiceover narration** layered over **B-roll, screen recordings, motion graphics, and data visualizations**. Design every single line for that format.

==================================================
PART 0 — VIDEO DURATION
==================================================
Default duration: **4 minutes**
The user may specify 3, 4, or 5 minutes.

Target speaking length (voiceover only):
- 3 minutes → 390–450 words
- 4 minutes → 520–600 words
- 5 minutes → 650–750 words

Prioritize natural spoken flow over exact word count. On-screen visuals are NOT counted in the word budget.

==================================================
PART 1 — FACELESS PRODUCTION RULES (NON-NEGOTIABLE)
==================================================
1. **No human appears anywhere in frame.** No face, no silhouette, no hands typing, no over-the-shoulder shots, no interview clips.
2. **No "as you can see, on my screen"** type phrases — the viewer sees the screen, not the presenter.
3. Narration must be **tense-agnostic and person-agnostic**: "the system", "the model", "the pipeline", "the operator". Never "I", never "we built" unless the PPT explicitly uses first person. Prefer "our system" is allowed ONLY if the PPT states it.
4. Every narration block must be **paired with a visual direction**. A line with no visual is a script failure.
5. Write narration for **AI text-to-speech** (ElevenLabs / Murf / Google Cloud TTS style):
   - Short sentences (8–16 words).
   - Use commas and full stops for breathing pauses. Never run-on sentences.
   - Avoid ALL CAPS, avoid asterisks, avoid parentheses inside narration, avoid symbols like `→` inside spoken lines.
   - Numbers must be speakable: "42 point 7 megabytes", "ninety two point nine percent".
   - Avoid abbreviations that TTS mangles — write "Y O L O" as "YOLO", but write "API" as "A P I", "UI" as "U I", "GPU" as "G P U".
6. **Pacing:** assume ~0.5s padding per sentence break. Do not write sentences longer than 3 clauses.

==================================================
PART 2 — GROUND TRUTH (VERIFIED FACTS ABOUT THIS PROJECT)
==================================================
Treat this as authoritative. The PPT is the primary source, but where the PPT is silent, you MAY use these verified values. **Do not contradict them.** Do not invent anything beyond them.

**PROJECT IDENTITY**
- Project name: **Ocean IQ**
- Problem Statement: **SIH 2026 — PS 26057**
- Title: AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery
- Organization: Ministry of Earth Sciences (MoES) & National Institute of Ocean Technology (NIOT), Chennai
- Theme: Disaster Management & Marine Conservation
- Category: Software — Deep-Tech / Computer Vision / Edge Robotics / GIS

**PROBLEM (grounded)**
- Abandoned, lost or discarded fishing gear — ALDFG, also called "ghost nets" — accumulate on the seabed.
- Critical Indian hotspots: Gulf of Mannar, Palk Bay corridor, Andaman & Nicobar, Lakshadweep reefs.
- Ghost gear traps and kills endangered marine life: Dugongs, Olive Ridley sea turtles, dolphins; it damages coral reefs and fouls vessel propellers.
- Optical cameras fail underwater: beyond roughly 20 metres depth, or in turbid water, visibility collapses to 2–5 metres. So vessels and AUVs use Side-Scan Sonar.
- The human bottleneck: an AUV or towed sonar surveys hundreds of kilometres of seafloor and produces massive volumes of acoustic imagery. Manual inspection by hydrographers takes weeks to months, is expensive, and suffers from human fatigue.

**WHY SIDE-SCAN SONAR (grounded)**
- SSS fires high-frequency acoustic pulses sideways from a tow-fish or AUV.
- Acoustic backscatter: dense objects (nets, metal hulls, pipelines) return strong echoes and appear bright.
- Acoustic shadow: no sound returns behind an object, so a black zero-return region forms behind it.
- KEY INSIGHT: shadow length plus sonar altitude yields the physical height of the object above the seabed.
- Speckle noise: multi-path interference creates heavy grain, requiring dedicated despeckling.

**SOLUTION — DETECTION STACK (grounded, measured)**
- Detector: YOLOv8s fine-tuned on side-scan sonar, exported to **ONNX** — a 42.7 megabyte file, versus roughly 1.5 gigabytes for the original PyTorch weights. This is what makes edge deployment possible.
- Five classes: `crab_pot`, `submarine_pipeline`, `shipwreck`, `ghost_net`, `mine_cylinder`.
- Preprocessing: CLAHE contrast-limited adaptive histogram equalization on the luminance channel, 8 by 8 grid, clip limit 2.0; plus a bilateral filter, diameter 5, sigma 50, to kill speckle while preserving sharp shadow edges.
- Tiling: sliding window, 640 pixel tiles, 20 percent overlap, stride 512. This exists because real sonar strips are 20,000 pixel long waterfall images — naively resizing to 640 by 640 destroys small nets.
- Deduplication: class-aware NMS, both inside each tile and globally across tile seams, so a net tangled on a pipeline keeps both labels.

**MEASURED PERFORMANCE (real numbers — use these)**
- Ghost net image: 1 detection, class `ghost_net`, confidence 92.9 percent, total 193.5 milliseconds.
- Shipwreck image: 1 detection, class `shipwreck`, confidence 84.5 percent, 148.7 milliseconds.
- Submarine pipeline image: 1 detection, class `submarine_pipeline`, confidence 64.5 percent, 150.4 milliseconds.
- Clean seabed negative control: **0 detections — zero false positives**, 280.4 milliseconds.
- Everything runs on CPU via ONNX Runtime. No GPU required.

**PHYSICS ENGINE (grounded)**
- Object height equals sonar altitude multiplied by shadow length, divided by the sum of slant range and shadow length.
- This converts a 2D image into a real-world clearance height in metres.

**SYSTEM 1 — EDGE REFLEX ENGINE (grounded)**
- Inspired by the dual-process cognitive model: a fast reflex loop and a slow reasoning loop.
- System 1 runs entirely on the vehicle, offline, with no internet, at sub-100 millisecond latency.
- Backed by a reinforcement-learning reflex agent called Laya, with a deterministic rule-based fallback if the weights are absent.
- It outputs one of three decision primitives:
  - `EMERGENCY_PROP_HAZARD` — evade
  - `LOITER_AND_RESCAN` — hold station and re-survey
  - `PASSIVE_LOG` — log and continue
- Plus a calibrated hazard severity score from 1 to 10.
- **Safety guardrail, state this explicitly:** the system is advisory only. It does not control actuators. It never drives the vehicle.

**SYSTEM 2 — TACTICAL COMMANDER (grounded)**
- Runs at the surface base station or command ship.
- Uses a cloud LLM — Groq-hosted LLaMA 3.3, or Sarvam — to produce a hydrographic briefing, inter-agency communication, and natural-language Q&A for naval officers.
- If no API key is present, a deterministic tactical synthesis engine takes over seamlessly.
- Auto-compiles the "Operation Net-Zero" recovery sheet for the Indian Coast Guard.

**TEMPORAL PERSISTENCE (grounded)**
- AUVs fly multiple passes over the same coordinates.
- If a contact appears at the same location across passes, it is upgraded from a new contact to persistent and confirmed debris.
- This filters out transient moving targets — fish schools, marine mammals — that a single-frame detector would false-alarm on.

**SOFTWARE STACK (grounded)**
- Backend: Python, FastAPI, Uvicorn.
- Inference: ONNX Runtime, OpenCV, NumPy.
- Frontend: dependency-free HTML, CSS and vanilla JavaScript. No build step, no CDN, no framework.
- Mapping: Leaflet for the geospatial chart.
- Reporting: ReportLab for PDF generation.
- PDF generation endpoint plus GZip compression and CORS middleware.
- 45 pytest test modules, all passing.
- Deploys on a 512 megabyte free cloud tier. No database required.

**HARDWARE (grounded)**
- The deployed artefact is the software stack. The physical sensor layer is a **side-scan sonar** carried on an AUV or towed behind a survey vessel.
- **If the PPT does not show a specific AUV model, microcontroller, or sensor part number, do NOT name one.** Describe the hardware generically as a side-scan sonar payload. This is the single most important anti-fabrication rule in this project.

**DATASETS (grounded)**
- DRISHTI SSS — Creative Commons BY-SA 4.0.
- SubPipe, REMARO Network / OceanScan-MST — Creative Commons BY 4.0.
- AI4Shipwrecks, University of Michigan with NOAA Thunder Bay — Creative Commons BY 4.0.
- NOAA Ocean Exploration archive — US Public Domain.

**OPERATOR UI — WHAT THE DASHBOARD ACTUALLY SHOWS (grounded)**
This is a dark tactical cockpit, and the demo narration must describe only these real elements:
- A persistent DEMO MODE indicator.
- A sonar waterfall viewer with canvas bounding-box overlays, zoom controls, and a confidence value on each box.
- A Detection Inspector panel: track ID, class label, persistence badge, observation count, track age, and a confidence sparkline showing history.
- A measurement provenance matrix labelling every value as MEASURED, DERIVED, ASSUMED, SIMULATED, DEMO, or UNAVAILABLE.
- A System 1 reflex HUD: active engine, decision primitive, hazard score, and an advisory-only disclaimer.
- A System 2 tactical briefing HUD.
- A mission event timeline with real timestamps.
- A Leaflet map with a breadcrumb trajectory trail and a standoff range ring.
- Operator action buttons: Confirm Contact, Dismiss Transient, Toggle Visibility.
- A PDF export button producing the Coast Guard dispatch sheet.
- A live structured log console.

**CLOSE-CORRECTNESS GUARDRAIL (state this in the video)**
The UI deliberately labels anything that is not physically measured — elevation as DERIVED, missing GPS as UNAVAILABLE, demo coordinates as DEMO. The system refuses to present an unmeasured value as a measured one. This is a design decision worth showing judges.

==================================================
PART 3 — UNDERSTAND THE PPT
==================================================
Read the whole attached PPT. Extract: problem statement, users, pain points, existing process, solution, features, innovation, AI/ML, software stack, hardware, APIs, database, architecture, data flow, workflow, impact, prototype/demo, metrics, future scope.

Rules:
- The PPT plus the GROUND TRUTH block above is the **only** source of truth.
- Do not require the user to explain the project separately.
- If something is missing, do NOT fabricate. Either omit it, or phrase it safely without the technical claim.
- If the PPT and ground truth appear to conflict, prefer the PPT for wording but never invent a number that contradicts the ground truth.

==================================================
PART 4 — THE STORY ARC
==================================================
Do not follow slide order. Build a story:

PROBLEM → WHY IT MATTERS → SOLUTION → HOW IT WORKS →
ARCHITECTURE → LIVE DEMO → IMPACT → CLOSING

The viewer must fully understand the project from the video alone, without ever seeing the PPT.

==================================================
PART 5 — SCRIPT STRUCTURE
==================================================

### SECTION 1 — HOOK + PROBLEM
**Duration: 15–25 seconds**
- Open on the problem or its real-world consequence. No "Hello everyone, today we will present."
- Strong but factual. Consider opening on the Dugong, the ghost net, or the fact that light simply does not reach.
- Name who faces it and what happens if it is not solved.
- No exaggerated claims.

**Required visuals:** deep-sea murk, sonar waterfall texture, ghost net, marine life, map of the Gulf of Mannar.

---

### SECTION 2 — THE SOLUTION
**Duration: 35–50 seconds**
- What the system does, who uses it, how it solves the problem, the 3 to 4 most important features.
- Explain in plain language FIRST, then introduce technical terms.
- Tie every feature to the problem it solves. Do NOT dump a feature list.

**Required visuals:** title card, annotated sonar strip, the model file, the class list, before/after preprocessing split.

---

### SECTION 3 — SOFTWARE + HARDWARE ARCHITECTURE
**Duration: 60–90 seconds — this is the technical core**

Tell the architecture as a **journey of data**:

INPUT → SENSOR → PREPROCESSING → TILING → AI MODEL →
PHYSICS → GEO → DECISION → FRONTEND → USER ACTION

**Software** — frontend, backend, AI/ML, APIs, processing logic, deployment.
**Hardware** — the side-scan sonar payload, and its AUV or tow-fish carrier. If no specific part is in the PPT, stay generic.

Always explain **WHY** a technology is used, never just name it:
- "ONNX is used because..." → 42.7 megabytes instead of 1.5 gigabytes.
- "Tiling exists because..." → 20,000 pixel strips, resizing destroys small nets.
- "The bilateral filter is used because..." → speckle noise, but edges must survive.
- "Two loops instead of one because..." → a slow LLM cannot steer a vehicle in real time.

**Required visuals:** animated data-flow diagram, pipeline stages appearing one by one, ONNX size comparison bar, tiling grid overlay, System 1 vs System 2 split-screen.

---

### SECTION 4 — LIVE DEMO
**Duration: 60–90 seconds**

Transition naturally: "Now let's see this in action." or "Instead of just describing it, let's test the real system."

Structure the demo narration as:
1. What screen is shown
2. What action occurs
3. What the system receives
4. What processing happens
5. What result appears
6. Why that result matters

**Base the demo ONLY on these real, verified flows** — pick the strongest 2 or 3:
- Opening the dashboard and showing the dark tactical cockpit with the DEMO MODE indicator.
- Loading a real sonar sample — ghost net, shipwreck, pipeline, or mine.
- Bounding boxes appearing on the waterfall viewer with confidence scores.
- The Detection Inspector opening with track ID, class and confidence sparkline.
- The System 1 HUD updating with the decision primitive and hazard score.
- The System 2 tactical briefing text appearing.
- The provenance matrix labelling values MEASURED, DERIVED, UNAVAILABLE.
- The Leaflet map pinning the contact with a breadcrumb trail.
- The mission timeline logging the event.
- Clicking Export and the Coast Guard recovery PDF downloading.

**HARD ANTI-FABRICATION RULE FOR THE DEMO:**
Do NOT invent buttons, dropdowns, screens, sensor values, or API responses not in the list above. If the PPT does not pin down an exact click sequence, mark that beat:
`[DEMO DETAIL TO VERIFY]`

Also state the **negative control** — the clean seabed image returns zero detections. A demo that only shows positives is unconvincing; showing zero false positives on clean seabed is the strongest single proof point in this project.

**Required visuals:** full screen recording, cursor highlights, zoom into the inspector, zoom into the hazard HUD, a side-by-side positive vs clean-seabed comparison.

---

### SECTION 5 — IMPACT + CLOSING
**Duration: 20–30 seconds**
- Who benefits: NIOT and MoES survey teams, the Indian Coast Guard, marine conservation agencies.
- The practical improvement: weeks of manual review reduced to automated triage.
- Scalability and future scope ONLY if the PPT supports it.
- End with a memorable, professional, non-hyped closing plus a YouTube call to action.

**BANNED phrases unless explicitly in the PPT:** "completely eliminate", "revolutionize", "world's first", "save the planet", "game-changing".

**Required visuals:** Gulf of Mannar map, Coast Guard vessel, recovery diver with cutting shears, closing title card.

==================================================
PART 6 — NARRATION STYLE
==================================================
**Use:** short sentences, conversational language, strong transitions, active voice, natural pauses, occasional emphasis, plain-English explanations before jargon.

**Avoid:** academic register, heavy jargon, reading slide titles, long paragraphs, repetition, generic motivational filler, dramatic storytelling, unexplained technical detail.

The viewer must end up thinking:
- "I understand the problem."
- "I understand the solution."
- "I understand how it works."
- "I can see the prototype proving it."

==================================================
PART 7 — OUTPUT FORMAT
==================================================
Return EXACTLY this structure:

# YouTube Video Script 🎬 (Faceless)

## Project
[Name + one-line PS reference]

## Duration
[Selected duration]

## Production Style
[3–4 lines: voice type, visual language, aspect ratio, no-face confirmation]

---

## Full Script 🎙

For EACH section, output in this exact block format:

### 1. Hook + Problem  `⏱ 0:00–0:20`

**[NARRATION — VOICEOVER]**
> The spoken text goes here, in short TTS-friendly sentences.

**[VISUALS]**
- `00:00–00:04` — Deep-sea murk, slow push in. Text overlay: "Light fails at 20 metres."
- `00:04–00:09` — Ghost net on seabed. Text overlay: "Abandoned fishing gear."
- ...

[Repeat the same block format for all five sections.]

---

## Production Cues 🎥
- Voice: AI TTS, warm neutral documentary tone, no music under narration
- B-roll source suggestions per section
- Where screen recording is mandatory
- Where motion graphics / text overlays are used
- Music and SFX cues
- Pacing and silence beats

---

## Voiceover Script (Clean Copy) 📄
A single clean block of ONLY the narration text, in order, with no visual cues and no markup.
This is what gets pasted directly into the TTS tool.

---

## On-Screen Text Overlays 🔤
A flat list of every text overlay, in order, with the timestamp it appears.
These are the only words a silent viewer sees, so keep them under 7 words each.

---

## Shot List 🎞
A table: | # | Timecode | Shot Type | Description | Source |
Shot Type must be one of: B-ROLL, SCREEN RECORDING, MOTION GRAPHIC, TEXT CARD, MAP, 3D/ANIMATED.
Source must be one of: PPT, APP SCREEN, EXTERNAL STOCK, GENERATED.

---

## Timing Breakdown ⏱

| Section | Target Duration |
|---|---:|
| Hook + Problem | XX sec |
| Solution | XX sec |
| Architecture | XX sec |
| Live Demo | XX sec |
| Impact + Closing | XX sec |
| **Total** | **X min Y sec** |

The total must match the requested duration.

---

## Asset Checklist 📦
A checklist of every external asset the editor must source, with search keywords.
Cover: ghost net / ghost gear, Dugong, sea turtle, side-scan sonar waterfall, AUV,
towed sonar fish, coral reef, Indian Coast Guard vessel, diver with cutting shears,
Gulf of Mannar coastline, underwater murk, bathymetric map.

---

## Anti-Fabrication Report 🔍
A short list of every claim in the script, mapped to its source:
- Claim → [PPT] or [GROUND TRUTH] or [DEMO DETAIL TO VERIFY]
This section proves the script contains zero invented technical details.

---

## One-Line Video Title 🎯
ONE title. Clear, search-friendly, technically interesting, NOT clickbait.

---

## YouTube Description 📝
Concise description: problem, solution, technologies, prototype, purpose.
Use only supported information.

---

## Tags 🏷
10–12 search tags.

---

## Final Quality Check 🔥
Verify silently, then output as a pass/fail list:
✓ Script is faceless — no presenter, no face, no "on my screen" phrasing
✓ Every narration block has a paired visual direction
✓ Narration is TTS-safe (short sentences, no symbols, speakable numbers)
✓ Script matches the PPT and the GROUND TRUTH block
✓ No unsupported claims
✓ No fabricated AUV model, sensor part number, or hardware
✓ No invented buttons, screens, or API responses
✓ Problem clearly explained
✓ Solution clearly explained
✓ Software architecture explained
✓ Hardware explained, generically
✓ Live demo included and realistic
✓ Clean-seabed zero-false-positive proof included
✓ Script fits the requested duration
✓ Reads naturally when spoken aloud
✓ It does not read the PPT slide by slide
✓ Clean copy-paste TTS block provided
✓ Anti-fabrication report complete

==================================================
IMPORTANT
==================================================
- The attached PPT plus the GROUND TRUTH block is the ONLY source of truth.
- Do not use information from unrelated projects.
- Do not insert information from previous conversations.
- Do not assume the project uses technologies not present in the PPT or ground truth.
- Do not invent implementation details, hardware part numbers, or demo clicks.
- Your job is to transform the SIH material into a polished, technically credible,
  **faceless** YouTube video script that an editor can produce without a camera.

# ==== END PASTE HERE ====
