/**
 * RailETA Production Dashboard — Real-Time WebSocket & State Controller
 */

let ws = null;
let currentTrain = 12952;
let currentMode = "simulator";
let lastPacket = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initEventListeners();
  connectWebSocket();
  fetchInitialMetadata();
  fetchRouteTimetable(currentTrain);
});

function initEventListeners() {
  const trainSelect = document.getElementById("train-select");
  trainSelect.addEventListener("change", (e) => {
    currentTrain = parseInt(e.target.value, 10);
    switchTrain(currentTrain);
  });

  const btnSim = document.getElementById("btn-mode-sim");
  const btnRr = document.getElementById("btn-mode-rr");

  btnSim.addEventListener("click", () => switchMode("simulator"));
  btnRr.addEventListener("click", () => switchMode("railradar"));
}

function switchMode(mode) {
  currentMode = mode;
  document.getElementById("btn-mode-sim").classList.toggle("active", mode === "simulator");
  document.getElementById("btn-mode-rr").classList.toggle("active", mode === "railradar");

  fetch(`/api/stream/mode?mode=${mode}&train_no=${currentTrain}`, { method: "POST" })
    .then(r => r.json())
    .catch(err => console.error("Error switching stream mode:", err));

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ set_mode: mode, train_no: currentTrain }));
  }
}

function switchTrain(trainNo) {
  fetch(`/api/stream/mode?mode=${currentMode}&train_no=${trainNo}`, { method: "POST" })
    .then(r => r.json())
    .catch(err => console.error("Error switching train:", err));

  fetchRouteTimetable(trainNo);

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ set_mode: currentMode, train_no: trainNo }));
  }
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;

  const statusEl = document.getElementById("ws-status");
  const statusIndicator = statusEl.querySelector(".status-indicator");
  const connText = document.getElementById("conn-text");

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    statusIndicator.className = "status-indicator online";
    connText.textContent = "LIVE FEED CONNECTED";
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateDashboard(data);
    } catch (err) {
      console.error("Error parsing WebSocket payload:", err);
    }
  };

  ws.onerror = (err) => {
    console.warn("WebSocket error:", err);
    statusIndicator.className = "status-indicator";
    statusIndicator.style.background = "#f43f5e";
    connText.textContent = "FEED ERROR";
  };

  ws.onclose = () => {
    statusIndicator.className = "status-indicator";
    statusIndicator.style.background = "#f59e0b";
    connText.textContent = "RECONNECTING...";
    setTimeout(connectWebSocket, 3000);
  };
}

function updateDashboard(packet) {
  lastPacket = packet;

  // Panel 1: Live Train Card
  document.getElementById("train-no-display").textContent = packet.train_no || currentTrain;
  document.getElementById("train-name-display").textContent = packet.train_name || "EXPRESS";
  document.getElementById("current-station").textContent = `${packet.station_code} (${packet.station_name || packet.station_code})`;
  document.getElementById("telemetry-delay").textContent = `+${packet.current_delay_min || 0} min`;
  document.getElementById("telemetry-speed").textContent = `${Math.round(packet.speed_kmph || 0)} km/h`;
  document.getElementById("telemetry-progress").textContent = `Stop ${packet.seq || 1} of ${packet.total_stations || 8}`;
  document.getElementById("telemetry-event").textContent = packet.event_type || "NONE";
  document.getElementById("passenger-msg-text").textContent = packet.passenger_message || "Normal operational running";

  // Panel 2: Prediction Block
  document.getElementById("sched-arrival-display").textContent = packet.scheduled_arrival || "--:--:--";
  document.getElementById("predicted-eta-display").textContent = packet.predicted_eta || "--:--:--";
  
  if (packet.predicted_eta_range && packet.predicted_eta_range.length === 2) {
    document.getElementById("predicted-eta-range-display").textContent = 
      `${packet.predicted_eta_range[0]} – ${packet.predicted_eta_range[1]}`;
  }

  const pDelay = packet.predicted_delay_min !== undefined ? packet.predicted_delay_min : 0;
  document.getElementById("pred-delay-display").innerHTML = `<strong>+${pDelay} min</strong>`;

  if (packet.predicted_delay_range_min && packet.predicted_delay_range_min.length === 2) {
    document.getElementById("pred-delay-range-display").textContent = 
      `(Range: +${packet.predicted_delay_range_min[0]}m to +${packet.predicted_delay_range_min[1]}m, 84% coverage)`;
  }

  // Panel 3: Confidence Badge
  const confBadge = document.getElementById("confidence-badge");
  const confLabel = document.getElementById("confidence-label");
  const confSubtext = document.getElementById("conf-subtext");
  const conf = packet.confidence || "Medium";

  confBadge.className = `confidence-badge conf-${conf.toLowerCase()}`;
  confLabel.textContent = `${conf} Confidence`;

  if (conf === "High") {
    confSubtext.textContent = "Expected prediction error \u2264 13.1 min (~78% within \u00b110 min). Lower risk section.";
  } else if (conf === "Medium") {
    confSubtext.textContent = "Expected error 13.1 to 21.6 min (~45% within \u00b110 min). Moderate section congestion variance.";
  } else {
    confSubtext.textContent = "Expected error > 21.6 min or chronic/unseen service. High dispersion interval.";
  }

  // Panel 4: Severe Delay Risk Gauge
  const severeRisk = packet.severe_delay_risk !== undefined ? packet.severe_delay_risk : 0;
  const severePct = Math.round(severeRisk * 100);
  document.getElementById("severe-risk-pct").textContent = `${severePct}%`;
  document.getElementById("severe-risk-bar").style.width = `${Math.min(100, Math.max(2, severePct))}%`;

  const riskLabel = document.getElementById("severe-risk-label");
  if (severeRisk >= 0.5) {
    riskLabel.textContent = "CRITICAL RISK (>360m data boundary)";
    riskLabel.style.color = "var(--accent-rose)";
  } else if (severeRisk >= 0.2) {
    riskLabel.textContent = "Moderate Delay Exposure";
    riskLabel.style.color = "var(--accent-amber)";
  } else {
    riskLabel.textContent = "Nominal Risk (<0.10)";
    riskLabel.style.color = "var(--accent-emerald)";
  }

  // Panel 5: Chronic Tier Badge
  const tier = packet.chronic_delay_tier || "T3";
  const tierBadge = document.getElementById("chronic-tier-badge");
  tierBadge.textContent = tier;
  const tierTitle = document.getElementById("chronic-tier-title");
  const tierDesc = document.getElementById("chronic-tier-desc");
  const fallbackFlag = document.getElementById("chronic-fallback-flag");

  if (packet.chronic_tier_is_fallback) {
    fallbackFlag.classList.remove("hidden");
  } else {
    fallbackFlag.classList.add("hidden");
  }

  if (tier === "T1" || tier === "T1_best") {
    tierTitle.textContent = "Tier 1 — Punctual Benchmark";
    tierDesc.textContent = "Highest corridor punctuality rate (>92%). Minimal historical delay drift.";
  } else if (tier === "T2") {
    tierTitle.textContent = "Tier 2 — High Punctuality Corridor";
    tierDesc.textContent = "Historical punctuality exceeds 85%. Minimal persistent buffer penalty.";
  } else if (tier === "T3") {
    tierTitle.textContent = "Tier 3 — Baseline Performance";
    tierDesc.textContent = "Standard schedule variance (~70% on-time). Production default prior.";
  } else if (tier === "T4") {
    tierTitle.textContent = "Tier 4 — Recurring Delay Profile";
    tierDesc.textContent = "High susceptibility to corridor regulation and junction bottlenecks.";
  } else {
    tierTitle.textContent = "Tier 5 — Chronic Delay Service";
    tierDesc.textContent = "Chronic disruption corridor (average historical delay > 50 min). Wide interval applied.";
  }

  // Panel 6: "Why this prediction?" Panel
  if (packet.explanations && Array.isArray(packet.explanations)) {
    renderWhyDrivers(packet.explanations);
  }

  // Running MAE Counter
  if (packet.running_mae !== undefined) {
    document.getElementById("live-mae-counter").textContent = `${packet.running_mae} min`;
  }

  // Highlight current row in route timetable
  highlightCurrentRouteStop(packet.station_code);
}

function renderWhyDrivers(drivers) {
  const container = document.getElementById("why-drivers-list");
  container.innerHTML = "";

  drivers.forEach(d => {
    const item = document.createElement("div");
    item.className = "driver-item";
    const impactClass = d.direction === "increase" ? "positive" : "neutral";
    item.innerHTML = `
      <div class="driver-header">
        <span class="driver-factor">${escapeHtml(d.factor)}</span>
        <span class="driver-impact ${impactClass}">${escapeHtml(d.impact)}</span>
      </div>
      <p class="driver-desc">${escapeHtml(d.description)}</p>
    `;
    container.appendChild(item);
  });
}

function fetchInitialMetadata() {
  fetch("/model/metrics")
    .then(r => r.json())
    .then(meta => {
      if (meta.mae_non_capped) {
        document.getElementById("metric-mae").textContent = `${meta.mae_non_capped.toFixed(2)}m`;
      }
    })
    .catch(err => console.log("Metadata load note:", err));
}

function fetchRouteTimetable(trainNo) {
  fetch(`/train/${trainNo}/eta`)
    .then(r => r.json())
    .then(data => {
      renderRouteTimetable(data.stations);
    })
    .catch(err => console.error("Error loading route timetable:", err));
}

function renderRouteTimetable(stations) {
  const tbody = document.getElementById("route-table-body");
  tbody.innerHTML = "";

  if (!stations || stations.length === 0) {
    tbody.innerHTML = "<tr><td colspan='7' style='text-align:center;'>No route data available</td></tr>";
    return;
  }

  stations.forEach(st => {
    const tr = document.createElement("tr");
    tr.id = `row-st-${st.station_code}`;
    const pDelay = st.predicted_delay_min !== null ? `+${st.predicted_delay_min}m` : "On Time";
    const pEta = st.predicted_eta || st.scheduled_arrival;

    tr.innerHTML = `
      <td>${st.seq}</td>
      <td><strong>${st.station_code}</strong> <span style="color:var(--text-muted);font-size:11px;">(${st.station_name || ''})</span></td>
      <td>${st.distance_km} km</td>
      <td>${st.scheduled_arrival}</td>
      <td style="font-family:var(--font-mono);font-weight:700;color:var(--accent-cyan);">${pEta}</td>
      <td style="font-family:var(--font-mono);">${pDelay}</td>
      <td><span style="font-size:11px;padding:2px 6px;border-radius:4px;background:rgba(255,255,255,0.06);">${st.confidence || 'Medium'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function highlightCurrentRouteStop(stationCode) {
  if (!stationCode) return;
  const rows = document.querySelectorAll("#route-table-body tr");
  rows.forEach(r => r.classList.remove("current-row"));
  const target = document.getElementById(`row-st-${stationCode}`);
  if (target) {
    target.classList.add("current-row");
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
