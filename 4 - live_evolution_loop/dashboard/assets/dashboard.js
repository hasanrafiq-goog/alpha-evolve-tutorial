/**
 * Google Cloud AlphaEvolve • Architecture Search Console
 * Real-time client-side polling, KPI compute, execution log rendering,
 * Chart.js multi-objective visualizations, and code inspection modal.
 */

let selectedSessionId = "active";
let currentSessionData = null;
let trajectoryChart = null;
let paretoChart = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  fetchSessionList();
  pollSessionState();
  setInterval(pollSessionState, 1000);
});

function setupEventListeners() {
  const sessionSelect = document.getElementById("session-select");
  if (sessionSelect) {
    sessionSelect.addEventListener("change", (e) => {
      selectedSessionId = e.target.value;
      if (selectedSessionId === "active") {
        pollSessionState();
      } else {
        loadHistoricalSession(selectedSessionId);
      }
    });
  }

  // Modal close handlers
  const modalClose = document.getElementById("modal-close");
  const modal = document.getElementById("candidate-modal");
  if (modalClose && modal) {
    modalClose.addEventListener("click", () => closeModal());
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && modal.classList.contains("open")) {
        closeModal();
      }
    });
  }

  const copyBtn = document.getElementById("modal-copy-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const code = document.getElementById("modal-code").textContent;
      navigator.clipboard.writeText(code).then(() => {
        copyBtn.textContent = "Copied!";
        setTimeout(() => (copyBtn.textContent = "Copy Code"), 2000);
      });
    });
  }
}

// 1. Fetch available sessions list
async function fetchSessionList() {
  try {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    const data = await res.json();
    const select = document.getElementById("session-select");
    if (!select || !data.sessions) return;

    const currentVal = select.value || "active";
    select.innerHTML = "";
    data.sessions.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = s.label;
      if (s.id === currentVal) opt.selected = true;
      select.appendChild(opt);
    });
  } catch (err) {
    console.debug("Could not refresh session list:", err);
  }
}

// 2. Poll Active Session
async function pollSessionState() {
  if (selectedSessionId !== "active") return;
  try {
    const res = await fetch(`/runs.json?_t=${Date.now()}`);
    if (res.ok) {
      const data = await res.json();
      currentSessionData = data;
      renderDashboard(data);
    }
  } catch (err) {
    console.debug("Poll fetch error:", err);
  }
}

// 3. Load Historical Session
async function loadHistoricalSession(sessionId) {
  try {
    const res = await fetch(`/api/sessions/${sessionId}`);
    if (res.ok) {
      const data = await res.json();
      currentSessionData = data;
      renderDashboard(data);
    }
  } catch (err) {
    console.error("Failed to load historical session:", err);
  }
}

// 4. Render Master Dashboard
function renderDashboard(data) {
  updateHeader(data);
  updateKPIs(data);
  renderTable(data);
  renderTrajectoryChart(data);
  renderParetoChart(data);
}

function updateHeader(data) {
  const statusBadge = document.getElementById("session-status-badge");
  const statusText = document.getElementById("status-text");
  const budgetBadge = document.getElementById("budget-badge");
  const metricBadge = document.getElementById("metric-badge");

  if (budgetBadge && data.target_params) {
    budgetBadge.textContent = `Budget: < ${data.target_params.toLocaleString()} Params`;
  }

  if (statusBadge && statusText) {
    const allRuns = data.runs || [];
    const evolvedCount = allRuns.filter((r) => r.run_number > 0 && r.status !== "SEED").length;
    const maxProg = data.max_programs || 40;

    if (data.status === "COMPLETED") {
      statusBadge.className = "badge badge-success";
      statusText.textContent = `Optimization Completed (${evolvedCount}/${maxProg})`;
    } else if (data.status === "IN_PROGRESS" || data.status === "INITIALIZING") {
      statusBadge.className = "badge badge-warning";
      statusText.textContent = `In Progress (${evolvedCount}/${maxProg})`;
    } else {
      statusBadge.className = "badge badge-outline";
      statusText.textContent = data.status || "Unknown";
    }
  }
}

function updateKPIs(data) {
  const kpis = data.kpis || {};
  const topAcc = kpis.top_accuracy;
  const accDelta = kpis.accuracy_delta;
  const relErr = kpis.relative_error_reduction;
  const baseAcc = kpis.baseline_accuracy || data.baseline_accuracy || 0.954;

  // Accuracy Card (Rank 1 Champion)
  const topAccEl = document.getElementById("kpi-top-acc");
  const accDeltaEl = document.getElementById("kpi-acc-delta");
  const accSubEl = document.getElementById("kpi-acc-subtext");
  if (topAccEl) {
    topAccEl.textContent = topAcc != null ? `${(topAcc * 100).toFixed(2)}%` : "—";
  }
  if (accDeltaEl) {
    if (accDelta != null) {
      const sign = accDelta >= 0 ? "+" : "";
      accDeltaEl.textContent = `${sign}${(accDelta * 100).toFixed(2)}%`;
      accDeltaEl.style.display = "inline-block";
    } else {
      accDeltaEl.textContent = "Waiting";
    }
  }
  if (accSubEl) {
    if (relErr != null) {
      accSubEl.textContent = `Baseline Seed: ${(baseAcc * 100).toFixed(2)}% | Relative Error: ${relErr.toFixed(1)}%`;
    } else {
      accSubEl.textContent = `Baseline Seed: ${(baseAcc * 100).toFixed(2)}%`;
    }
  }

  // Params Card (Rank 1 Champion)
  const optParams = kpis.optimal_params;
  const paramRed = kpis.param_reduction_percent;
  const compRatio = kpis.param_compression_ratio;
  const optParamsEl = document.getElementById("kpi-optimal-params");
  const paramDeltaEl = document.getElementById("kpi-param-delta");
  const paramSubEl = document.getElementById("kpi-param-subtext");
  if (optParamsEl) {
    optParamsEl.textContent = optParams != null ? optParams.toLocaleString() : "—";
  }
  if (paramDeltaEl) {
    if (paramRed != null) {
      paramDeltaEl.textContent = `-${paramRed.toFixed(1)}% Size`;
      paramDeltaEl.style.display = "inline-block";
    } else {
      paramDeltaEl.textContent = "Pending";
    }
  }
  if (paramSubEl) {
    if (compRatio != null) {
      paramSubEl.textContent = `${compRatio.toFixed(2)}x parameter reduction (Fits in MCU L1 SRAM)`;
    } else {
      paramSubEl.textContent = `Baseline Seed: ${(kpis.baseline_params || 21898).toLocaleString()} Params`;
    }
  }

  // Efficiency Card
  const effEl = document.getElementById("kpi-efficiency");
  if (effEl && kpis.convergence_efficiency) {
    effEl.textContent = kpis.convergence_efficiency.split("(")[0].trim();
  }
}

// 5. Render Chronological Execution Log Table
function renderTable(data) {
  const tbody = document.getElementById("runs-table-body");
  const countBadge = document.getElementById("runs-count-badge");
  if (!tbody) return;

  // Filter out Generation 0 seed baseline model (run_number === 0 or status === 'SEED')
  const allRuns = data.runs || [];
  const runs = allRuns.filter((r) => r.run_number > 0 && r.status !== "SEED");

  if (countBadge) {
    countBadge.textContent = `${runs.length} Candidates Generated & Evaluated`;
  }

  tbody.innerHTML = "";

  // In-flight row
  if (data.in_flight && data.status !== "COMPLETED") {
    const ifTr = document.createElement("tr");
    ifTr.className = "row-in-flight";
    ifTr.innerHTML = `
      <td>Run #${data.in_flight.run_number}</td>
      <td><code>${data.in_flight.program_id}</code></td>
      <td><span class="badge badge-warning">Evaluating...</span></td>
      <td colspan="4"><em>Worker compiling & training candidate on proxy dataset...</em></td>
    `;
    tbody.appendChild(ifTr);
  }

  // Render evolved runs in chronological order
  runs.forEach((r) => {
    const tr = document.createElement("tr");
    // Format Rank Badge
    const rankStr = String(r.rank || "").trim();
    const isChampion = rankStr.includes("RANK 1") || r.status === "CHAMPION";
    if (isChampion) tr.className = "row-champion";

    let rankHtml = rankStr || "-";
    if (rankStr.includes("RANK 1") || rankStr === "1") {
      rankHtml = '<span class="badge badge-success">RANK 1 ⭐️</span>';
    } else if (rankStr.includes("RANK 2") || rankStr === "2") {
      rankHtml = '<span class="badge badge-accent">RANK 2</span>';
    } else if (rankStr.includes("RANK 3") || rankStr === "3") {
      rankHtml = '<span class="badge badge-accent">RANK 3</span>';
    } else if (rankStr.toLowerCase() === "base") {
      rankHtml = '<span class="badge badge-outline">Base</span>';
    } else if (rankStr === "-") {
      rankHtml = '<span style="color: #9aa0a6;">—</span>';
    }

    const rewardClass = r.nas_reward < 0.1 ? "metric-bad" : isChampion ? "metric-good" : "";
    const accClass = r.val_accuracy < 0.5 ? "metric-bad" : isChampion ? "metric-good" : "";

    tr.innerHTML = `
      <td><strong>${r.run_number}</strong></td>
      <td><code>${r.candidate_id}</code></td>
      <td>${rankHtml}</td>
      <td class="${rewardClass}">${r.nas_reward.toFixed(4)}</td>
      <td class="${accClass}">${(r.val_accuracy * 100).toFixed(2)}%</td>
      <td>${r.param_count ? r.param_count.toLocaleString() : "N/A"}</td>
      <td class="mutation-cell" title="${r.mutation_summary}">${r.mutation_summary || "Candidate Mutation"}</td>
    `;

    tr.addEventListener("click", () => openModal(r));
    tbody.appendChild(tr);
  });
}

// 6. Multi-Objective Reward Trajectory Chart (Line)
function renderTrajectoryChart(data) {
  const canvas = document.getElementById("trajectory-chart");
  if (!canvas || !window.Chart) return;

  // Filter out seed baseline
  const evolvedRuns = (data.runs || []).filter(
    (r) => r.run_number > 0 && r.status !== "SEED"
  );
  if (evolvedRuns.length === 0) return;

  const labels = [];
  const candidateRewards = [];
  const pointColors = [];
  const pointRadii = [];
  let runningMax = -1.0;

  evolvedRuns.forEach((r) => {
    labels.push(`R${r.run_number}`);
    const rew = Number(r.nas_reward) || 0.0;
    candidateRewards.push(rew);

    // Milestone breakthrough gets green dot (#34a853), regular exploration gets blue (#1a73e8)
    if (rew > runningMax && rew > 0.05) {
      runningMax = rew;
      pointColors.push("#34a853");
      pointRadii.push(6);
    } else {
      pointColors.push("#1a73e8");
      pointRadii.push(4);
    }
  });

  if (trajectoryChart) {
    trajectoryChart.data.labels = labels;
    trajectoryChart.data.datasets[0].data = candidateRewards;
    trajectoryChart.data.datasets[0].pointBackgroundColor = pointColors;
    trajectoryChart.data.datasets[0].pointBorderColor = pointColors;
    trajectoryChart.data.datasets[0].pointRadius = pointRadii;
    trajectoryChart.update("none");
    return;
  }

  const ctx = canvas.getContext("2d");
  trajectoryChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Candidate Reward",
          data: candidateRewards,
          borderColor: "#1a73e8",
          backgroundColor: "rgba(26, 115, 232, 0.06)",
          borderWidth: 2,
          fill: false,
          tension: 0.15,
          pointBackgroundColor: pointColors,
          pointBorderColor: pointColors,
          pointRadius: pointRadii,
          pointHoverRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const idx = ctx.dataIndex;
              const isMilestone = pointColors[idx] === "#34a853";
              const tag = isMilestone ? " ⭐️ (New High-Water Mark)" : "";
              return ` Reward: ${ctx.parsed.y.toFixed(4)}${tag}`;
            },
          },
        },
      },
      scales: {
        x: { grid: { display: false } },
        y: {
          min: 0.0,
          suggestedMax: 1.0,
          grid: { color: "#f1f3f4" },
        },
      },
    },
  });
}

// 7. Pareto Frontier Scatter Plot
function renderParetoChart(data) {
  const canvas = document.getElementById("pareto-chart");
  if (!canvas || !window.Chart) return;

  // Filter out seed baseline and invalid runs
  const runs = (data.runs || []).filter(
    (r) => r.run_number > 0 && r.status !== "SEED" && r.param_count > 0 && r.val_accuracy > 0
  );
  if (runs.length === 0) return;

  // Identify Pareto optimal models
  const paretoPoints = [];
  const dominatedPoints = [];

  // Calculate non-dominated
  runs.forEach((cand) => {
    let dominated = false;
    for (const other of runs) {
      if (other === cand) continue;
      if (
        other.val_accuracy >= cand.val_accuracy &&
        other.param_count <= cand.param_count &&
        (other.val_accuracy > cand.val_accuracy || other.param_count < cand.param_count)
      ) {
        dominated = true;
        break;
      }
    }
    const pt = { x: cand.param_count, y: cand.val_accuracy * 100, run: cand };
    if (!dominated) {
      paretoPoints.push(pt);
    } else {
      dominatedPoints.push(pt);
    }
  });

  paretoPoints.sort((a, b) => a.x - b.x);

  const datasets = [
    {
      label: "Dominated Models",
      data: dominatedPoints,
      backgroundColor: "#4285f4",
      pointRadius: 4,
      pointHoverRadius: 6,
    },
    {
      label: "Optimal Pareto Frontier",
      data: paretoPoints,
      backgroundColor: "#34a853",
      borderColor: "#34a853",
      borderDash: [5, 5],
      showLine: true,
      pointRadius: 6,
      pointHoverRadius: 8,
    },
  ];

  // Dynamic y-axis lower bound: accommodate proxy models (< 70%) with dynamic padding
  const minAcc = runs.length > 0 ? Math.min(...runs.map((r) => r.val_accuracy * 100)) : 0;
  const yMin = Math.max(0, Math.floor(Math.max(0, minAcc - 5) / 5) * 5);

  // Subtle budget target guide line
  if (data.target_params) {
    datasets.push({
      label: `Budget Target (< ${data.target_params.toLocaleString()}p)`,
      data: [
        { x: data.target_params, y: yMin },
        { x: data.target_params, y: 100 },
      ],
      borderColor: "rgba(234, 67, 53, 0.4)",
      borderDash: [4, 4],
      borderWidth: 1.5,
      pointRadius: 0,
      showLine: true,
      fill: false,
    });
  }

  // Target accuracy horizontal guide line
  if (data.target_accuracy) {
    const targetAccPct = data.target_accuracy > 1 ? data.target_accuracy : data.target_accuracy * 100;
    const maxX = data.target_params ? Math.round(data.target_params * 1.5) : 5000;
    datasets.push({
      label: `Accuracy Target (${targetAccPct.toFixed(1)}%)`,
      data: [
        { x: 0, y: targetAccPct },
        { x: maxX, y: targetAccPct },
      ],
      borderColor: "rgba(52, 168, 83, 0.4)",
      borderDash: [4, 4],
      borderWidth: 1.5,
      pointRadius: 0,
      showLine: true,
      fill: false,
    });
  }

  if (paretoChart) {
    paretoChart.data.datasets = datasets;
    if (paretoChart.options.scales && paretoChart.options.scales.y) {
      paretoChart.options.scales.y.min = yMin;
    }
    paretoChart.update("none");
    return;
  }

  const ctx = canvas.getContext("2d");
  paretoChart = new Chart(ctx, {
    type: "scatter",
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      onClick: (e, elements) => {
        if (elements.length > 0) {
          const el = elements[0];
          const ds = datasets[el.datasetIndex];
          if (ds && ds.data) {
            const pt = ds.data[el.index];
            if (pt && pt.run) openModal(pt.run);
          }
        }
      },
      plugins: {
        legend: { position: "top", labels: { boxWidth: 10, font: { size: 10 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const pt = ctx.raw;
              if (!pt || !pt.run) return "";
              const runLabel = `Run #${pt.run.run_number}`;
              return `${runLabel}: ${(pt.y).toFixed(2)}% Acc (${pt.x.toLocaleString()} Params)`;
            },
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Parameter Count →", font: { size: 11 } },
          min: 0,
          suggestedMax: data.target_params ? Math.round(data.target_params * 1.5) : 5000,
          grid: { color: "#f1f3f4" },
        },
        y: {
          title: { display: true, text: "Validation Accuracy (%) ↑", font: { size: 11 } },
          min: yMin,
          max: 100,
          grid: { color: "#f1f3f4" },
        },
      },
    },
  });
}

// 8. Interactive Candidate Inspection Modal
function openModal(candidate) {
  const modal = document.getElementById("candidate-modal");
  const title = document.getElementById("modal-title");
  const rankBadge = document.getElementById("modal-rank-badge");
  const metricsGrid = document.getElementById("modal-metrics");
  const codeBlock = document.getElementById("modal-code");
  const insightsContainer = document.getElementById("modal-insights");

  if (!modal || !candidate) return;

  const runName = candidate.run_number === 0 ? "Baseline Seed Model" : `Candidate Run #${candidate.run_number}`;
  title.textContent = `${runName} (${candidate.candidate_id})`;

  if (rankBadge) {
    rankBadge.textContent = `Rank ${candidate.rank}`;
    rankBadge.className = candidate.rank.includes("1") ? "badge badge-success" : "badge badge-accent";
  }

  if (metricsGrid) {
    metricsGrid.innerHTML = `
      <div class="metric-box">
        <div class="metric-box-label">NAS Reward</div>
        <div class="metric-box-val">${candidate.nas_reward.toFixed(4)}</div>
      </div>
      <div class="metric-box">
        <div class="metric-box-label">Val Accuracy</div>
        <div class="metric-box-val">${(candidate.val_accuracy * 100).toFixed(2)}%</div>
      </div>
      <div class="metric-box">
        <div class="metric-box-label">Parameters</div>
        <div class="metric-box-val">${candidate.param_count ? candidate.param_count.toLocaleString() : "N/A"}</div>
      </div>
      <div class="metric-box">
        <div class="metric-box-label">Train Time</div>
        <div class="metric-box-val">${candidate.train_time_sec}s</div>
      </div>
    `;
  }

  if (codeBlock) {
    codeBlock.textContent = candidate.code_content || "# Code not available";
  }

  if (insightsContainer) {
    const list = candidate.insights || [];
    if (list.length === 0) {
      insightsContainer.innerHTML = "<em>No diagnostic warnings or tracebacks recorded.</em>";
    } else {
      insightsContainer.innerHTML = list.map((item) => `<div>• ${escapeHtml(item)}</div>`).join("");
    }
  }

  modal.classList.add("open");
  modal.setAttribute("aria-hidden", "false");
}

function closeModal() {
  const modal = document.getElementById("candidate-modal");
  if (modal) {
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
