// Adaptive Meta-Scheduler — control panel client
// Talks only to this Flask app's own /api/* routes, which call the real
// Python simulator/schedulers/LSTM/bandit — nothing here re-implements
// scheduling logic in JS.

const PROCESS_PALETTE = [
  "#E8A33D", "#5FC9D0", "#9CBF7A", "#D98CA0", "#8FA6E8",
  "#D0B24C", "#6FBFA8", "#C08FD9", "#7A93B0", "#E08A6B",
];

const ALGO_LABELS = {
  FCFS: "FCFS",
  SJF_Oracle: "SJF (oracle)",
  RoundRobin: "Round Robin",
  Priority: "Priority",
  MLFQ: "MLFQ",
  Lottery: "Lottery",
  EDF: "EDF",
  SJF_Predicted: "SJF (predicted)",
  RL_Adaptive: "RL Adaptive",
};

let state = {
  processes: null,
  label: null,
  config: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function colorForPid(pid) {
  return PROCESS_PALETTE[pid % PROCESS_PALETTE.length];
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `Request to ${url} failed`);
  }
  return res.json();
}

// ── source toggle ──────────────────────────────────────────────────────

function initSourceToggle() {
  $$(".source-toggle__btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".source-toggle__btn").forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      const isReal = btn.dataset.source === "real";
      $("#controls-synthetic").classList.toggle("is-hidden", isReal);
      $("#controls-real").classList.toggle("is-hidden", !isReal);
    });
  });

  const slider = $("#window-index");
  slider.addEventListener("input", () => {
    $("#window-index-label").textContent = Number(slider.value) + 1;
  });
}

function currentSource() {
  const active = $(".source-toggle__btn.is-active");
  return active ? active.dataset.source : "synthetic";
}

// ── workload generation ────────────────────────────────────────────────

async function generateWorkload() {
  const btn = $("#btn-generate");
  btn.disabled = true;
  btn.textContent = "Generating…";
  try {
    const source = currentSource();
    const body = { source };
    if (source === "synthetic") {
      body.workload_type = $("#workload-type").value;
    } else {
      body.window_index = Number($("#window-index").value);
    }
    const data = await postJSON("/api/workload", body);
    state.processes = data.processes;
    state.label = data.label;
    $("#workload-label").textContent = `${data.label} — ${data.processes.length} processes loaded`;
    $$(".algo-chip").forEach((c) => (c.disabled = false));
    $("#btn-compare").disabled = false;
  } catch (err) {
    $("#workload-label").textContent = `Error: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate workload";
  }
}

// ── algorithm grid ──────────────────────────────────────────────────────

function buildAlgoGrid(allAlgos) {
  const grid = $("#algo-grid");
  grid.innerHTML = "";
  allAlgos.forEach((algo) => {
    const chip = document.createElement("button");
    chip.className = "algo-chip";
    if (algo === "RL_Adaptive") chip.classList.add("is-hero");
    chip.textContent = ALGO_LABELS[algo] || algo;
    chip.dataset.algo = algo;
    chip.disabled = true;
    chip.addEventListener("click", () => runAlgorithm(algo, chip));
    grid.appendChild(chip);
  });
}

async function runAlgorithm(algo, chipEl) {
  if (!state.processes) return;
  $$(".algo-chip").forEach((c) => c.classList.remove("is-selected"));
  chipEl.classList.add("is-selected");

  $("#result-compare").classList.add("is-hidden");
  $("#result-empty").classList.add("is-hidden");
  $("#result-single").classList.remove("is-hidden");

  try {
    const data = await postJSON("/api/run", { processes: state.processes, algorithm: algo });
    renderSingleResult(data);
  } catch (err) {
    $("#result-single").innerHTML = `<p class="truncated-note">${err.message}</p>`;
  }
}

// ── single-run rendering (metrics + gantt) ──────────────────────────────

function renderSingleResult(data) {
  const m = data.metrics;
  const rows = [
    ["Avg wait", m.avg_waiting_time, true],
    ["Avg turnaround", m.avg_turnaround_time, false],
    ["CPU util %", m.cpu_utilization, false],
    ["Completed", m.completed_count, false],
  ];
  $("#metrics-row").innerHTML = rows
    .map(
      ([label, val, amber]) => `
      <div class="metric">
        <div class="metric__label">${label}</div>
        <div class="metric__value${amber ? " is-amber" : ""}">${fmt(val)}</div>
      </div>`
    )
    .join("");

  const chosenNote = $("#chosen-note");
  if (data.chosen_scheduler) {
    chosenNote.textContent = `RL bandit chose: ${ALGO_LABELS[chosenSchedulerKey(data.chosen_scheduler)] || data.chosen_scheduler} — this run is that scheduler, executing for real.`;
    chosenNote.classList.remove("is-hidden");
  } else {
    chosenNote.classList.add("is-hidden");
  }

  $("#truncated-note").classList.toggle("is-hidden", !data.truncated);

  renderGantt(data.timeline);
}

function chosenSchedulerKey(name) {
  // trainer.py reports "SJF_Pred" — map back to the SJF_Predicted label
  return name === "SJF_Pred" ? "SJF_Predicted" : name;
}

function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number" && !Number.isInteger(v)) return v.toFixed(2);
  return String(v);
}

function renderGantt(timeline) {
  const gantt = $("#gantt");
  const axis = $("#gantt-axis");
  gantt.innerHTML = "";
  if (!timeline || timeline.length === 0) {
    axis.innerHTML = "";
    return;
  }

  // Collapse consecutive same-pid ticks into segments.
  const segments = [];
  for (const { tick, pid } of timeline) {
    const last = segments[segments.length - 1];
    if (last && last.pid === pid && tick === last.end + 1) {
      last.end = tick;
    } else {
      segments.push({ pid, start: tick, end: tick });
    }
  }

  const totalTicks = timeline[timeline.length - 1].tick + 1;
  segments.forEach((seg) => {
    if (seg.pid === null) return; // idle — leave background showing
    const div = document.createElement("div");
    div.className = "gantt__segment";
    div.style.left = `${(seg.start / totalTicks) * 100}%`;
    div.style.width = `${Math.max(((seg.end - seg.start + 1) / totalTicks) * 100, 0.15)}%`;
    div.style.background = colorForPid(seg.pid);
    div.title = `pid ${seg.pid}: tick ${seg.start}–${seg.end}`;
    gantt.appendChild(div);
  });

  axis.innerHTML = `<span>0</span><span>${Math.round(totalTicks / 2)}</span><span>${totalTicks}</span>`;
}

// ── compare-all rendering ────────────────────────────────────────────────

async function compareAll() {
  const btn = $("#btn-compare");
  btn.disabled = true;
  btn.textContent = "Running all 9…";
  $$(".algo-chip").forEach((c) => c.classList.remove("is-selected"));
  $("#result-empty").classList.add("is-hidden");
  $("#result-single").classList.add("is-hidden");
  $("#result-compare").classList.remove("is-hidden");

  try {
    const data = await postJSON("/api/compare", { processes: state.processes });
    renderCompare(data.results);
  } catch (err) {
    $("#compare-chart").innerHTML = `<p class="truncated-note">${err.message}</p>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Compare all 9";
  }
}

function renderCompare(results) {
  const sorted = [...results].sort((a, b) => a.avg_waiting_time - b.avg_waiting_time);
  const maxWait = Math.max(...sorted.map((r) => r.avg_waiting_time));
  $("#compare-chart").innerHTML = sorted
    .map((r) => {
      const isHero = r.algorithm === "RL_Adaptive";
      const pct = maxWait > 0 ? (r.avg_waiting_time / maxWait) * 100 : 0;
      const label = ALGO_LABELS[r.algorithm] || r.algorithm;
      const suffix =
        r.algorithm === "RL_Adaptive" && r.chosen_scheduler
          ? ` <span style="opacity:.6">(chose ${ALGO_LABELS[chosenSchedulerKey(r.chosen_scheduler)] || r.chosen_scheduler})</span>`
          : "";
      return `
        <div class="compare-row${isHero ? " is-hero" : ""}">
          <div class="compare-row__name">${label}</div>
          <div class="compare-row__track"><div class="compare-row__bar" style="width:${pct}%"></div></div>
          <div class="compare-row__value">${fmt(r.avg_waiting_time)}${suffix}</div>
        </div>`;
    })
    .join("");
}

// ── init ──────────────────────────────────────────────────────────────

async function init() {
  initSourceToggle();
  $("#btn-generate").addEventListener("click", generateWorkload);
  $("#btn-compare").addEventListener("click", compareAll);

  const config = await fetch("/api/config").then((r) => r.json());
  state.config = config;
  buildAlgoGrid(config.all_algorithms);
}

init();
