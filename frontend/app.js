const apiInput = document.getElementById("apiBase");
const symbolSelect = document.getElementById("symbolSelect");
const modelSelect = document.getElementById("modelSelect");
const refreshBtn = document.getElementById("refreshBtn");
const assetLabel = document.getElementById("assetLabel");
const historyTitle = document.getElementById("historyTitle");
const forecastTitle = document.getElementById("forecastTitle");
const assetSwitch = document.getElementById("assetSwitch");
const chartSwitch = document.getElementById("chartSwitch");
const historyCanvas = document.getElementById("historyChart");
const candleDiv = document.getElementById("candlestickChart");

const latestCloseEl = document.getElementById("latestClose");
const dailyChangeEl = document.getElementById("dailyChange");
const forecastAvgEl = document.getElementById("forecastAvg");
const forecastTableBody = document.getElementById("forecastTableBody");
const todayValueEl = document.getElementById("todayValue");
const dataAsOfValueEl = document.getElementById("dataAsOfValue");
const modelValueEl = document.getElementById("modelValue");
const capabilityStatusEl = document.getElementById("capabilityStatus");
const forecastEvidenceEl = document.getElementById("forecastEvidence");
const researchStatusEl = document.getElementById("researchStatus");
const researchDetailsEl = document.getElementById("researchDetails");
const researchRowsEl = document.getElementById("researchEvidenceRows");
const researchWinsEl = document.getElementById("researchWins");
const researchCiEl = document.getElementById("researchCi");
const researchCoverageEl = document.getElementById("researchCoverage");
const explanationStatusEl = document.getElementById("explanationStatus");
const explanationDetailsEl = document.getElementById("explanationDetails");
const explanationAssetEl = document.getElementById("explanationAsset");
const explanationHorizonEl = document.getElementById("explanationHorizon");
const explanationReconstructionEl = document.getElementById("explanationReconstruction");
const explanationBarsEl = document.getElementById("explanationBars");
let explanationEvidence = null;
const policyStatusEl = document.getElementById("policyStatus");
const policyDetailsEl = document.getElementById("policyDetails");
const policyMethodCardsEl = document.getElementById("policyMethodCards");
const policyAssetEl = document.getElementById("policyAsset");
const policyModelEl = document.getElementById("policyModel");
const policyHorizonEl = document.getElementById("policyHorizon");
const policyWidthEl = document.getElementById("policyWidth");
const policyChartCanvasEl = document.getElementById("policyRiskChart");
let policyEvidence = null;
let policyChartInstance = null;

let historyChart;
let forecastChart;
let activeAsset = "stocks";
let activeChart = "line";
let activeModel = "arima";

let marketEnabled = false; // Fail closed until public-config explicitly permits this.


async function loadPublicMode() {
  try {
    const config = await fetchJson("/api/public-config");
    if (config.marketForecastsValidated !== false ||
        config.historicalResearchIsNotLive !== true ||
        typeof config.marketEnabled !== "boolean") {
      throw new Error("Unrecognized public configuration");
    }
    marketEnabled = config.marketEnabled;
  } catch {
    marketEnabled = false;
  }
  document.body.classList.toggle("market-disabled", !marketEnabled);
  const notice = document.getElementById("marketPreviewNotice");
  notice.hidden = marketEnabled;
  refreshBtn.disabled = !marketEnabled;
  if (!marketEnabled) {
    forecastEvidenceEl.textContent =
      "Live market data and model fitting are disabled on this preview. "
      + "The research sections remain retrospective and are not investment advice.";
    modelValueEl.textContent = "Unavailable";
    dataAsOfValueEl.textContent = "Not connected";
  }
}


function inferDefaultApiBase() {
  const origin = window.location.origin;

  // Local dev on static server uses backend on 8000.
  if (origin.includes(":5500")) {
    return "http://127.0.0.1:8000";
  }

  // Production deployment should run same-origin frontend + API.
  return origin;
}

const DEFAULT_API_BASE = inferDefaultApiBase();

const savedBase = localStorage.getItem("finwise_api_base") || DEFAULT_API_BASE;
apiInput.value = savedBase;

function getBaseUrl() {
  const raw = apiInput.value.trim() || DEFAULT_API_BASE;

  // Accept only absolute http(s) URLs to avoid accidental relative paths.
  const normalizedInput = /^https?:\/\//i.test(raw) ? raw : DEFAULT_API_BASE;

  try {
    const parsed = new URL(normalizedInput);
    const base = `${parsed.protocol}//${parsed.host}`;
    localStorage.setItem("finwise_api_base", base);
    apiInput.value = base;
    return base;
  } catch {
    localStorage.setItem("finwise_api_base", DEFAULT_API_BASE);
    apiInput.value = DEFAULT_API_BASE;
    return DEFAULT_API_BASE;
  }
}

function fmtMoney(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
}

function fmtPct(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function fmtDate(value) {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  }).format(new Date(value));
}

async function fetchJson(path) {
  const res = await fetch(`${getBaseUrl()}${path}`);
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || `Request failed: ${res.status}`);
  }
  return res.json();
}

async function loadCapabilities() {
  // Fail conservatively for older API deployments: never imply validation.
  try {
    const contract = await fetchJson("/api/capabilities");
    if (contract.contractVersion !== "finwise-web-api-v1.1" ||
        contract.forecastReliability?.legacyIntervalsEmpiricallyValidated !== false ||
        contract.researchModelIntegration !== "historical_offline_only_not_in_live_forecast_api") {
      throw new Error("Unexpected API contract version or status");
    }
    capabilityStatusEl.textContent =
      "Live models: legacy statistical forecasts. Ridge/conformal research remains offline.";
  } catch {
    capabilityStatusEl.textContent =
      "API contract unavailable: live forecast calibration is not verified.";
  }
}

async function loadResearchEvidence() {
  // Evidence is optional and HISTORICAL. Fail closed rather than rendering
  // speculative numbers or treating live forecasts as validated.
  researchDetailsEl.hidden = true;
  researchRowsEl.replaceChildren();
  try {
    const data = await fetchJson("/api/research/evidence");
    if (data.schemaVersion !== "finwise-historical-research-evidence-v1" ||
        data.executionMode !== "OFFLINE_AUDITED_EXPORT_NOT_LIVE_PREDICTION" ||
        !Array.isArray(data.results) || data.results.length !== 8) {
      throw new Error("Unexpected historical research evidence contract");
    }
    researchWinsEl.textContent =
      `${data.results.filter((row) => row.ridgeMinusNaiveMae < 0).length}/8`;
    researchCiEl.textContent =
      `${data.results.filter((row) => row.bootstrapContainsZero === true).length}/8`;
    researchCoverageEl.textContent =
      `${data.results.filter((row) => row.ridgeCoverageReady >= 0.9).length}/8`;
    const number = (value) => Number(value).toFixed(3);
    const cells = (row) => [
      row.asset,
      `H${row.horizon}`,
      number(row.naiveMae),
      number(row.ridgeMae),
      number(row.ridgeMinusNaiveMae),
      `[${number(row.bootstrap95Low)}, ${number(row.bootstrap95High)}]`,
      `${(100 * row.ridgeCoverageReady).toFixed(1)}%`,
      `${row.ridgeAccepted}/${row.nOrigins}`,
    ];
    for (const row of data.results) {
      const tr = document.createElement("tr");
      for (const value of cells(row)) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      }
      researchRowsEl.appendChild(tr);
    }
    researchDetailsEl.hidden = false;
    researchStatusEl.textContent =
      "Audited historical summary loaded locally. These results do NOT validate the live forecast.";
  } catch {
    researchStatusEl.textContent =
      "No validated local research export available. Live models remain unvalidated; run the offline W2 exporter to view historical results.";
  }
}

function renderModelExplanations() {
  if (!explanationEvidence) return;
  const selected = explanationEvidence.results.find(
    (row) => row.asset === explanationAssetEl.value &&
             row.horizon === Number(explanationHorizonEl.value)
  );
  explanationBarsEl.replaceChildren();
  if (!selected) {
    explanationReconstructionEl.textContent = "No matching historical explanation.";
    return;
  }
  explanationReconstructionEl.textContent =
    `32 historical origins · max price reconstruction error: ${selected.maxReconstructionError.toExponential(2)} · units: predicted log-return`;
  const maxMagnitude = Math.max(
    1e-12, ...selected.features.map((x) => x.meanAbsLogReturnContribution)
  );
  for (const feature of selected.features) {
    const item = document.createElement("div");
    item.className = "explanation-row";
    const name = document.createElement("div");
    name.textContent = feature.name;
    name.className = "explanation-name";
    const track = document.createElement("div");
    track.className = "explanation-track";
    const fill = document.createElement("div");
    fill.className = "explanation-fill";
    fill.style.width = `${Math.max(0, Math.min(100, 100 * feature.meanAbsLogReturnContribution / maxMagnitude))}%`;
    track.appendChild(fill);
    const value = document.createElement("div");
    value.className = "explanation-value";
    value.textContent = feature.meanAbsLogReturnContribution.toFixed(5);
    item.append(name, track, value);
    explanationBarsEl.appendChild(item);
  }
}

async function loadModelExplanations() {
  explanationDetailsEl.hidden = true;
  explanationEvidence = null;
  explanationBarsEl.replaceChildren();
  try {
    const payload = await fetchJson("/api/research/explanations");
    if (payload.schemaVersion !== "finwise-ridge-explanation-development-v1" ||
        payload.mode !== "HISTORICAL_EXPLORATORY_MODEL_INTERNAL_NOT_LIVE" ||
        payload.attributionType !== "exact_linear_model_decomposition_not_SHAP_not_causal" ||
        !Array.isArray(payload.results) || payload.results.length !== 8) {
      throw new Error("Unrecognized model explanation schema");
    }
    explanationEvidence = payload;
    explanationDetailsEl.hidden = false;
    explanationStatusEl.textContent =
      "Audited M4.1 development explanations loaded. Not derived from live forecasts.";
    renderModelExplanations();
  } catch {
    explanationStatusEl.textContent =
      "No validated M4.1 development export configured. The live ARIMA forecast has no Ridge explanation.";
  }
}

explanationAssetEl.addEventListener("change", renderModelExplanations);
explanationHorizonEl.addEventListener("change", renderModelExplanations);

function renderReliabilityPolicy() {
  if (!policyEvidence) return;
  const asset = policyAssetEl.value;
  const model = policyModelEl.value;
  const horizon = Number(policyHorizonEl.value);
  const threshold = Number(policyWidthEl.value);
  const current = policyEvidence.results.filter((row) =>
    row.asset === asset && row.model === model && row.horizon === horizon
    && Math.abs(row.maxRelativeWidth - threshold) < 1e-9
  );
  policyMethodCardsEl.replaceChildren();
  const label = (method) => method === "fixed_rolling" ? "Fixed rolling" : "Delayed-feedback ACI-style";
  for (const method of ["fixed_rolling", "delayed_feedback_aci_style"]) {
    const item = current.find((r) => r.method === method);
    if (!item) continue;
    const card = document.createElement("article");
    card.className = "policy-method-card";
    const heading = document.createElement("h3");
    heading.textContent = label(method);
    card.appendChild(heading);
    const status = document.createElement("p");
    status.className = "policy-acceptance-status";
    status.textContent = item.nAccepted === 0
      ? "NO FORECAST ACCEPTED — selected risk unavailable"
      : item.nAccepted < 30
        ? "SMALL SAMPLE — descriptive only"
        : "DESCRIPTIVE RESULT — no coverage guarantee";
    card.appendChild(status);
    const grid = document.createElement("dl");
    const entries = [
      ["Accepted / all origins", `${item.nAccepted} / ${item.nOrigins} (${(item.acceptanceAll*100).toFixed(1)}%)`],
      ["Accepted / calibrated", `${item.nAccepted} / ${item.nReady}`],
      ["Coverage (calibrated)", item.intervalCoverageReady == null ? "N/A" : `${(item.intervalCoverageReady*100).toFixed(1)}%`],
      ["Coverage (accepted)", item.intervalCoverageSelected == null ? "N/A" : `${(item.intervalCoverageSelected*100).toFixed(1)}%`],
      ["Selected relative MAE", item.relativeMaeSelected == null ? "N/A" : `${(item.relativeMaeSelected*100).toFixed(2)}%`],
    ];
    for (const [key, value] of entries) {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = key;
      dd.textContent = value;
      grid.append(dt, dd);
    }
    card.appendChild(grid);
    policyMethodCardsEl.appendChild(card);
  }
  const all = policyEvidence.results.filter((r) =>
    r.asset === asset && r.model === model && r.horizon === horizon
  );
  if (policyChartInstance) {
    policyChartInstance.destroy();
    policyChartInstance = null;
  }
  if (typeof Chart === "undefined") {
    return; // Cards remain useful even when third-party chart CDN is unavailable.
  }
  const pairs = [
    ["fixed_rolling", "Fixed rolling", "#2f6f62"],
    ["delayed_feedback_aci_style", "Delayed-feedback ACI-style", "#ef8354"]
  ];
  const datasets = pairs.map(([method, title, color]) => ({
    label: title,
    data: all.filter((row) => row.method === method && row.relativeMaeSelected != null)
      .sort((a, b) => a.maxRelativeWidth - b.maxRelativeWidth)
      .map((row) => ({
        x: row.acceptanceAll * 100,
        y: row.relativeMaeSelected * 100
      })),
    borderColor: color,
    backgroundColor: color,
    showLine: true,
    fill: false,
    pointRadius: 4,
    tension: 0.1
  }));
  policyChartInstance = new Chart(policyChartCanvasEl.getContext("2d"), {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      parsing: false,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${context.parsed.x.toFixed(1)}% accepted; ${context.parsed.y.toFixed(2)}% selected relative MAE`
          }
        }
      },
      scales: {
        x: { title: { display: true, text: "All origins accepted (%)" }, min: 0, max: 100 },
        y: { title: { display: true, text: "Selected relative MAE (%)" }, beginAtZero: true }
      }
    }
  });
}

async function loadReliabilityPolicies() {
  policyDetailsEl.hidden = true;
  policyMethodCardsEl.replaceChildren();
  policyEvidence = null;
  if (policyChartInstance) {
    policyChartInstance.destroy();
    policyChartInstance = null;
  }
  try {
    const data = await fetchJson("/api/research/reliability-policies");
    if (data.schemaVersion !== "finwise-dev-reliability-policy-explorer-v1" ||
        data.executionMode !== "OFFLINE_EXPLORATORY_NOT_LIVE_FORECASTING" ||
        !Array.isArray(data.results) || data.results.length !== 160) {
      throw new Error("Unverified selective policy evidence");
    }
    policyEvidence = data;
    policyStatusEl.textContent =
      "Audited R9 development policy sensitivity loaded locally. This does NOT calibrate live forecasts.";
    policyDetailsEl.hidden = false;
    renderReliabilityPolicy();
  } catch {
    policyStatusEl.textContent =
      "No audited development risk/availability package configured. Live forecast uncertainty remains unvalidated.";
  }
}

for (const select of [policyAssetEl, policyModelEl, policyHorizonEl, policyWidthEl]) {
  select.addEventListener("change", renderReliabilityPolicy);
}

async function loadSymbols() {
  const endpoint = activeAsset === "stocks" ? "/api/stocks" : "/api/crypto/coins";
  const key = activeAsset === "stocks" ? "symbols" : "coins";
  const data = await fetchJson(endpoint);
  symbolSelect.innerHTML = "";

  data[key].forEach((symbol) => {
    const option = document.createElement("option");
    option.value = symbol;
    option.textContent = symbol;
    symbolSelect.appendChild(option);
  });
}

async function loadForecastModels() {
  const data = await fetchJson("/api/forecast-models");
  modelSelect.innerHTML = "";

  data.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label;
    option.title = model.description;
    if (model.id === activeModel) {
      option.selected = true;
    }
    modelSelect.appendChild(option);
  });
}

function setMetricText(element, text, cls = "") {
  element.classList.remove("pos", "neg");
  if (cls) {
    element.classList.add(cls);
  }
  element.textContent = text;
}

function toggleHistoryMode() {
  const showCandle = activeAsset === "stocks" && activeChart === "candlestick";
  historyCanvas.className = showCandle ? "plot-hidden" : "chart-visible";
  candleDiv.className = showCandle ? "plot-visible" : "plot-hidden";
  if (!showCandle) {
    Plotly.purge(candleDiv);
  }
}

function drawCandlestickChart(historyRows, symbol) {
  const dates = historyRows.map((r) => r.date);
  const open = historyRows.map((r) => r.open);
  const high = historyRows.map((r) => r.high);
  const low = historyRows.map((r) => r.low);
  const close = historyRows.map((r) => r.close);

  const valid = open.every((v) => Number.isFinite(v)) &&
    high.every((v) => Number.isFinite(v)) &&
    low.every((v) => Number.isFinite(v));

  if (!valid) {
    activeChart = "line";
    setSwitchActive(chartSwitch, "line", "chart");
    toggleHistoryMode();
    drawHistoryChart(historyRows, symbol);
    return;
  }

  const trace = {
    x: dates,
    open,
    high,
    low,
    close,
    type: "candlestick",
    increasing: { line: { color: "#2e8b57" } },
    decreasing: { line: { color: "#b02a30" } },
  };

  const layout = {
    margin: { l: 40, r: 10, t: 20, b: 35 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    xaxis: { rangeslider: { visible: false } },
    yaxis: { fixedrange: false },
    font: { family: "Space Grotesk, sans-serif" },
  };

  Plotly.newPlot(candleDiv, [trace], layout, { responsive: true, displayModeBar: false });
}

function drawHistoryChart(historyRows, symbol) {
  const ctx = historyCanvas.getContext("2d");
  const labels = historyRows.map((r) => r.date.slice(0, 10));
  const values = historyRows.map((r) => r.close);

  if (historyChart) {
    historyChart.destroy();
  }

  historyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `${symbol} Close`,
          data: values,
          borderColor: "#2f6f62",
          backgroundColor: "rgba(47, 111, 98, 0.12)",
          fill: true,
          tension: 0.2,
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 8 } },
      },
    },
  });
}

function drawForecastChart(historyRows, forecastRows, symbol) {
  const ctx = document.getElementById("forecastChart").getContext("2d");

  const historyTail = historyRows.slice(-50);
  const historyLabels = historyTail.map((r) => r.date.slice(0, 10));
  const historyValues = historyTail.map((r) => r.close);

  const forecastLabels = forecastRows.map((r) => r.date.slice(0, 10));
  const forecastValues = forecastRows.map((r) => r.predictedClose);

  const labels = [...historyLabels, ...forecastLabels];
  const split = historyLabels.length;

  const mergedHistory = [...historyValues, ...new Array(forecastValues.length).fill(null)];
  const mergedForecast = [...new Array(split - 1).fill(null), historyValues[historyValues.length - 1], ...forecastValues];

  if (forecastChart) {
    forecastChart.destroy();
  }

  forecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: `${symbol} Recent`,
          data: mergedHistory,
          borderColor: "#14213d",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: "7-Day Forecast",
          data: mergedForecast,
          borderColor: "#ef8354",
          borderDash: [7, 5],
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 10 } },
      },
    },
  });
}

function renderForecastTable(forecastRows) {
  forecastTableBody.innerHTML = "";

  forecastRows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.date.slice(0, 10)}</td>
      <td>${fmtMoney(row.predictedClose)}</td>
      <td>${fmtMoney(row.lower95)}</td>
      <td>${fmtMoney(row.upper95)}</td>
    `;
    forecastTableBody.appendChild(tr);
  });
}

async function refreshData() {
  if (!marketEnabled) return;
  const selected = symbolSelect.value;
  if (!selected) return;

  refreshBtn.disabled = true;
  refreshBtn.textContent = "Loading...";

  try {
    const basePath = activeAsset === "stocks" ? `/api/stocks/${selected}` : `/api/crypto/${selected}`;
    const [historyData, forecastData] = await Promise.all([
      fetchJson(`${basePath}/history?days=365`),
      fetchJson(`${basePath}/forecast?days=7&model=${encodeURIComponent(activeModel)}`),
    ]);

    setMetricText(latestCloseEl, fmtMoney(historyData.latestClose));

    const cls = historyData.dailyChangePct >= 0 ? "pos" : "neg";
    setMetricText(dailyChangeEl, fmtPct(historyData.dailyChangePct), cls);

    const avgForecast =
      forecastData.forecast.reduce((sum, row) => sum + row.predictedClose, 0) /
      forecastData.forecast.length;
    setMetricText(forecastAvgEl, fmtMoney(avgForecast));
    dataAsOfValueEl.textContent = fmtDate(historyData.history[historyData.history.length - 1].date);
    modelValueEl.textContent = forecastData.model || "-";
    const evidence = forecastData.metadata;
    forecastEvidenceEl.textContent =
      evidence?.intervalEmpiricallyCalibrated === false &&
      evidence?.researchEvidenceAttached === false
        ? "Unvalidated heuristic interval. No 95% coverage guarantee; not investment advice."
        : "Forecast reliability not verified by this API. Do not treat bounds as calibrated.";

    toggleHistoryMode();
    if (activeAsset === "stocks" && activeChart === "candlestick") {
      drawCandlestickChart(historyData.history, selected.toUpperCase());
      if (historyChart) {
        historyChart.destroy();
      }
    } else {
      drawHistoryChart(historyData.history, selected.toUpperCase());
    }

    drawForecastChart(historyData.history, forecastData.forecast, selected.toUpperCase());
    renderForecastTable(forecastData.forecast);
  } catch (err) {
    alert(`Failed to load data. ${err.message}`);
  } finally {
    refreshBtn.disabled = false;
    refreshBtn.textContent = "Refresh";
  }
}

function setSwitchActive(container, value, attribute) {
  const buttons = container.querySelectorAll(".chip");
  buttons.forEach((btn) => {
    const isActive = btn.dataset[attribute] === value;
    btn.classList.toggle("active", isActive);
  });
}

function syncLabels() {
  const isStocks = activeAsset === "stocks";
  assetLabel.textContent = isStocks ? "Stock Symbol" : "Crypto Coin";
  historyTitle.textContent = isStocks
    ? "Historical Price (Last 365 Days)"
    : "Crypto Price (Last 365 Days)";
  forecastTitle.textContent = isStocks
    ? "Next 7 Trading Days Forecast"
    : "Next 7 Days Forecast";

  if (!isStocks && activeChart === "candlestick") {
    activeChart = "line";
    setSwitchActive(chartSwitch, "line", "chart");
  }
}

refreshBtn.addEventListener("click", async () => {
  await refreshData();
});

symbolSelect.addEventListener("change", async () => {
  await refreshData();
});

modelSelect.addEventListener("change", async () => {
  activeModel = modelSelect.value;
  await refreshData();
});

apiInput.addEventListener("change", async () => {
  await loadResearchEvidence();
  await loadModelExplanations();
  await loadReliabilityPolicies();
  await loadCapabilities();
  await loadSymbols();
  await refreshData();
});

assetSwitch.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLButtonElement) || !target.dataset.asset) return;

  activeAsset = target.dataset.asset;
  setSwitchActive(assetSwitch, activeAsset, "asset");
  syncLabels();
  await loadSymbols();
  await refreshData();
});

chartSwitch.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLButtonElement) || !target.dataset.chart) return;

  if (activeAsset !== "stocks" && target.dataset.chart === "candlestick") {
    return;
  }

  activeChart = target.dataset.chart;
  setSwitchActive(chartSwitch, activeChart, "chart");
  await refreshData();
});

(async function init() {
  try {
    todayValueEl.textContent = fmtDate(new Date());
    syncLabels();
    await loadPublicMode();
    await loadCapabilities();
    await loadResearchEvidence();
    await loadModelExplanations();
    await loadReliabilityPolicies();
    if (marketEnabled) {
      await loadForecastModels();
      await loadSymbols();
      await refreshData();
    }
  } catch (err) {
    alert(`Unable to initialize app. ${err.message}`);
  }
})();
