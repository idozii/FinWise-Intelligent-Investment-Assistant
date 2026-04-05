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

let historyChart;
let forecastChart;
let activeAsset = "stocks";
let activeChart = "line";
let activeModel = "arima";

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
    await loadForecastModels();
    await loadSymbols();
    await refreshData();
  } catch (err) {
    alert(`Unable to initialize app. ${err.message}`);
  }
})();
