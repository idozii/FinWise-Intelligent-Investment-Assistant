"use strict";

// This page never calls the live forecasting endpoint. It displays only
// independently audited, offline-generated research aggregate packages.
const summaryStatus = document.getElementById("evidenceStatus");
const summaryRoot = document.getElementById("evidenceData");
const summaryStats = document.getElementById("evidenceStats");
const summaryRows = document.getElementById("evidenceRows");
const policyStatus = document.getElementById("policyStatus");

function element(tag, cls, label) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (label !== undefined) el.textContent = label;
  return el;
}
async function readJSON(endpoint) {
  const response = await fetch(endpoint, {
    headers: { Accept: "application/json" },
    cache: "no-store"
  });
  if (!response.ok) throw new Error("Research evidence unavailable");
  return response.json();
}
function isFiniteNum(value) {
  return typeof value === "number" && Number.isFinite(value);
}

async function showM5Evidence() {
  summaryRoot.hidden = true;
  summaryStats.replaceChildren();
  summaryRows.replaceChildren();
  try {
    const data = await readJSON("/api/research/evidence");
    if (data.schemaVersion !== "finwise-historical-research-evidence-v1" ||
        data.executionMode !== "OFFLINE_AUDITED_EXPORT_NOT_LIVE_PREDICTION" ||
        !Array.isArray(data.results) || data.results.length !== 8) {
      throw new Error("Wrong evidence type");
    }
    const expected = new Set(["GLD:1","GLD:7","IWM:1","IWM:7",
                              "XLE:1","XLE:7","XLV:1","XLV:7"]);
    for (const r of data.results) {
      const key = r.asset + ":" + r.horizon;
      if (!expected.delete(key) || ![r.naiveMae, r.ridgeMae,
           r.ridgeMinusNaiveMae, r.bootstrap95Low, r.bootstrap95High,
           r.ridgeCoverageReady].every(isFiniteNum) ||
           !Number.isInteger(r.nOrigins) || r.nOrigins !== 64 ||
           !Number.isInteger(r.ridgeAccepted) || r.ridgeAccepted < 0 ||
           r.ridgeAccepted > 64) {
        throw new Error("Malformed first-run research summary");
      }
    }
    if (expected.size !== 0) throw new Error("Incomplete historical asset groups");
    const stats = [
      ["Ridge MAE wins", data.results.filter(r => r.ridgeMinusNaiveMae < 0).length + "/8",
       "All-origin, historical comparison"],
      ["Descriptive CIs include zero", data.results.filter(r => r.bootstrapContainsZero === true).length + "/8",
       "Not adjusted for multiple testing"],
      ["Ridge ready coverage ≥90%", data.results.filter(r => r.ridgeCoverageReady >= .9).length + "/8",
       "Nominal target, not guaranteed"]
    ];
    for (const [name, value, caption] of stats) {
      const card = element("article","stat");
      card.append(
        element("span", "", name),
        element("strong", "", value),
        element("small", "", caption)
      );
      summaryStats.appendChild(card);
    }
    for (const r of data.results) {
      const cells = [
        r.asset, "H" + r.horizon, r.naiveMae.toFixed(3),
        r.ridgeMae.toFixed(3),
        (r.ridgeMinusNaiveMae >= 0 ? "+" : "") + r.ridgeMinusNaiveMae.toFixed(3),
        "[" + r.bootstrap95Low.toFixed(3) + ", " + r.bootstrap95High.toFixed(3) + "]",
        r.ridgeAccepted + "/64"
      ];
      const tr = element("tr");
      for (const cell of cells) tr.appendChild(element("td", "", cell));
      summaryRows.appendChild(tr);
    }
    summaryRoot.hidden = false;
    summaryStatus.textContent = "Verified local historical evidence loaded. The data is not a live model performance claim.";
  } catch {
    summaryStatus.textContent =
      "Audited historical evidence is not configured on this deployment. The study remains documented in the repository; no performance numbers are fabricated here.";
  }
}

async function showDevelopmentPolicy() {
  try {
    const data = await readJSON("/api/research/reliability-policies");
    if (data.schemaVersion !== "finwise-dev-reliability-policy-explorer-v1" ||
        data.executionMode !== "OFFLINE_EXPLORATORY_NOT_LIVE_FORECASTING" ||
        !Array.isArray(data.results) || data.results.length !== 160) {
      throw new Error("Unexpected R9 evidence type");
    }
    const rows = data.results.filter(r =>
      r.asset === "MSFT" && r.model === "ridge" && r.horizon === 7 &&
      Math.abs(r.maxRelativeWidth - .1) < 1e-10
    );
    const fixed = rows.find(r => r.method === "fixed_rolling");
    const adaptive = rows.find(r => r.method === "delayed_feedback_aci_style");
    if (!fixed || !adaptive ||
        ![fixed.nAccepted, adaptive.nAccepted].every(Number.isInteger) ||
        fixed.nOrigins !== 143 || adaptive.nOrigins !== 143 ||
        fixed.nAccepted < 0 || adaptive.nAccepted < 0 ||
        fixed.nAccepted > 143 || adaptive.nAccepted > 143) {
      throw new Error("Invalid development policy summary");
    }
    document.getElementById("fixedAccepted").textContent =
      fixed.nAccepted + " / " + fixed.nOrigins;
    document.getElementById("adaptiveAccepted").textContent =
      adaptive.nAccepted + " / " + adaptive.nOrigins;
    policyStatus.textContent =
      "R9 historical development-only policy evidence loaded, not validated for live forecasts.";
  } catch {
    policyStatus.textContent =
      "Development policy summary not configured. No selective-risk figures are asserted for this deployment.";
  }
}

void showM5Evidence();
void showDevelopmentPolicy();
