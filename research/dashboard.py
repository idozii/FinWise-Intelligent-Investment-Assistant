"""M4: build an offline research dashboard from actual M2.1/M3 CSV outputs.

No internet, no brokerage, no trade recommendations. The dashboard does NOT
invent model performance or claim SHAP/causal explanations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL = "finwise-offline-research-dashboard-v1"


def _read(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    return frame


def build_payload(robustness_dir: Path, uncertainty_dir: Path) -> dict:
    p = robustness_dir / "asset_predictions.csv"
    c = robustness_dir / "paired_comparisons.csv"
    r = robustness_dir / "regime_metrics.csv"
    i = uncertainty_dir / "interval_metrics.csv"

    predictions = _read(p, {"asset", "model", "horizon_step", "origin_index",
                            "actual", "predicted", "train_end"})
    comparisons = _read(c, {"asset", "model", "horizon_step",
                            "mae_delta_model_minus_naive",
                            "bootstrap_ci_low", "bootstrap_ci_high", "n_origins"})
    regimes = _read(r, {"asset", "regime", "model", "horizon_step",
                        "mae", "n_origins"})
    uncertainty = _read(i, {"asset", "model", "horizon_step",
                            "empirical_coverage_ready", "n_calibrated",
                            "acceptance_rate_all_origins", "coverage_accepted",
                            "mean_relative_width_ready", "mae_accepted"})
    for name, frame in (("paired", comparisons), ("regimes", regimes),
                        ("uncertainty", uncertainty)):
        unknown = set(frame.asset.astype(str)) - set(predictions.asset.astype(str))
        if unknown:
            raise ValueError(f"{name} has assets absent from predictions: {unknown}")
    metrics = []
    for (asset, model, horizon), frame in predictions.groupby(
        ["asset", "model", "horizon_step"], sort=True
    ):
        errors = frame.actual.to_numpy(float) - frame.predicted.to_numpy(float)
        if not np.isfinite(errors).all():
            raise ValueError("Prediction residuals must be finite")
        metrics.append({
            "asset": str(asset), "model": str(model), "horizon": int(horizon),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "n_origins": int(frame.origin_index.nunique()),
        })
    def clean_records(frame: pd.DataFrame) -> list[dict]:
        # JSON-compatible: NaN/inf becomes null, not nonstandard JSON constants.
        data = frame.replace([np.inf, -np.inf], np.nan)
        return json.loads(data.to_json(orient="records", date_format="iso"))

    assets = sorted(set(str(a) for a in predictions.asset.unique()))
    return {
        "protocol": PROTOCOL,
        "stage": "EXPLORATORY_NOT_INDEPENDENT_HOLDOUT",
        "assets": assets,
        "horizons": sorted(int(h) for h in predictions.horizon_step.unique()),
        "model_metrics": metrics,
        "paired": clean_records(comparisons),
        "regimes": clean_records(regimes),
        "uncertainty": clean_records(uncertainty),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (p, c, r, i)
        },
        "disclaimer": "Offline retrospective research; no verified trading edge, "
                      "forecast coverage guarantee, or personalized investment advice.",
        "method_note": "Regimes reflect only past returns. Paired bootstrap CIs "
                       "are exploratory. Coverage is empirical, conditional on matured calibration.",
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FinWise | Research Transparency Dashboard</title>
<style>
:root{color-scheme:dark;--bg:#08121d;--panel:#102337;--panel2:#152d43;
 --text:#ebf4f7;--muted:#a9c0d0;--line:#2e5065;--teal:#5de1cd;--amber:#ffd18b}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0%,#163950,#08121d 53%);
 color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,Arial,sans-serif}
main{max-width:1200px;padding:34px 20px 80px;margin:auto}
h1{font-size:clamp(30px,4vw,52px);line-height:1.1;letter-spacing:-.035em;margin:6px 0 15px}
h2{font-size:20px;margin:0 0 14px}h3{font-size:15px;color:var(--muted);margin:0 0 8px}
p{margin:6px 0 14px}.eyebrow{text-transform:uppercase;letter-spacing:.17em;color:var(--teal);
 font-weight:750;font-size:12px}.lead{color:var(--muted);font-size:17px;max-width:720px}
.meta{color:var(--muted);font-size:12px} .warning{border-left:3px solid var(--amber);
 background:#302619;padding:12px 16px;color:#ffdfaf;border-radius:4px;margin:25px 0}
.controls,.metrics,.panels{display:grid;gap:14px}
.controls{grid-template-columns:repeat(2,minmax(0,1fr));margin:23px 0}
.control,section.panel,.metric{border:1px solid var(--line);border-radius:16px;background:var(--panel);
 padding:18px 19px}
label{font-size:12px;color:var(--muted);display:block;margin:0 0 9px}
select{appearance:none;color:var(--text);background:var(--panel2);border:1px solid #47708b;
 font:inherit;width:100%;border-radius:9px;padding:11px 12px}
.metrics{grid-template-columns:repeat(3,minmax(0,1fr));margin:12px 0 17px}
.metric .number{font-size:clamp(20px,3vw,32px);font-weight:780;letter-spacing:-.02em}
.metric p{margin:0;color:var(--muted);font-size:12px}
.panels{grid-template-columns:1.15fr .85fr;margin-bottom:16px}
section.panel{overflow-x:auto}.bars{display:grid;gap:14px}
.barrow{display:grid;grid-template-columns:102px 1fr 75px;align-items:center;gap:10px}
.track{background:#243e51;height:12px;border-radius:20px;overflow:hidden}
.fill{height:100%;border-radius:20px;background:var(--teal)}.barrow:first-child .fill{background:#ffe09a}
.value{text-align:right;font-variant-numeric:tabular-nums}table{border-collapse:collapse;width:100%;
 min-width:390px;text-align:left}th{font-size:11px;letter-spacing:.05em;color:var(--muted);text-transform:uppercase}
th,td{padding:11px 10px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
td{font-size:13px}small{color:var(--muted)}.pill{color:var(--teal);font-weight:700}
footer{border-top:1px solid var(--line);margin-top:28px;padding-top:18px;color:var(--muted);
 font-size:12px}#sha{word-break:break-word}
@media(max-width:750px){.metrics,.panels,.controls{grid-template-columns:1fr}
 .barrow{grid-template-columns:82px 1fr 65px}main{padding-top:24px}}
</style>
</head>
<body><main>
<div class="eyebrow">FinWise / Independent research prototype</div>
<h1>Forecasts need evidence.<br>Not just confidence.</h1>
<p class="lead">Explore out-of-sample-style walk-forward errors, historical
market regimes, and the rate at which a model abstains when its forecast
uncertainty is too wide.</p>
<div class="warning"><strong>Exploratory results — not investment advice.</strong>
The tested periods and models are not an untouched independent holdout.
Bootstrap CIs and retrospective interval coverage are not guarantees.</div>
<div class="controls">
  <div class="control"><label for="asset">Asset</label><select id="asset"></select></div>
  <div class="control"><label for="horizon">Forecast horizon (observations)</label>
    <select id="horizon"></select></div>
</div>
<div class="metrics">
  <article class="metric"><h3>Naive baseline MAE</h3><div class="number" id="naive-mae">—</div>
  <p>Last observed closing price</p></article>
  <article class="metric"><h3>Best observed MAE</h3><div class="number" id="best-mae">—</div>
  <p id="best-name">Compared on the same horizons</p></article>
  <article class="metric"><h3>Evaluation origins</h3><div class="number" id="origins">—</div>
  <p>Historical expanding-window origins</p></article>
</div>
<div class="panels">
  <section class="panel"><h2>Model errors</h2>
  <p class="meta">MAE (USD price units). Lower is better. Values are retrospective.</p>
  <div class="bars" id="bars"></div></section>
  <section class="panel"><h2>Paired comparison vs naive</h2>
  <p class="meta">MAE difference and exploratory 95% block-bootstrap CI.
  Positive = worse than naive.</p><div id="comparisons"></div></section>
</div>
<div class="panels">
  <section class="panel"><h2>Forecast uncertainty</h2>
  <p class="meta">Coverage is empirical on origins with enough matured
  calibration errors. Acceptance is based on interval width, not future outcomes.</p>
  <div id="uncertainty"></div></section>
  <section class="panel"><h2>Performance by market regime</h2>
  <p class="meta">Regimes are defined using volatility measured before each origin.
  This is descriptive, not a causal explanation.</p>
  <div id="regimes"></div></section>
</div>
<section class="panel"><h2>What the models observe</h2>
<p>The research pipeline uses historical log returns (1, 3, 5, 10, 20, 30 observations),
realized volatility (5, 20, 30), price distance from 5/20-observation averages,
and the fraction of positive recent returns.</p>
<p class="meta">These are input features, <strong>not</strong> SHAP importance or
causal attributions. Model attribution, calibrated conditional coverage,
trading costs and independent confirmation remain future work.</p></section>
<footer><p><strong>Data integrity:</strong> SHA-256 fingerprints of local evaluation
artifacts (no market data uploaded to a hosted service).</p><div id="sha"></div>
<p>FinWise research dashboard · Offline artifact · No model or trading
recommendations are generated here.</p></footer>
</main>
<script id="payload" type="application/json">__FINWISE_DATA__</script>
<script>
"use strict";
const dataset = JSON.parse(document.getElementById("payload").textContent);
const assetSelect = document.getElementById("asset");
const horizonSelect = document.getElementById("horizon");
function text(id, value){document.getElementById(id).textContent = String(value)}
function number(v, digits=3){return v === null || v === undefined || !Number.isFinite(Number(v))
  ? "N/A" : Number(v).toFixed(digits)}
function pct(v){return v === null || v === undefined ? "N/A" : (Number(v)*100).toFixed(1)+"%"}
function add(tag, label, parent, cls){let e=document.createElement(tag);
  if(cls)e.className=cls;e.textContent=String(label);parent.appendChild(e);return e}
function clear(node){while(node.firstChild)node.removeChild(node.firstChild)}
function table(container, cols, rows){
  clear(container);
  if(!rows.length){add("p","No results for this selection",container,"meta");return}
  const tab=add("table","",container);
  const head=add("thead","",tab), tr=add("tr","",head);
  cols.forEach(c=>add("th",c[0],tr));
  const body=add("tbody","",tab);
  rows.forEach(row=>{
    const r=add("tr","",body);
    cols.forEach(c=>add("td",c[1](row),r));
  });
}
function redraw(){
  const a=assetSelect.value,h=Number(horizonSelect.value);
  const metrics=dataset.model_metrics.filter(x=>x.asset===a&&x.horizon===h);
  const baseline=metrics.find(x=>x.model==="naive");
  if(!baseline){text("naive-mae","N/A");return}
  const ranked=[...metrics].sort((x,y)=>x.mae-y.mae);
  text("naive-mae",number(baseline.mae));
  text("best-mae",number(ranked[0].mae));
  text("best-name",ranked[0].model+" (retrospective, not statistically validated)");
  text("origins",baseline.n_origins);
  const bars=document.getElementById("bars");clear(bars);
  const max=Math.max(...metrics.map(x=>x.mae),0.0001);
  ranked.forEach(row=>{
    const line=add("div","",bars,"barrow");
    add("span",row.model,line);
    const track=add("div","",line,"track");
    const bar=add("div","",track,"fill");bar.style.width=(100*row.mae/max).toFixed(1)+"%";
    add("span",number(row.mae),line,"value");
  });
  const comparisons=dataset.paired.filter(x=>x.asset===a&&x.horizon_step===h);
  table(document.getElementById("comparisons"),[
    ["Model",x=>x.model],["Δ MAE",x=>number(x.mae_delta_model_minus_naive)],
    ["95% CI",x=>"["+number(x.bootstrap_ci_low)+", "+number(x.bootstrap_ci_high)+"]"]
  ],comparisons);
  const uncertainty=dataset.uncertainty.filter(x=>x.asset===a&&x.horizon_step===h);
  table(document.getElementById("uncertainty"),[
    ["Model",x=>x.model],
    ["Coverage",x=>pct(x.empirical_coverage_ready)],
    ["Accepted",x=>pct(x.acceptance_rate_all_origins)],
    ["Calibrated",x=>x.n_calibrated]
  ],uncertainty);
  const regimes=dataset.regimes.filter(x=>x.asset===a&&x.horizon_step===h);
  table(document.getElementById("regimes"),[
    ["Regime",x=>x.regime],["Model",x=>x.model],
    ["MAE",x=>number(x.mae)],["Origins",x=>x.n_origins]
  ],regimes);
}
dataset.assets.forEach(a=>{const o=document.createElement("option");o.value=a;o.textContent=a;assetSelect.appendChild(o)});
dataset.horizons.forEach(h=>{const o=document.createElement("option");o.value=h;
 o.textContent=h+" observations";horizonSelect.appendChild(o)});
assetSelect.addEventListener("change",redraw);
horizonSelect.addEventListener("change",redraw);
const fingerprints=document.getElementById("sha");
Object.entries(dataset.source_sha256).forEach(([name,sha])=>{
  add("p",name+": "+sha,fingerprints,"meta");
});
if(dataset.assets.length && dataset.horizons.length)redraw();
</script></body></html>
"""


def generate(robustness_dir: Path, uncertainty_dir: Path, output: Path) -> dict:
    payload = build_payload(robustness_dir, uncertainty_dir)
    serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    # JSON string stays intact, but cannot terminate the embedding <script>.
    serialized = (serialized.replace("&", "\\u0026").replace("<", "\\u003c")
                  .replace(">", "\\u003e").replace("\u2028", "\\u2028")
                  .replace("\u2029", "\\u2029"))
    html = HTML.replace("__FINWISE_DATA__", serialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robustness-dir", type=Path, required=True)
    parser.add_argument("--uncertainty-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = generate(args.robustness_dir, args.uncertainty_dir, args.output)
    print(f"Built offline dashboard for {len(payload['assets'])} assets: {args.output}")


if __name__ == "__main__":
    main()
