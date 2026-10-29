"""W3 strict optional read-only Ridge development explanation summary."""
from __future__ import annotations
import json
import math
from pathlib import Path

SCHEMA="finwise-ridge-explanation-development-v1"
TYPE="exact_linear_model_decomposition_not_SHAP_not_causal"
ASSETS={"AAPL","MSFT","SPY","TSLA"}


def load_explanations(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError("No audited development attribution summary")
    if path.stat().st_size>128000:
        raise ValueError("Oversized model explanation package")
    data=json.loads(path.read_text())
    if (data.get("schemaVersion")!=SCHEMA or
        data.get("mode")!="HISTORICAL_EXPLORATORY_MODEL_INTERNAL_NOT_LIVE" or
        data.get("attributionType")!=TYPE or
        data.get("contributionUnit")!="predicted_log_return"):
        raise ValueError("Unapproved Ridge explanation schema")
    rows=data.get("results")
    if not isinstance(rows,list) or len(rows)!=8:
        raise ValueError("Expected eight development explanation groups")
    seen=set()
    clean=[]
    for row in rows:
        asset,h=row.get("asset"),row.get("horizon")
        if (asset not in ASSETS or type(h) is not int or h not in (1,7) or
            (asset,h) in seen or type(row.get("nOrigins")) is not int or
            row["nOrigins"]!=32):
            raise ValueError("Unrecognized or duplicate development asset/horizon")
        seen.add((asset,h))
        features=row.get("features")
        if not isinstance(features,list) or len(features)!=6:
            raise ValueError("Expected top six Ridge contributions per group")
        names=set()
        selected=[]
        for x in features:
            name=x.get("name")
            if (not isinstance(name,str) or not name.replace("_","").isalnum()
                or len(name)>80 or name in names):
                raise ValueError("Invalid or duplicate feature name")
            names.add(name)
            numbers=[x.get(k) for k in ("meanAbsLogReturnContribution",
                     "meanSignedLogReturnContribution","fractionPositive")]
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or
                   not math.isfinite(v) for v in numbers):
                raise ValueError("Invalid feature contribution values")
            if numbers[0]<0 or not 0<=numbers[2]<=1:
                raise ValueError("Invalid feature contribution bounds")
            selected.append({
                "name":name,
                "meanAbsLogReturnContribution":numbers[0],
                "meanSignedLogReturnContribution":numbers[1],
                "fractionPositive":numbers[2],
            })
        error=row.get("maxReconstructionError")
        if (not isinstance(error,(int,float)) or not math.isfinite(error)
                or error<0 or error>1e-8):
            raise ValueError("Ridge reconstruction check failed")
        clean.append({"asset":asset,"horizon":h,"nOrigins":32,
                      "features":selected,"maxReconstructionError":error})
    if seen != {(a,h) for a in ASSETS for h in (1,7)}:
        raise ValueError("Missing development explanations")
    return {
        "schemaVersion":SCHEMA,
        "mode":"HISTORICAL_EXPLORATORY_MODEL_INTERNAL_NOT_LIVE",
        "attributionType":TYPE,
        "contributionUnit":"predicted_log_return",
        "warning":"Model-internal additive contributions; NOT SHAP, causal "
                  "attribution, forecasting skill or live investment advice.",
        "results":clean,
    }
