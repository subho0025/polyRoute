"""Connections to the other PolyRoute modules.

Each function here is the ONE place where the dashboard talks to another module.
Until a teammate's module is ready, a clearly marked stand-in is used instead,
so the dashboard always runs. Swap the stand-in for the real call when it's ready.

    classifier  -> load_items_from_*      (stand-in: synthetic/uploaded JSON)
    optimiser   -> run_optimiser()        (real: optimiser.optimize_routes + EPA WARM factors)
    explainer   -> explain_plan()         (real: IBM Granite via explainer, template if no model)
"""

import json

from comp_analyzer.config import SYNTHETIC_DIR
import os
from pathlib import Path

from explainer import explainer as _explainer
from explainer.explainer import build_granite_prompt, generate_explanation, generate_fallback
from optimiser.factors import load_factors
from optimiser.route_opt import optimize_routes

ROUTES = ["recycle", "compost", "combust", "landfill"]


# ---------- classifier ----------

def list_sample_files():
    """Synthetic classifier outputs available as demo inputs."""
    if not SYNTHETIC_DIR.exists():
        return []
    return sorted(p.name for p in SYNTHETIC_DIR.glob("*.json"))


def load_items_from_sample(name):
    path = SYNTHETIC_DIR / name
    if not path.exists() or path.parent != SYNTHETIC_DIR:
        raise ValueError(f"Sample file {name!r} not found. Run: python -m comp_analyzer.gen")
    return json.loads(path.read_text())


def load_items_from_upload(file_storage):
    """A classifier output JSON uploaded through the form."""
    try:
        items = json.loads(file_storage.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"The uploaded file isn't valid JSON: {e}") from e
    if not isinstance(items, list):
        raise ValueError("The uploaded file must contain a JSON list of classifier records.")
    return items


def classify_photos(files):
    """TODO (classifier teammate): photos -> list of classifier records.
    Replace the body with the real classifier call."""
    raise NotImplementedError("Photo classification isn't connected yet.")


# ---------- optimiser ----------

FACTORS = load_factors()  # EPA WARM v16, loaded once when the app starts
FACTORS_SOURCE = "US EPA WARM v16 (Dec 2023)"
CAP_KEYS = {"recycle": "recycle_cap_kg", "compost": "compost_cap_kg", "landfill": "landfill_cap_kg"}
ROUTE_WORDS = {"recycle": "recycling", "compost": "composting", "combust": "energy recovery",
               "landfill": "landfill"}


def _f(polymer, route):
    v = FACTORS.get(polymer, {}).get(route, "NA")
    return None if v == "NA" else v


def run_optimiser(composition_data, limits):
    """Calls optimiser/route_opt.py unchanged, then adds what the page needs:
    optimised and baseline CO2e, and plain-language notes."""
    try:
        result = optimize_routes(composition_data, limits, FACTORS)
    except Exception as e:  # e.g. PulpSolverError when the GLPK solver (glpsol) isn't installed
        raise ValueError(f"The route optimiser couldn't run: {e}") from e
    if "error" in result:
        raise ValueError(result["error"])

    kg = composition_data.get("kg", {})
    # same baseline rule as route_opt: everything to landfill (energy recovery if no landfill factor)
    baseline = sum(v * (_f(p, "landfill") if _f(p, "landfill") is not None else (_f(p, "combust") or 0))
                   for p, v in kg.items())
    notes = [n for n in result.get("notes", []) if n != "No plastic detected"]
    notes += _notes(kg, result["plan"], limits)
    return {**result, "baseline_kgco2e": round(baseline, 2),
            "optimised_kgco2e": round(baseline - result["saved_kgco2e"], 2), "notes": notes}


def _notes(kg, plan, limits):
    """Why the plan looks the way it does (shown on the page and sent to Granite)."""
    notes = []
    no_recycling = [p for p in kg if _f(p, "recycle") is None]
    if no_recycling:
        verb = "has" if len(no_recycling) == 1 else "have"
        notes.append(f"{_join(no_recycling)} {verb} no recycling factor in EPA WARM, so the plan uses "
                     "landfill where there is space and energy recovery after that.")
    for r, key in CAP_KEYS.items():
        cap = limits.get(key)
        if cap is None:
            continue
        used = sum(routes.get(r, 0) for routes in plan.values())
        pushed = {}  # kg that would have been better on route r but went elsewhere
        for p, routes in plan.items():
            for r2, amount in routes.items():
                if r2 != r and _f(p, r) is not None and _f(p, r) < _f(p, r2):
                    pushed[r2] = pushed.get(r2, 0) + amount
        if used >= cap - 0.01 and pushed:
            notes.append(f"{ROUTE_WORDS[r].capitalize()} is full at {cap:g} kg, so "
                         f"{sum(pushed.values()):.2f} kg goes to {_join([ROUTE_WORDS[x] for x in pushed])} instead.")
    return notes


# ---------- explainer ----------

# The Granite model file (large: *.gguf is in .gitignore). Override with POLYROUTE_GRANITE_GGUF.
GRANITE_MODEL = os.environ.get(
    "POLYROUTE_GRANITE_GGUF",
    str(Path(__file__).resolve().parent.parent / "explainer" / "granite-4.0-h-instruct.gguf"))


def _join(names):
    """['PET'] -> 'PET'; ['PET', 'HDPE', 'PP'] -> 'PET, HDPE and PP'."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _granite_ready():
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(GRANITE_MODEL)


# Works with the original explainer and the improved one (chat format, model loaded once, preload).
if _granite_ready() and hasattr(_explainer, "warm_up"):
    _explainer.warm_up(GRANITE_MODEL)  # load Granite in the background as the app starts


def _template_explanation(plan_result):
    """Used when Granite isn't available: a short explanation filled in from the numbers."""
    plan = plan_result["plan"]
    if not any(plan.values()):
        return "No plastic was found in this batch, so there is nothing to route."
    by_route = {}
    for polymer, routes in plan.items():
        if routes:
            by_route.setdefault(max(routes, key=routes.get), []).append(polymer)
    parts = [f"{_join(ps)} mainly to {ROUTE_WORDS[r]}" for r, ps in by_route.items()]
    saved = plan_result["saved_kgco2e"]
    verdict = (f"saves {saved:.1f} kg CO₂e compared with sending everything to landfill"
               if saved > 0 else f"adds {-saved:.1f} kg CO₂e compared with landfill, because capacity limits force energy recovery")
    return f"The plan sends {'; '.join(parts)}. Overall it {verdict}."


def explain_plan(composition_data, plan_result, batch_name):
    """IBM Granite via explainer/explainer.py when the model is available, else a template.
    Returns (text, source, seconds) with source "granite" or "template"."""
    if plan_result["plan"] and _granite_ready():
        if hasattr(_explainer, "build_granite_messages"):
            prompt = _explainer.build_granite_messages(batch_name, composition_data, plan_result)
        else:
            prompt = build_granite_prompt(batch_name, composition_data, plan_result)
        text, seconds = generate_explanation(prompt, plan_result, model_path=GRANITE_MODEL)
        fallback = generate_fallback(plan_result["saved_kgco2e"], plan_result["plan"], plan_result["notes"])
        if text and text != fallback:
            return text, "granite", seconds
    return _template_explanation(plan_result), "template", 0.0