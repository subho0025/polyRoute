"""Connections to the other PolyRoute modules.

Each function here is the ONE place where the dashboard talks to another module.
Until a teammate's module is ready, a clearly marked stand-in is used instead,
so the dashboard always runs. Swap the stand-in for the real call when it's ready.

    classifier  -> load_items_from_*      (stand-in: synthetic/uploaded JSON)
    optimiser   -> run_optimiser()        (stand-in: simple greedy, placeholder factors)
    explainer   -> explain_plan()         (stand-in: text template)
"""

import json

from comp_analyzer.config import SYNTHETIC_DIR

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

# STAND-IN emission factors, kg CO2e per kg (None = route not available).
# PLACEHOLDERS, NOT WARM VALUES. The real optimiser uses the sourced table.
_STANDIN_FACTORS = {
    "PET":   {"recycle": -1.00, "compost": None, "combust": 1.20, "landfill": 0.02},
    "HDPE":  {"recycle": -0.90, "compost": None, "combust": 1.30, "landfill": 0.02},
    "PVC":   {"recycle": None,  "compost": None, "combust": 0.70, "landfill": 0.02},
    "LDPE":  {"recycle": None,  "compost": None, "combust": 1.30, "landfill": 0.02},
    "PP":    {"recycle": -0.80, "compost": None, "combust": 1.30, "landfill": 0.02},
    "PS":    {"recycle": None,  "compost": None, "combust": 1.70, "landfill": 0.02},
    "Other": {"recycle": -0.50, "compost": None, "combust": 1.10, "landfill": 0.02},
}


def _standin_optimiser(composition_data, limits):
    """Greedy stand-in: fill the lowest-carbon routes first, within capacity.
    Not optimal in general – only here so the dashboard runs before the real optimiser."""
    caps = {"recycle": limits["recycle_cap_kg"], "compost": limits["compost_cap_kg"],
            "landfill": limits["landfill_cap_kg"], "combust": float("inf")}
    left = dict(composition_data["kg"])
    plan = {p: {} for p in left}
    options = sorted((f, p, r) for p in left for r, f in _STANDIN_FACTORS.get(p, {}).items() if f is not None)
    for factor, p, r in options:
        amount = min(left[p], caps[r])
        if amount > 0:
            plan[p][r] = round(plan[p].get(r, 0) + amount, 2)
            left[p] -= amount
            caps[r] -= amount

    optimised = sum(kg * _STANDIN_FACTORS[p][r] for p, routes in plan.items() for r, kg in routes.items())
    baseline = sum(kg * _STANDIN_FACTORS[p]["landfill"] for p, kg in composition_data["kg"].items())
    no_recycling = [p for p in plan if _STANDIN_FACTORS[p]["recycle"] is None]
    notes = ([f"{_join(no_recycling)} {'has' if len(no_recycling) == 1 else 'have'} no recycling route, "
              f"so {'it takes' if len(no_recycling) == 1 else 'they take'} the next-best option."]
             if no_recycling else [])
    return {"plan": plan, "optimised_kgco2e": round(optimised, 2),
            "baseline_kgco2e": round(baseline, 2), "saved_kgco2e": round(baseline - optimised, 2),
            "notes": notes, "is_standin": True}


def run_optimiser(composition_data, limits):
    """TODO (optimiser teammate): replace with the real optimiser call, e.g.
        from optimiser import optimise
        return optimise(composition_data, limits)
    Expected return: {"plan": {polymer: {route: kg}}, "optimised_kgco2e", "baseline_kgco2e",
                      "saved_kgco2e", "notes": [...]}"""
    return _standin_optimiser(composition_data, limits)


# ---------- explainer ----------

ROUTE_WORDS = {"recycle": "recycling", "compost": "composting", "combust": "energy recovery",
               "landfill": "landfill"}


def _join(names):
    """['PET'] -> 'PET'; ['PET', 'HDPE', 'PP'] -> 'PET, HDPE and PP'."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def explain_plan(plan_result):
    """TODO (explainer teammate): replace with the Granite call.
    Stand-in: a short explanation filled in from the numbers."""
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
               if saved > 0 else f"adds {-saved:.1f} kg CO₂e compared with landfill, because capacity limits force combustion")
    return f"The plan sends {'; '.join(parts)}. Overall it {verdict}."
