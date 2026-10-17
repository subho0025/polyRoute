"""Runs the whole PolyRoute chain for one request and prepares what the page shows.

form input -> classifier records -> comp_analyzer -> optimiser -> explainer -> view model
"""

import platform

from comp_analyzer import (POLYMERS, estimate_composition, load_confusion,
                           load_item_mass, load_weight_params, to_optimiser_input)

from . import integrations
from .integrations import ROUTES

# Reference data: loaded once when the app starts
MASSES, MASSES_PLACEHOLDER = load_item_mass()
CONFUSION, CONFUSION_SYNTHETIC = load_confusion()
WEIGHT_PARAMS, WEIGHT_PLACEHOLDER = load_weight_params()

RESIN_CODE = {"PET": 1, "HDPE": 2, "PVC": 3, "LDPE": 4, "PP": 5, "PS": 6, "Other": 7}
POLYMER_NAMES = {
    "PET": "Polyethylene terephthalate", "HDPE": "High-density polyethylene",
    "PVC": "Polyvinyl chloride", "LDPE": "Low-density polyethylene",
    "PP": "Polypropylene", "PS": "Polystyrene", "Other": "Multilayer, PLA and others",
}
ROUTE_LABELS = {"recycle": "Recycling", "compost": "Composting", "combust": "Energy recovery",
                "landfill": "Landfill"}

DEFAULT_LIMITS = {"recycle_cap_kg": 40.0, "compost_cap_kg": 10.0, "landfill_cap_kg": 15.0}


def _number(form, key, default=None):
    """Read a positive number from the form, or return default if left empty."""
    raw = (form.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"'{raw}' isn't a number. Enter a value like 50 or 1.2.")
    if value < 0:
        raise ValueError("Values can't be negative.")
    return value


def read_inputs(form, files):
    """Turn the submitted form into (items, total_kg, batch, limits, source_label)."""
    upload = files.get("classifier_file")
    if upload and upload.filename:
        items, source = integrations.load_items_from_upload(upload), upload.filename
    else:
        sample = form.get("sample_file") or "medium_100.json"
        items, source = integrations.load_items_from_sample(sample), sample

    mode = form.get("weight_mode", "measured")
    total_kg, batch = None, None
    if mode == "measured":
        total_kg = _number(form, "total_kg")
        if not total_kg:
            raise ValueError("Enter the batch weight in kg, or choose another way to estimate it.")
    elif mode == "volume":
        volume = _number(form, "volume_m3")
        if not volume:
            raise ValueError("Enter the batch volume in m³.")
        batch = {"volume_m3": volume}
    elif mode == "count":
        count = _number(form, "item_count")
        if not count:
            raise ValueError("Enter the number of items in the batch.")
        batch = {"item_count": int(count)}

    limits = {k: _number(form, k, v) for k, v in DEFAULT_LIMITS.items()}
    return items, total_kg, batch, limits, source


def run(form, files):
    """Full chain for one request. Returns the view model for the template."""
    items, total_kg, batch, limits, source = read_inputs(form, files)

    result = estimate_composition(items, total_kg=total_kg, batch=batch, masses=MASSES,
                                  confusion=CONFUSION, weight_params=WEIGHT_PARAMS)
    composition_data = to_optimiser_input(result)["composition_data"]
    plan_result = integrations.run_optimiser(composition_data, limits)
    explanation = integrations.explain_plan(composition_data, plan_result, source)

    return build_view(result, plan_result, explanation, limits, source, len(items))


def build_view(result, plan_result, explanation, limits, source, n_items):
    """explanation = (text, source, seconds) from integrations.explain_plan."""
    """Everything the template needs, pre-computed (widths, labels, signs)."""
    kg = result["kg"]
    biggest = max(kg.values(), default=0) or 1
    lanes = []
    for p in POLYMERS:
        if p not in kg:
            continue
        routes = plan_result["plan"].get(p, {})
        segments = [{"route": r, "label": ROUTE_LABELS[r], "kg": routes[r],
                     "pct": 100 * routes[r] / kg[p] if kg[p] else 0}
                    for r in ROUTES if routes.get(r, 0) > 0.005]
        low, high = result["kg_ci"].get(p, [kg[p], kg[p]])
        lanes.append({"code": RESIN_CODE[p], "polymer": p, "name": POLYMER_NAMES[p],
                      "kg": kg[p], "low": low, "high": high,
                      "width": 100 * kg[p] / biggest, "segments": segments,
                      "items": result["item_counts"].get(p, 0)})

    base, opt = plan_result["baseline_kgco2e"], plan_result["optimised_kgco2e"]
    scale = max(abs(base), abs(opt), 0.01)
    carbon = [
        {"label": "Everything to landfill", "value": base, "width": 50 * abs(base) / scale},
        {"label": "PolyRoute plan", "value": opt, "width": 50 * abs(opt) / scale},
    ]

    placeholders = [name for flag, name in [
        (MASSES_PLACEHOLDER, "typical item masses"), (CONFUSION_SYNTHETIC, "classifier confusion matrix"),
        (WEIGHT_PLACEHOLDER, "weight parameters")] if flag]    

    return {
        "source": source, "n_items": n_items, "result": result, "lanes": lanes,
        "routes_used": [r for r in ROUTES if any(s["route"] == r for l in lanes for s in l["segments"])],
        "route_labels": ROUTE_LABELS, "carbon": carbon,
        "saved": plan_result["saved_kgco2e"], "notes": plan_result.get("notes", []),
        "explanation": explanation[0], "explained_by": explanation[1], "explain_seconds": explanation[2],
        "factors_source": integrations.FACTORS_SOURCE, "limits": limits, "placeholders": placeholders,
    }


def machine():
    """Shown in the footer: s390x on IBM LinuxONE."""
    return platform.machine()
