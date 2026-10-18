"""End-to-end check with real photos: classifier -> composition -> optimiser -> explainer -> dashboard API.

Run from src/:
    python -m checks.pipeline_check /path/to/photos            # weighed batch of 50 kg (default)
    python -m checks.pipeline_check /path/to/photos --kg 12.5
    python -m checks.pipeline_check /path/to/photos --photos-only   # photos show the whole batch
Granite is used if POLYROUTE_GRANITE_GGUF points to the model; set POLYROUTE_USE_GRANITE=0 to skip it.
"""

import argparse
import io
import sys
import time
from pathlib import Path

from comp_analyzer import estimate_composition, to_optimiser_input
from dashboard import integrations, pipeline

parser = argparse.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("--kg", type=float, default=50.0)
parser.add_argument("--photos-only", action="store_true")
args = parser.parse_args()

paths = sorted(p for p in Path(args.folder).iterdir()
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
if not paths:
    sys.exit(f"No photos (.jpg/.jpeg/.png) found in {args.folder}")


class Upload:  # what Flask hands the pipeline for each uploaded photo
    def __init__(self, path):
        self.filename, self.stream = path.name, io.BytesIO(path.read_bytes())


print(f"\n[1] CLASSIFIER: {len(paths)} photos")
t = time.time()
records = integrations.classify_photos([Upload(p) for p in paths])
print(f"    ({time.time() - t:.1f} s, includes loading TensorFlow and the model the first time)")
for r in records:
    top = sorted(r.get("probs", {}).items(), key=lambda kv: -kv[1])[:2]
    print(f"    {r['image'][:40]:40} {r['material']:10} {str(r['polymer']):6} conf {r['confidence']:.2f}"
          f"   top-2: {', '.join(f'{k} {v:.2f}' for k, v in top)}")

print("\n[2] COMPOSITION ESTIMATE")
total_kg = None if args.photos_only else args.kg  # None + no batch info = "the photos show the whole batch"
result = estimate_composition(records, total_kg=total_kg, masses=pipeline.MASSES,
                              confusion=pipeline.CONFUSION, weight_params=pipeline.WEIGHT_PARAMS)
print(f"    batch {result['total_kg']} kg ({result['total_kg_source']}), plastic {result['plastic_kg']} kg, "
      f"method {result['method']}, left out {result['excluded']}")
for p, kg in result["kg"].items():
    lo, hi = result["kg_ci"][p]
    print(f"    {p:6} {kg:7.2f} kg   (95%: {lo}-{hi})")
for w in result["warnings"]:
    print(f"    note: {w}")

print(f"\n[3] OPTIMISER (limits {pipeline.DEFAULT_LIMITS})")
cd = to_optimiser_input(result)["composition_data"]
plan = integrations.run_optimiser(cd, pipeline.DEFAULT_LIMITS)
for p, routes in plan["plan"].items():
    print(f"    {p:6} " + ", ".join(f"{r} {kg} kg" for r, kg in routes.items()))
print(f"    CO2e: everything to landfill {plan['baseline_kgco2e']} -> plan {plan['optimised_kgco2e']} "
      f"=> {'saves' if plan['saved_kgco2e'] >= 0 else 'COSTS'} {abs(plan['saved_kgco2e'])} kg")
for n in plan["notes"]:
    print(f"    note: {n}")

print("\n[4] EXPLAINER")
t = time.time()
text, source, secs = integrations.explain_plan(cd, plan, f"{len(paths)} uploaded photos")
print(f"    by {source} ({time.time() - t:.1f} s)\n    {text}")

print("\n[5] DASHBOARD API (same photos through POST /api/analyse)")
from dashboard.app import app

form = {"weight_mode": "photos" if args.photos_only else "measured", "total_kg": str(args.kg),
        "photos": [(io.BytesIO(p.read_bytes()), p.name) for p in paths]}
r = app.test_client().post("/api/analyse", data=form, content_type="multipart/form-data")
j = r.get_json()
if r.status_code != 200:
    sys.exit(f"    FAIL {r.status_code}: {j.get('error')}")
same = abs(j["saved"] - plan["saved_kgco2e"]) < 0.01 and j["n_items"] == len(paths)
print(f"    {r.status_code}, source '{j['source']}', {j['n_items']} items, saved {j['saved']} kg, "
      f"explained by {j['explained_by']}  ->  {'matches steps 1-4' if same else 'DIFFERS from steps 1-4'}")
print(f"    placeholders in use: {j['placeholders'] or 'none'}")
print("\nDone.")