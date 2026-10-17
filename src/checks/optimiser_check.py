"""Check the optimiser integration: WARM factors -> optimiser/route_opt.py, then comp_analyzer -> optimiser.

Run from src/:   python -m checks.optimiser_check

Part A  setup (stops at the first problem and says how to fix it)
Part B  hand-checked cases (exact expected plans)
Part C  edge cases
Part D  stress test: 500 random batches compared with an independent solver (SciPy)
Part E  your teammate's generate_composition presets
Part F  full chain: comp_analyzer synthetic files -> to_optimiser_input -> optimiser

Every plan in B-F is also checked for: all kg routed, limits respected, no NA route used,
saved = landfill baseline - plan CO2e, and plan CO2e equal to the true optimum (SciPy).
"""

import random
import sys

import numpy as np
from scipy.optimize import linprog

FAILS = []
STEP = [0]
POLYMERS = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]
ROUTES = ["recycle", "compost", "landfill", "combust"]
CAPS = {"recycle": "recycle_cap_kg", "compost": "compost_cap_kg", "landfill": "landfill_cap_kg"}
DEFAULT_LIMITS = {"recycle_cap_kg": 40.0, "compost_cap_kg": 10.0, "landfill_cap_kg": 15.0}


def step(title):
    STEP[0] += 1
    print(f"\n[{STEP[0]}] {title}")


def stop(msg, fix=""):
    print(f"    FAIL {msg}")
    if fix:
        print(f"    FIX  {fix}")
    sys.exit(1)


def bad(msg):
    print(f"    FAIL {msg}")
    FAILS.append(f"[{STEP[0]}] {msg}")


def good(msg=""):
    print(f"    PASS {msg}")


# =========================================================== A. setup
print("A. SETUP")
step("PuLP is installed")
try:
    import pulp
except ImportError:
    stop("PuLP isn't installed.", "pip install pulp==4.0.1   (the version in the LinuxONE container)")
good(f"PuLP {pulp.__version__} ({'4.x API' if hasattr(pulp.LpProblem, 'add_variable') else '2/3 API'})")

step("The GLPK solver (glpsol) is available")
if not pulp.GLPK_CMD(msg=False).available():
    stop("route_opt.py solves with GLPK, but glpsol wasn't found.",
         "Fedora: sudo dnf install glpk-utils  |  LinuxONE container: "
         'docker exec -u root app bash -c "apt-get update && apt-get install -y glpk-utils"')
good()

step("Emission factors load (EPA WARM v16)")
from optimiser.factors import load_factors
from optimiser.route_opt import optimize_routes

F = load_factors()
if sorted(F) != sorted(POLYMERS):
    stop(f"Expected the 7 classes, got {sorted(F)}", "Check optimiser/data/emission_factors.csv")
for p, v in {"PET": -1.1464, "HDPE": -0.8378, "PP": -0.8708, "Other": -1.0251}.items():
    if abs(F[p]["recycle"] - v) > 1e-4:
        stop(f"{p} recycle factor is {F[p]['recycle']}, expected {v}")
if F["LDPE"]["recycle"] != "NA" or F["PET"]["compost"] != "NA":
    stop("NA routes should load as the string 'NA'")
if any(F[p]["landfill"] == "NA" or F[p]["combust"] == "NA" for p in POLYMERS):
    stop("Every polymer should have landfill and combustion factors")
good("7 classes x 4 routes, values match the WARM table")

step("route_opt.py solves a trivial batch")
probe = optimize_routes({"kg": {"PET": 1.0}}, DEFAULT_LIMITS, F)
if "error" in probe:
    stop("The optimiser reports 'Infeasible' for a trivial batch.",
         "Use the route_opt.py that supports both PuLP 3 and PuLP 4 (status_text = ...).")
good()


# =========================================================== helpers
def fv(p, r):
    v = F[p][r]
    return None if v == "NA" else v


def co2(plan):
    return sum(kg * fv(p, r) for p, routes in plan.items() for r, kg in routes.items())


def baseline(kg):
    return sum(v * fv(p, "landfill") for p, v in kg.items() if p in F and v > 0)


def reference_optimum(kg, limits):
    """The same model solved independently with SciPy (HiGHS)."""
    kg = {p: v for p, v in kg.items() if p in POLYMERS and v > 0}
    if not kg:
        return 0.0
    pairs = [(p, r) for p in kg for r in ROUTES if fv(p, r) is not None]
    c = np.array([fv(p, r) for p, r in pairs])
    a_eq = [[1.0 if q == p else 0.0 for q, _ in pairs] for p in kg]
    a_ub, b_ub = [], []
    for r, key in CAPS.items():
        if key in limits:
            a_ub.append([1.0 if rr == r else 0.0 for _, rr in pairs])
            b_ub.append(limits[key])
    res = linprog(c, A_ub=a_ub or None, b_ub=b_ub or None, A_eq=a_eq, b_eq=list(kg.values()),
                  bounds=(0, None), method="highs")
    return float(res.fun)


def check(kg, limits, out, quiet=False):
    """All the invariants every plan must meet. Returns a list of problems."""
    problems = []
    if "error" in out:
        return [f"optimiser error: {out['error']}"]
    plan = out["plan"]
    for p, v in kg.items():
        if p in POLYMERS and v > 0:
            routed = sum(plan.get(p, {}).values())
            if abs(routed - v) > 0.011 * max(1, len(plan.get(p, {}))):
                problems.append(f"{p}: {v} kg in, {routed:.3f} kg routed")
    for r, key in CAPS.items():
        if key in limits:
            used = sum(x.get(r, 0) for x in plan.values())
            if used > limits[key] + 0.03:
                problems.append(f"{r} uses {used:.2f} kg, limit {limits[key]}")
    for p, routes in plan.items():
        for r in routes:
            if fv(p, r) is None:
                problems.append(f"{p} sent to {r}, which is NA")
    plan_co2 = co2(plan)
    if abs((baseline(kg) - plan_co2) - out["saved_kgco2e"]) > 0.05:
        problems.append(f"saved {out['saved_kgco2e']} but baseline - plan = {baseline(kg) - plan_co2:.2f}")
    best = reference_optimum(kg, limits)
    if abs(plan_co2 - best) > 0.05:
        problems.append(f"plan CO2e {plan_co2:.3f} isn't optimal (best possible {best:.3f})")
    return problems


def case(title, kg, limits, expect_plan=None, extra=None):
    step(title)
    out = optimize_routes({"kg": kg}, limits, F)
    print(f"    kg in : {kg}")
    print(f"    limits: {limits}")
    if "error" in out:
        bad(out["error"])
        return out
    print(f"    plan  : {out['plan']}")
    print(f"    saved : {out['saved_kgco2e']} kg CO2e")
    problems = check(kg, limits, out)
    if expect_plan is not None:
        got = {p: {r: v for r, v in rs.items() if v > 0} for p, rs in out["plan"].items()}
        if got != expect_plan:
            problems.append(f"expected plan {expect_plan}")
        else:
            exp_saved = round(baseline(kg) - co2(expect_plan), 2)
            if abs(out["saved_kgco2e"] - exp_saved) > 0.011:
                problems.append(f"expected saving {exp_saved}")
    if extra:
        problems += extra(out)
    if problems:
        for pr in problems:
            bad(pr)
    else:
        good("plan is valid and optimal" + (" and matches the hand-worked answer" if expect_plan else ""))
    return out


BIG = {"recycle_cap_kg": 1e6, "compost_cap_kg": 1e6, "landfill_cap_kg": 1e6}

# =========================================================== B. hand-checked cases
print("\n\nB. HAND-CHECKED CASES")
case("Plenty of space: PET 10 + LDPE 5",
     {"PET": 10.0, "LDPE": 5.0}, DEFAULT_LIMITS,
     expect_plan={"PET": {"recycle": 10.0}, "LDPE": {"landfill": 5.0}})

case("Exact fit: PET 40 kg into exactly 40 kg of recycling",
     {"PET": 40.0}, DEFAULT_LIMITS, expect_plan={"PET": {"recycle": 40.0}})

case("Recycling priority: PET, Other, PP, HDPE 20 kg each, only 30 kg of recycling\n"
     "    (best savers first: PET -1.146 > Other -1.025 > PP -0.871 > HDPE -0.838)",
     {"PET": 20.0, "Other": 20.0, "PP": 20.0, "HDPE": 20.0},
     {"recycle_cap_kg": 30, "compost_cap_kg": 10, "landfill_cap_kg": 1000},
     expect_plan={"PET": {"recycle": 20.0}, "Other": {"recycle": 10.0, "landfill": 10.0},
                  "PP": {"landfill": 20.0}, "HDPE": {"landfill": 20.0}})

case("Landfill priority: PS 5 + PVC 5, landfill only 5 kg\n"
     "    (PS burns worst (+1.819) so it gets the landfill space; PVC (+0.728) is burned)",
     {"PS": 5.0, "PVC": 5.0}, {"recycle_cap_kg": 40, "compost_cap_kg": 10, "landfill_cap_kg": 5},
     expect_plan={"PS": {"landfill": 5.0}, "PVC": {"combust": 5.0}})

case("No recycling, landfill 10 kg: PS, HDPE, PET, PVC 5 kg each\n"
     "    (landfill goes to the worst burners: PS +1.819, HDPE +1.422; PET and PVC burned)",
     {"PS": 5.0, "HDPE": 5.0, "PET": 5.0, "PVC": 5.0},
     {"recycle_cap_kg": 0, "compost_cap_kg": 10, "landfill_cap_kg": 10},
     expect_plan={"PS": {"landfill": 5.0}, "HDPE": {"landfill": 5.0}, "PET": {"combust": 5.0},
                  "PVC": {"combust": 5.0}})

case("Recycling full: PET 30 + HDPE 20, recycling 40",
     {"PET": 30.0, "HDPE": 20.0}, DEFAULT_LIMITS,
     expect_plan={"PET": {"recycle": 30.0}, "HDPE": {"recycle": 10.0, "landfill": 10.0}})

case("Landfill full: LDPE 20, landfill 15 -> 5 kg overflow burned (a carbon cost)",
     {"LDPE": 20.0}, DEFAULT_LIMITS, expect_plan={"LDPE": {"landfill": 15.0, "combust": 5.0}},
     extra=lambda o: [] if o["saved_kgco2e"] < 0 else ["saving should be negative"])

# =========================================================== C. edge cases
print("\n\nC. EDGE CASES")
case("All limits zero: everything must be burned (combustion has no limit)",
     {"PET": 10.0, "PS": 5.0}, {"recycle_cap_kg": 0, "compost_cap_kg": 0, "landfill_cap_kg": 0},
     expect_plan={"PET": {"combust": 10.0}, "PS": {"combust": 5.0}})

case("No limits given at all (empty limits dict)",
     {"PET": 10.0, "LDPE": 10.0}, {},
     expect_plan={"PET": {"recycle": 10.0}, "LDPE": {"landfill": 10.0}})

case("Zero recycling capacity: recyclables go to landfill, not burned",
     {"PET": 10.0, "HDPE": 10.0}, {"recycle_cap_kg": 0, "compost_cap_kg": 10, "landfill_cap_kg": 100},
     expect_plan={"PET": {"landfill": 10.0}, "HDPE": {"landfill": 10.0}})

case("A polymer with 0 kg is ignored",
     {"PET": 0.0, "HDPE": 5.0}, DEFAULT_LIMITS, expect_plan={"HDPE": {"recycle": 5.0}})

case("Only non-recyclable plastics (PVC, LDPE, PS) fit in landfill",
     {"PVC": 4.0, "LDPE": 6.0, "PS": 3.0}, DEFAULT_LIMITS,
     expect_plan={"PVC": {"landfill": 4.0}, "LDPE": {"landfill": 6.0}, "PS": {"landfill": 3.0}})

out = case("Tiny amounts (grams): PET 0.004 kg, LDPE 0.001 kg",
           {"PET": 0.004, "LDPE": 0.001}, DEFAULT_LIMITS)
zeros = [(p, r) for p, rs in out.get("plan", {}).items() for r, v in rs.items() if v == 0]
if zeros:
    print(f"    INFO entries rounded to 0.0 kg: {zeros} (harmless; the dashboard hides amounts under 0.005 kg)")

case("Huge batch: 10 tonnes of every plastic with the default limits",
     {p: 10000.0 for p in POLYMERS}, DEFAULT_LIMITS)

case("Odd decimals like comp_analyzer produces",
     {"PET": 9.86, "HDPE": 7.87, "PVC": 1.59, "LDPE": 1.6, "PP": 3.25, "PS": 0.53, "Other": 1.16},
     DEFAULT_LIMITS)

case("Ties: HDPE and LDPE burn equally (+1.422); any split of the 10 kg landfill is optimal",
     {"HDPE": 10.0, "LDPE": 10.0}, {"recycle_cap_kg": 0, "compost_cap_kg": 10, "landfill_cap_kg": 10})

step("Scaling: 10x the kg and 10x the limits gives 10x the plan and 10x the saving")
kg1 = {"PET": 20.0, "Other": 20.0, "PP": 20.0, "HDPE": 20.0, "LDPE": 7.0}
lim1 = {"recycle_cap_kg": 30, "compost_cap_kg": 10, "landfill_cap_kg": 25}
o1 = optimize_routes({"kg": kg1}, lim1, F)
o10 = optimize_routes({"kg": {p: v * 10 for p, v in kg1.items()}}, {k: v * 10 for k, v in lim1.items()}, F)
if abs(o10["saved_kgco2e"] - 10 * o1["saved_kgco2e"]) > 0.2:
    bad(f"saving {o1['saved_kgco2e']} x10 = {10 * o1['saved_kgco2e']:.2f}, got {o10['saved_kgco2e']}")
else:
    good(f"{o1['saved_kgco2e']} -> {o10['saved_kgco2e']} kg CO2e")

step("Composting limit changes nothing (no plastic class can be composted)")
kg2 = {"PET": 9.86, "HDPE": 7.87, "PVC": 1.59, "LDPE": 1.6, "PP": 3.25, "PS": 0.53, "Other": 1.16}
a = optimize_routes({"kg": kg2}, {**DEFAULT_LIMITS, "compost_cap_kg": 0}, F)
b = optimize_routes({"kg": kg2}, {**DEFAULT_LIMITS, "compost_cap_kg": 1000}, F)
if a["plan"] != b["plan"] or a["saved_kgco2e"] != b["saved_kgco2e"]:
    bad("plan changed with the composting limit")
elif any("compost" in rs for rs in a["plan"].values()):
    bad("something was sent to composting")
else:
    good()

step("Empty batch")
o = optimize_routes({"kg": {}}, DEFAULT_LIMITS, F)
if o.get("plan") != {} or o.get("saved_kgco2e") != 0.0:
    bad(f"expected an empty plan, got {o}")
else:
    good(f"{o}")

step("A class name the optimiser doesn't know (e.g. 'PLA')")
o = optimize_routes({"kg": {"PET": 5.0, "PLA": 3.0}}, DEFAULT_LIMITS, F)
print(f"    plan: {o.get('plan')}")
if "PLA" in o.get("plan", {}):
    good("PLA was routed")
else:
    print("    INFO the 3 kg of PLA is silently left out: route_opt.py only routes its 7 fixed classes.\n"
          "         comp_analyzer only ever outputs those 7, so this can't happen in the pipeline.")

# =========================================================== D. stress test
print("\n\nD. STRESS TEST")
step("500 random batches and limits, each compared with SciPy's optimum")
rng = random.Random(42)
problems_seen = 0
worst_gap = 0.0
for i in range(500):
    kg = {p: round(rng.choice([rng.uniform(0.01, 5), rng.uniform(5, 60), rng.uniform(60, 500)]), 2)
          for p in rng.sample(POLYMERS, rng.randint(1, 7))}
    lim = {k: round(rng.choice([0, rng.uniform(0, 20), rng.uniform(20, 200), 1e5]), 1) for k in DEFAULT_LIMITS}
    if rng.random() < 0.1:
        lim.pop(rng.choice(list(lim)))  # sometimes a limit is missing
    out = optimize_routes({"kg": kg}, lim, F)
    pr = check(kg, lim, out)
    if "error" not in out:
        worst_gap = max(worst_gap, abs(co2(out["plan"]) - reference_optimum(kg, lim)))
    if pr:
        problems_seen += 1
        if problems_seen <= 3:
            bad(f"batch {i}: {kg} {lim} -> {pr}")
if problems_seen:
    bad(f"{problems_seen} of 500 batches had problems")
else:
    good(f"all 500 valid and optimal (largest gap to SciPy: {worst_gap:.4f} kg CO2e, from rounding to 0.01 kg)")

# =========================================================== E. teammate's generator
print("\n\nE. TEAMMATE'S generate_composition PRESETS")
step("generate_composition presets (seeded), default limits")
try:
    from optimiser import generate_composition as gc
except ImportError:
    print("    SKIP optimiser/generate_composition.py not found")
else:
    random.seed(7)
    presets = [("Standard Mixed Batch", 50.0, ["PET", "HDPE", "PP", "PVC"]),
               ("PET Heavy Batch", 40.0, ["PET", "HDPE"]),
               ("Unrecyclable High-PVC Batch", 35.0, ["PVC", "PP"]),
               ("Large Scale Batch (Exceeds Caps)", 120.0, ["PET", "HDPE", "PP", "PVC", "LDPE", "PS"]),
               ("Small Batch", 15.0, ["PET", "PP"])]
    ok_all = True
    for name, w, polys in presets:
        cd = gc.generate_composition(total_kg=w, polymers=polys)
        out = optimize_routes(cd, DEFAULT_LIMITS, F)
        pr = check(cd["kg"], DEFAULT_LIMITS, out)
        print(f"    {name:34} saved {out.get('saved_kgco2e', 'ERR'):>8}  {'ok' if not pr else pr}")
        ok_all &= not pr
    if ok_all:
        good()
    else:
        bad("a preset failed")

# =========================================================== F. full chain
print("\n\nF. FULL CHAIN: comp_analyzer -> optimiser")
from comp_analyzer import (estimate_composition, load_confusion, load_item_mass, load_weight_params,
                           to_optimiser_input)
from comp_analyzer.config import SYNTHETIC_DIR
import json

step("Synthetic classifier outputs exist")
if not (SYNTHETIC_DIR / "medium_100.json").exists():
    stop("No synthetic files.", "python -m comp_analyzer.gen")
good()

MASSES, _ = load_item_mass()
CONF, _ = load_confusion()
WP, _ = load_weight_params()

runs = [("small_20.json", 10), ("medium_100.json", 50), ("large_500.json", 200), ("xlarge_1000.json", 400),
        ("pet_heavy_100.json", 50), ("all_non_plastic.json", 20), ("only_uncertain.json", 20), ("empty.json", 20)]
step("Each synthetic file -> estimate_composition -> to_optimiser_input -> optimize_routes")
chain_ok = True
for name, total in runs:
    items = json.loads((SYNTHETIC_DIR / name).read_text())
    try:
        result = estimate_composition(items, total_kg=total, masses=MASSES, confusion=CONF, weight_params=WP)
    except ValueError as e:
        print(f"    {name:22} comp_analyzer refused it: {e}")
        continue
    cd = to_optimiser_input(result)["composition_data"]
    out = optimize_routes(cd, DEFAULT_LIMITS, F)
    pr = check(cd["kg"], DEFAULT_LIMITS, out)
    plastic = sum(cd["kg"].values())
    print(f"    {name:22} {total:>4} kg batch, {plastic:6.2f} kg plastic -> saved {out.get('saved_kgco2e', 'ERR'):>7}"
          f"  {'ok' if not pr else pr}")
    chain_ok &= not pr
if chain_ok:
    good()
else:
    bad("a file failed in the full chain")

step("Same batch with total='plastic' (optimiser only cares about kg, so the plan must be identical)")
items = json.loads((SYNTHETIC_DIR / "medium_100.json").read_text())
result = estimate_composition(items, total_kg=50, masses=MASSES, confusion=CONF, weight_params=WP)
o_batch = optimize_routes(to_optimiser_input(result)["composition_data"], DEFAULT_LIMITS, F)
o_plastic = optimize_routes(to_optimiser_input(result, total="plastic")["composition_data"], DEFAULT_LIMITS, F)
if o_batch["plan"] != o_plastic["plan"]:
    bad("plans differ")
else:
    good()

# =========================================================== summary
print("\n" + "=" * 60)
if FAILS:
    print(f"{len(FAILS)} problem(s):")
    for f in FAILS:
        print("  " + f)
    sys.exit(1)
print("All optimiser checks passed. Next: python -m checks.explainer_check")