"""Self-checks for comp_analyzer. No extra packages needed.

Usage (from src/):
    python -m comp_analyzer.tests

Prints PASS/FAIL per test and exits with code 1 if anything fails.
"""

import sys

import numpy as np

from . import (POLYMERS, estimate_composition, load_confusion, load_item_mass,
               load_weight_params, to_optimiser_input)
from .config import CONFUSION_FILE, ITEM_MASS_FILE, WEIGHT_PARAMS_FILE
from .mix import adjusted_mix

MASSES, _ = load_item_mass(ITEM_MASS_FILE)
CONFUSION, _ = load_confusion(CONFUSION_FILE)
PARAMS, _ = load_weight_params(WEIGHT_PARAMS_FILE)


def plastic(polymer, probs=None):
    item = {"image": "x.jpg", "material": "plastic", "polymer": polymer, "confidence": 0.9}
    if probs:
        item["probs"] = probs
    return item


NON_PLASTIC = {"image": "x.jpg", "material": "non_plastic", "polymer": None, "confidence": 0.9}
UNCERTAIN = {"image": "x.jpg", "material": "uncertain", "polymer": None, "confidence": 0.4}

# The worked example from comp_analyzer/README.md, section 6
EXAMPLE = [plastic("PET")] * 3 + [plastic("LDPE")] * 2 + [plastic("HDPE")] + [NON_PLASTIC] * 2 + [UNCERTAIN] * 2


# ---------- tests ----------

def test_worked_example():
    r = estimate_composition(EXAMPLE, total_kg=10, masses=MASSES, n_boot=0)
    assert r["kg"] == {"PET": 1.76, "HDPE": 0.94, "LDPE": 0.28}, r["kg"]
    assert r["plastic_fraction"] == 0.297, r["plastic_fraction"]
    assert r["excluded"] == {"non_plastic": 2, "uncertain": 2}
    assert r["method"] == "count"


def test_without_masses_counts_equally():
    r = estimate_composition(EXAMPLE, total_kg=10, n_boot=0)
    # 6 of 8 known items are plastic -> 7.5 kg; PET is half of the plastic items
    assert r["plastic_kg"] == 7.5 and r["kg"]["PET"] == 3.75, r


def test_kg_adds_up():
    r = estimate_composition(EXAMPLE * 20, total_kg=50, masses=MASSES)
    assert abs(sum(r["kg"].values()) - r["plastic_kg"]) < 0.05
    assert r["plastic_kg"] <= r["total_kg"]


def test_intervals_contain_estimate():
    r = estimate_composition(EXAMPLE * 20, total_kg=50, masses=MASSES)
    for p, kg in r["kg"].items():
        low, high = r["kg_ci"][p]
        assert low <= kg <= high, (p, low, kg, high)


def test_reproducible():
    a = estimate_composition(EXAMPLE * 10, total_kg=50, masses=MASSES, seed=7)
    b = estimate_composition(EXAMPLE * 10, total_kg=50, masses=MASSES, seed=7)
    assert a == b


def test_weight_photos_only():
    r = estimate_composition(EXAMPLE, masses=MASSES, n_boot=0)
    assert r["total_kg_source"] == "photos_only" and r["total_kg"] == 0.53, r["total_kg"]


def test_weight_volume():
    r = estimate_composition(EXAMPLE, batch={"volume_m3": 1.2}, masses=MASSES,
                             weight_params=PARAMS, n_boot=200)
    typical = PARAMS["bulk_density_kg_m3"]["typical"]
    assert r["total_kg_source"] == "volume" and r["total_kg"] == round(1.2 * typical, 2)
    assert r["total_kg_ci"][0] < r["total_kg"] < r["total_kg_ci"][1]


def test_weight_item_count():
    r = estimate_composition(EXAMPLE, batch={"item_count": 1000}, masses=MASSES, n_boot=0)
    # average typical mass of the 10 items = 533.75 g / 10 -> 1000 items = 53.38 kg
    assert r["total_kg_source"] == "item_count" and abs(r["total_kg"] - 53.38) < 0.01, r["total_kg"]


def test_auto_method_choice():
    assert estimate_composition(EXAMPLE, 10, n_boot=0)["method"] == "count"
    probs = {p: (0.6 if p == "PET" else 0.4 / 6) for p in POLYMERS}
    assert estimate_composition([plastic("PET", probs)] * 5, 10, n_boot=0)["method"] == "soft"
    many = [plastic("PET")] * 300 + [plastic("PP")] * 250
    assert estimate_composition(many, 10, confusion=CONFUSION, n_boot=0)["method"] == "adjusted"


def test_soft_counts_split_items():
    probs = {p: 0.0 for p in POLYMERS} | {"PET": 0.6, "PP": 0.4}
    r = estimate_composition([plastic("PET", probs)], total_kg=10, n_boot=0)
    assert r["kg"] == {"PET": 6.0, "PP": 4.0}, r["kg"]


def test_adjusted_mix_recovers_truth():
    # Without sampling noise, observed = C^T @ truth; correction must give back the truth
    truth = np.array([0.35, 0.15, 0.03, 0.20, 0.15, 0.07, 0.05])
    observed = CONFUSION.T @ truth
    counts = np.round(observed * 100000).astype(int)
    items = [plastic(p) for p, c in zip(POLYMERS, counts) for _ in range(c)]
    assert np.allclose(adjusted_mix(items, CONFUSION), truth, atol=0.002)


def test_override():
    r = estimate_composition(EXAMPLE, total_kg=50, kg_override={"PET": 20, "PP": 5})
    assert r["kg"] == {"PET": 20.0, "PP": 5.0} and r["method"] == "user_override"


def test_empty_and_errors():
    r = estimate_composition([], total_kg=50)
    assert r["kg"] == {} and r["warnings"]
    for bad_call in (
        lambda: estimate_composition([plastic("pet")], 10),                 # wrong spelling
        lambda: estimate_composition([{"material": "glass"}], 10),          # unknown material
        lambda: estimate_composition(EXAMPLE, total_kg=-5),                 # negative weight
        lambda: estimate_composition([]),                                   # no items, no weight
    ):
        try:
            bad_call()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_optimiser_format():
    out = to_optimiser_input(estimate_composition(EXAMPLE * 5, total_kg=50, masses=MASSES))
    data = out["composition_data"]
    assert set(data) == {"total_kg", "kg", "margin", "excluded"}
    assert set(data["kg"]) == set(data["margin"])
    assert all(0 <= m <= 1 for m in data["margin"].values())
    assert sum(data["kg"].values()) <= data["total_kg"]


def test_reference_files_load():
    assert set(POLYMERS) | {"non_plastic"} <= set(MASSES)
    assert CONFUSION.shape == (7, 7) and np.allclose(CONFUSION.sum(axis=1), 1)
    assert {"mass_cv", "bulk_density_kg_m3"} <= set(PARAMS)


# ---------- runner ----------

def main():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # show every failure, not just the first
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
