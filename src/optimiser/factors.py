"""
EPA WARM v16 emission factors in the format route_opt.optimize_routes expects.

    from optimiser.factors import load_factors
    factors = load_factors()
    factors["PET"]["recycle"]   # -1.1464 kg CO2e per kg (negative = saving)
    factors["PET"]["compost"]   # "NA" -> route not allowed for PET

Source: US EPA WARM Version 16 (Dec 2023), "Containers, Packaging and Non-Durable Goods
Materials", Exhibit 5-3 (plastics) and Exhibit 4-4 (PLA). Converted from MTCO2E per short ton
to kg CO2e per kg (x 1.10231). "Other" uses WARM's "Mixed Plastics"
"""

import csv
from pathlib import Path

FACTORS_FILE = Path(__file__).resolve().parent / "data" / "emission_factors.csv"


def load_factors(path=FACTORS_FILE, include_pla=False):
    """ {polymer: {route: kg CO2e per kg, or "NA"}} for every polymer and all four routes."""
    factors = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["polymer"] == "PLA" and not include_pla:
                continue
            value = row["kg_co2e_per_kg"].strip()
            factors.setdefault(row["polymer"], {})[row["route"]] = float(value) if value else "NA"
    return factors