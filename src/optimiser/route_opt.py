import pulp

def optimize_routes(composition_data, limits, factors):
    """
    Optimizes plastic waste routing based on PuLP linear programming.
    Expects composition_data as defined in comp_analyzer.
    """
    kg_dict = composition_data.get("kg", {})
    if not kg_dict:
        return {
            "plan": {},
            "saved_kgco2e": 0.0,
            "notes": ["No plastic detected"]
        }

    total_plastic_weight = sum(kg_dict.values())
    
    prob = pulp.LpProblem("PolyRoute_Optimization", pulp.LpMinimize)
    x = {}
    
    polymers = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]
    routes = ["recycle", "compost", "landfill", "combust"]
    
    for p in polymers:
        p_weight = kg_dict.get(p, 0.0)
        if p_weight == 0.0:
            continue
            
        for r in routes:
            if factors.get(p, {}).get(r) != "NA":
                x[(p, r)] = pulp.LpVariable(f"x_{p}_{r}", lowBound=0)
                
    # Objective: minimize Σ x[p, r] × EF[p, r]
    prob += pulp.lpSum([x[(p, r)] * factors[p][r] for (p, r) in x])
    
    # Constraint 1: Each polymer fully routed
    for p in polymers:
        p_weight = kg_dict.get(p, 0.0)
        if p_weight > 0.0:
            prob += pulp.lpSum([x[(p, r)] for r in routes if (p, r) in x]) == p_weight
            
    # Constraints 2: Facility capacities
    if "recycle_cap_kg" in limits:
        prob += pulp.lpSum([x[(p, "recycle")] for p in polymers if (p, "recycle") in x]) <= limits["recycle_cap_kg"]
    if "compost_cap_kg" in limits:
        prob += pulp.lpSum([x[(p, "compost")] for p in polymers if (p, "compost") in x]) <= limits["compost_cap_kg"]
    if "landfill_cap_kg" in limits:
        prob += pulp.lpSum([x[(p, "landfill")] for p in polymers if (p, "landfill") in x]) <= limits["landfill_cap_kg"]
        
    status = prob.solve(pulp.GLPK_CMD(msg=False))
    
    if "Optimal" not in str(status):
        return {
            "error": (
                f"Infeasible: The plastic batch weight ({total_plastic_weight} kg) exceeds available facility capacities. "
                "Please raise the landfill or combustion limits in the dashboard to process this overflow."
            )
        }
        
    baseline_kgco2e = 0.0
    for p in polymers:
        p_weight = kg_dict.get(p, 0.0)
        if p_weight > 0.0:
            if factors[p].get("landfill") != "NA":
                baseline_kgco2e += p_weight * factors[p]["landfill"]
            elif factors[p].get("combust") != "NA":
                baseline_kgco2e += p_weight * factors[p]["combust"]
                
    optimized_kgco2e = pulp.value(prob.objective)
    saved_co2e = round(baseline_kgco2e - optimized_kgco2e, 2)
    
    plan = {}
    for (p, r) in x:
        val = pulp.value(x[(p, r)])
        if val and val > 0:
            if p not in plan:
                plan[p] = {}
            plan[p][r] = round(val, 2)
            
    return {
        "plan": plan,
        "saved_kgco2e": saved_co2e,
        "notes": []
    }
