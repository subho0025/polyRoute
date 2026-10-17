import json
import random
import math

def generate_composition(
    total_kg: float = 50.0,
    polymers: list = None,
    sampling_mode: bool = True,
    sample_size: int = 50
) -> dict:
    """
    Generates realistic dummy composition data mimicking the output 
    of the PolyRoute ML & Composition module.
    
    Parameters:
    - total_kg: Total weight of the waste batch in kg.
    - polymers: List of polymers to include in the batch.
    - sampling_mode: Whether to calculate margin of error (sampling mode).
    - sample_size: Total number of plastic items identified in sampling.
    """
    if polymers is None:
        polymers = ["PET", "HDPE", "PP", "PVC"]
    
    # Randomly assign item counts across selected polymers
    # to simulate raw item detections from the ML model
    raw_counts = {p: random.randint(1, 20) for p in polymers}
    total_items = sum(raw_counts.values())
    
    # Calculate weight per polymer proportional to item share
    kg_dist = {}
    margin_dist = {}
    
    for p in polymers:
        share = raw_counts[p] / total_items
        kg_dist[p] = round(share * total_kg, 2)
        
        # Calculate 95% confidence margin: 1.96 * sqrt(p * (1 - p) / n)
        if sampling_mode and sample_size > 0:
            margin = 1.96 * math.sqrt((share * (1.0 - share)) / sample_size)
            margin_dist[p] = round(margin, 4)
        else:
            margin_dist[p] = 0.0

    # Ensure total weights sum cleanly to total_kg
    current_sum = sum(kg_dist.values())
    diff = round(total_kg - current_sum, 2)
    if diff != 0 and polymers:
        kg_dist[polymers[0]] = round(kg_dist[polymers[0]] + diff, 2)

    # Excluded items count (non-plastic and low-confidence items)
    excluded = {
        "non_plastic": random.randint(0, 5),
        "uncertain": random.randint(0, 3)
    }

    return {
        "total_kg": float(total_kg),
        "kg": kg_dist,
        "margin": margin_dist,
        "excluded": excluded
    }

def generate_batch_dataset(num_samples: int = 5, output_file: str = "generated_compositions.json"):
    """Generates a batch of diverse test cases and writes them to a JSON file."""
    dataset = []
    
    # Defined presets to test edge cases
    presets = [
        {"name": "Standard Mixed Batch", "weight": 50.0, "polymers": ["PET", "HDPE", "PP", "PVC"]},
        {"name": "PET Heavy Batch", "weight": 40.0, "polymers": ["PET", "HDPE"]},
        {"name": "Unrecyclable High-PVC Batch", "weight": 35.0, "polymers": ["PVC", "PP"]},
        {"name": "Large Scale Batch (Exceeds Caps)", "weight": 120.0, "polymers": ["PET", "HDPE", "PP", "PVC", "LDPE", "PS"]},
        {"name": "Small Batch", "weight": 15.0, "polymers": ["PET", "PP"]}
    ]

    for i in range(num_samples):
        preset = presets[i % len(presets)]
        data = generate_composition(
            total_kg=preset["weight"],
            polymers=preset["polymers"],
            sampling_mode=True,
            sample_size=random.randint(30, 100)
        )
        dataset.append({
            "scenario_id": i + 1,
            "scenario_name": preset["name"],
            "composition_data": data
        })

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Successfully generated {len(dataset)} composition test cases in '{output_file}'")

if __name__ == "__main__":
    # Generate 5 test scenarios
    generate_batch_dataset(num_samples=5)
