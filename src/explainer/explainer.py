import json
import time
import os

_LLM_CACHE = {}

def build_granite_prompt(scenario_name, composition_data, opt_data):
    kg_dict = composition_data.get("kg", {})
    
    plan = opt_data.get("plan", {})
    saved_co2e = opt_data.get("saved_kgco2e", 0.0)
    notes = opt_data.get("notes", [])

    if saved_co2e >= 0:
        carbon_line = f"Total CO2e Saved: {saved_co2e} kg"
    else:
        carbon_line = (f"Total CO2e Saved: none. This plan EMITS {abs(saved_co2e)} kg CO2e MORE than "
                       "sending everything to landfill, because the facilities were full")

    prompt = f"""<|system|>
You are an AI sustainability advisor for PolyRoute. Explain the recycling batch results to a non-technical user.
Rules:
1. Output exactly 3 to 5 sentences.
2. State the total CO2e saved. If the plan emits more CO2e than landfill, say that plainly and never call it a saving.
3. Explain the routing decisions in simple, everyday language (e.g., mentioning facility space or material recyclability).
4. DO NOT mention algorithms, databases, 'WARM factors', or how the math works behind the scenes.
5. If a plastic had to be sent to a landfill or combusted, point out why (e.g., the recycling facility was full, or that specific plastic isn't recyclable).
6. DO NOT invent or calculate any numbers. Use only the numbers provided.
7. DO NOT add comparisons (cars, trees, homes) or reasons that are not in the Engine Notes.
8. Only give the TOTAL CO2e figure. Never give a CO2e amount for a single plastic; the kg numbers are weights, not CO2e.

<|user|>
Batch Context: {scenario_name}
Input Weights (kg): {json.dumps(kg_dict)}

Optimization Results:
- {carbon_line}
- Optimized Routing Plan: {json.dumps(plan)}
- Engine Notes: {', '.join(notes) if notes else 'None'}

<|assistant|>
"""
    return prompt

def generate_fallback(saved_co2e, plan, notes):
    if "No plastic detected" in notes or not plan:
        return "No plastic was detected in this batch, so no routing plan was generated."

    if saved_co2e >= 0:
        fallback_text = f"This batch optimization saved {saved_co2e} kg of CO2e compared to baseline disposal. "
    else:
        fallback_text = (f"This batch optimization emits {abs(saved_co2e)} kg of CO2e more than baseline "
                         "disposal, because the facilities were full. ")

    for polymer, routes in plan.items():
        for route, amount in routes.items():
            fallback_text += f"Routed {amount} kg of {polymer} to {route}. "

    if notes:
        fallback_text += "Notes: " + " ".join(notes)

    return fallback_text.strip()

def generate_explanation(prompt, opt_data, use_local_model=True, model_path="./granite-4.0-h-instruct.gguf"):
    start_time = time.time()

    plan = opt_data.get("plan", {})
    saved_co2e = opt_data.get("saved_kgco2e", 0.0)
    notes = opt_data.get("notes", [])

    if not use_local_model or not plan:
        return generate_fallback(saved_co2e, plan, notes), 0.0

    try:
        from llama_cpp import Llama

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local model {model_path} not found.")

        if model_path not in _LLM_CACHE:
            threads = int(os.environ.get("POLYROUTE_THREADS", "0")) or None
            _LLM_CACHE[model_path] = Llama(model_path=model_path, n_ctx=1024, n_threads=threads,
                                           n_gpu_layers=-1, verbose=False)
        llm = _LLM_CACHE[model_path]

        output = llm(
            prompt,
            max_tokens=250,
            temperature=0.0,
            stop=["<|user|>", "<|system|>"]
        )

        response_text = output['choices'][0]['text'].strip()
        elapsed_time = round(time.time() - start_time, 2)

        return response_text, elapsed_time

    except ImportError:
        print("[LLM Warning] llama-cpp-python not installed. Using template fallback.")
        elapsed_time = round(time.time() - start_time, 2)
        return generate_fallback(saved_co2e, plan, notes), elapsed_time

    except Exception as e:
        print(f"[LLM Warning] Local model failed ({str(e)}). Using template fallback.")
        elapsed_time = round(time.time() - start_time, 2)
        return generate_fallback(saved_co2e, plan, notes), elapsed_time