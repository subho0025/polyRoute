import json
import time
import os

def build_granite_prompt(scenario_name, composition_data, opt_data):
    kg_dict = composition_data.get("kg", {})
    
    plan = opt_data.get("plan", {})
    saved_co2e = opt_data.get("saved_kgco2e", 0.0)
    notes = opt_data.get("notes", [])
    
    prompt = f"""<|system|>
You are an AI sustainability advisor for PolyRoute. Explain the recycling batch results to a non-technical user.
Rules:
1. Output exactly 3 to 5 sentences.
2. State the total CO2e saved.
3. Explain the routing decisions in simple, everyday language (e.g., mentioning facility space or material recyclability). 
4. DO NOT mention algorithms, databases, 'WARM factors', or how the math works behind the scenes.
5. If a plastic had to be sent to a landfill or combusted, point out why (e.g., the recycling facility was full, or that specific plastic isn't recyclable).
6. DO NOT invent or calculate any numbers. Use only the numbers provided.

<|user|>
Batch Context: {scenario_name}
Input Weights (kg): {json.dumps(kg_dict)}

Optimization Results:
- Total CO2e Saved: {saved_co2e} kg
- Optimized Routing Plan: {json.dumps(plan)}
- Engine Notes: {', '.join(notes) if notes else 'None'}

<|assistant|>
"""
    return prompt

def generate_fallback(saved_co2e, plan, notes):
    if "No plastic detected" in notes or not plan:
        return "No plastic was detected in this batch, so no routing plan was generated."
        
    fallback_text = f"This batch optimization saved {saved_co2e} kg of CO2e compared to baseline disposal. "
    
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
    
    if not use_local_model:
        return generate_fallback(saved_co2e, plan, notes), 0.0

    try:
        from llama_cpp import Llama
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Local model {model_path} not found.")
            
        llm = Llama(model_path=model_path, n_ctx=1024, n_gpu_layers=-1, verbose=False)
        
        output = llm(
            prompt,
            max_tokens=250,
            temperature=0.1,
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
