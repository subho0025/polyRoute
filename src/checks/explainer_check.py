"""Check the explainer (explainer/explainer.py).   Run from src/:  python -m checks.explainer_check

Works with your teammate's original explainer.py, and with the improved version if you switch later.
It only relies on the functions both have: build_granite_prompt, generate_fallback, generate_explanation.

  FAIL = broken (crash, wrong format, missing fallback)
  WARN = works, but the result can be inaccurate or slow; listed at the end as things to improve

Part A  no model needed
Part B  with the Granite model (needs llama-cpp-python and the .gguf file)
        add --quick to run 2 scenarios instead of 5; --force to skip the free-memory check

Model file: set POLYROUTE_GRANITE_GGUF=/path/to/model.gguf, or put it at
src/explainer/granite-4.0-h-instruct.gguf or src/granite-4.0-h-instruct.gguf
"""

import os
import re
import sys
import time
from pathlib import Path

from explainer import explainer as ex

FAILS, WARNS = [], []
QUICK = "--quick" in sys.argv   # python -m checks.explainer_check --quick : 2 scenarios instead of 5
STEP = [0]
IMPROVED = hasattr(ex, "build_granite_messages")


def step(title):
    STEP[0] += 1
    print(f"\n[{STEP[0]}] {title}")


def good(msg=""):
    print(f"    PASS {msg}")


def bad(msg):
    print(f"    FAIL {msg}")
    FAILS.append(f"[{STEP[0]}] {msg}")


def warn(msg):
    print(f"    WARN {msg}")
    WARNS.append(f"[{STEP[0]}] {msg}")


def indent(text):
    return "    " + str(text).replace("\n", "\n    ")


def summary(ran_model=True):
    print("\n" + "=" * 64)
    if WARNS:
        print(f"{len(WARNS)} thing(s) to improve (the explainer still works):")
        for w in WARNS:
            print("  " + w)
    if FAILS:
        print(f"{len(FAILS)} problem(s):")
        for f in FAILS:
            print("  " + f)
        sys.exit(1)
    print("All explainer checks passed." if ran_model else
          "Part A passed. Part B (Granite) didn't run: see the STOP above.")
    sys.exit(0)


# ------------------------------------------------------------- an independent number check
def number_tokens(text):
    text = re.sub(r"CO\s?2|CO₂", "CO", text)                 # the 2 in CO2 isn't a quantity
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)          # 9,960 -> 9960
    return re.findall(r"\d+(?:\.\d+)?", text)


def invented_numbers(answer, facts):
    """Numbers in the answer that aren't in the facts (correct rounding is allowed)."""
    allowed = [float(t) for t in number_tokens(facts)]
    out = []
    for t in number_tokens(answer):
        n, d = float(t), (len(t.split(".")[1]) if "." in t else 0)
        tol = 1e-9 if d >= 2 else 0.5 * 10 ** -d + 1e-9
        if not any(abs(n - a) <= tol for a in allowed):
            out.append(t)
    return out


def facts_of(prompt):
    """Only the batch data, not the rules (the rules contain numbers like "3 to 5 sentences")."""
    if isinstance(prompt, str):
        return prompt.split("<|user|>")[-1].split("<|assistant|>")[0]
    return prompt[-1]["content"]


CACHED = hasattr(ex, "get_llm") or hasattr(ex, "_LLM_CACHE")
COMPARISONS = r"\b(cars?|trees?|homes?|houses?|flights?|equivalent to|the same as taking)\b"


def make_prompt(name, comp, opt):
    """Chat messages if the improved explainer is installed, otherwise the original raw prompt."""
    if IMPROVED:
        return ex.build_granite_messages(name, comp, opt)
    return ex.build_granite_prompt(name, comp, opt)


# ------------------------------------------------------------- scenarios (written out by hand)
SCENARIOS = {
    "normal batch": (
        {"kg": {"PET": 9.86, "HDPE": 7.87, "PVC": 1.59, "LDPE": 1.6, "PP": 3.25, "PS": 0.53, "Other": 1.16}},
        {"plan": {"PET": {"recycle": 9.86}, "HDPE": {"recycle": 7.87}, "PVC": {"landfill": 1.59},
                  "LDPE": {"landfill": 1.6}, "PP": {"recycle": 3.25}, "PS": {"landfill": 0.53},
                  "Other": {"recycle": 1.16}},
         "saved_kgco2e": 22.4,
         "notes": ["PVC, LDPE and PS have no recycling factor in EPA WARM, so the plan uses landfill where "
                   "there is space and energy recovery after that."]}),
    "recycling full": (
        {"kg": {"PET": 30.0, "HDPE": 20.0}},
        {"plan": {"PET": {"recycle": 30.0}, "HDPE": {"recycle": 10.0, "landfill": 10.0}},
         "saved_kgco2e": 43.65,
         "notes": ["Recycling is full at 40 kg, so 10 kg goes to landfill instead."]}),
    "landfill overflow (carbon cost)": (
        {"kg": {"LDPE": 20.0}},
        {"plan": {"LDPE": {"landfill": 15.0, "combust": 5.0}},
         "saved_kgco2e": -7.0,
         "notes": ["LDPE has no recycling factor in EPA WARM, so the plan uses landfill where there is "
                   "space and energy recovery after that.",
                   "Landfill is full at 15 kg, so 5 kg goes to energy recovery instead."]}),
    "single plastic, no notes": (
        {"kg": {"PET": 40.0}},
        {"plan": {"PET": {"recycle": 40.0}}, "saved_kgco2e": 46.74, "notes": []}),
    "huge batch": (
        {"kg": {"PET": 10000.0, "PS": 10000.0}},
        {"plan": {"PET": {"recycle": 40.0, "combust": 9960.0}, "PS": {"landfill": 15.0, "combust": 9985.0}},
         "saved_kgco2e": -31781.53,
         "notes": ["Recycling is full at 40 kg, so 9960 kg goes to energy recovery instead.",
                   "Landfill is full at 15 kg, so 9985 kg goes to energy recovery instead."]}),
}
EMPTY = ({"kg": {}}, {"plan": {}, "saved_kgco2e": 0.0, "notes": ["No plastic detected"]})

print(f"Testing explainer/explainer.py ({'improved version' if IMPROVED else 'original version'})")

# ============================================================= A. no model
print("\nA. WITHOUT THE MODEL")

step("Prompt contains the batch, the weights, the plan, the saving and the notes")
comp, opt = SCENARIOS["recycling full"]
p = ex.build_granite_prompt("medium_100.json", comp, opt)
print(indent(p.split("<|user|>")[1].split("<|assistant|>")[0].strip()))
missing = [x for x in ["<|system|>", "<|user|>", "<|assistant|>", "medium_100.json", "30", "43.65",
                       "Recycling is full at 40 kg"] if x not in p]
if missing:
    bad(f"prompt is missing: {missing}")
else:
    good()

step("Prompt with no notes")
comp, opt = SCENARIOS["single plastic, no notes"]
p = ex.build_granite_prompt("b", comp, opt)
if "None" in p or "Reasons" not in p:
    good()
else:
    bad("expected the notes to be shown as empty")

step("Fallback text for a normal batch")
comp, opt = SCENARIOS["recycling full"]
fb = ex.generate_fallback(opt["saved_kgco2e"], opt["plan"], opt["notes"])
print(indent(fb))
if "43.65" not in fb or "PET" not in fb or "HDPE" not in fb:
    bad("the fallback should state the saving and every plastic")
else:
    good()

step("Fallback for an empty plan, and for the 'No plastic detected' note")
a = ex.generate_fallback(0.0, {}, [])
b = ex.generate_fallback(0.0, {}, ["No plastic detected"])
if "No plastic" in a and "No plastic" in b:
    good(a)
else:
    bad(f"got: {a!r} / {b!r}")

step("A batch that COSTS carbon (saved = -7.0) is described correctly")
comp, opt = SCENARIOS["landfill overflow (carbon cost)"]
p = facts_of(make_prompt("b", comp, opt))
fb = ex.generate_fallback(opt["saved_kgco2e"], opt["plan"], opt["notes"])
print(indent("fallback: " + fb.split(". ")[0] + "."))
problems = []
if re.search(r"[Ss]aved:?\s*-\d", p):
    problems.append("the prompt tells Granite 'Saved: -7.0 kg', so it may say the plan saves carbon")
if re.search(r"saved\s*-\d", fb):
    problems.append("the fallback says 'saved -7.0 kg' instead of 'emits 7 kg more'")
for pr in problems:
    warn(pr)
if not problems:
    good()

step("Big numbers are written in full (no 1e+06)")
comp = {"kg": {"PET": 1000000.0}}
opt = {"plan": {"PET": {"combust": 1000000.0}}, "saved_kgco2e": -1350000.0, "notes": []}
p = facts_of(make_prompt("b", comp, opt)) + ex.generate_fallback(opt["saved_kgco2e"], opt["plan"], opt["notes"])
if "e+" in p:
    bad("a number was written like 1e+06")
else:
    good()

step("use_local_model=False returns the fallback instantly")
comp, opt = SCENARIOS["recycling full"]
t = time.time()
text, secs = ex.generate_explanation(make_prompt("b", comp, opt), opt, use_local_model=False)
if text != ex.generate_fallback(opt["saved_kgco2e"], opt["plan"], opt["notes"]) or time.time() - t > 0.5:
    bad("expected the fallback, instantly")
else:
    good()

step("Missing model file: falls back without crashing")
text, secs = ex.generate_explanation(make_prompt("b", comp, opt), opt, model_path="/does/not/exist.gguf")
if text != ex.generate_fallback(opt["saved_kgco2e"], opt["plan"], opt["notes"]):
    bad("expected the fallback text")
else:
    good("(the [LLM Warning] line above is expected)")

# ============================================================= B. with the model
print("\n\nB. WITH THE GRANITE MODEL")

step("llama-cpp-python is installed")
try:
    import llama_cpp
except Exception as e:  # ImportError, or a missing shared library / dependency
    print(f"    STOP llama-cpp-python can't be imported by this Python: {type(e).__name__}: {e}")
    print(f"         this Python: {sys.executable}")
    print("    FIX  install it into exactly this Python:")
    print(f"         {sys.executable} -m pip install llama-cpp-python")
    print("         (builds from source; needs: sudo dnf install cmake gcc-c++ python3-devel)")
    summary(ran_model=False)
good(f"version {getattr(llama_cpp, '__version__', '?')}")

step("Model file is in place")
here = Path(ex.__file__).resolve().parent
candidates = [os.environ.get("POLYROUTE_GRANITE_GGUF"), str(here / "granite-4.0-h-instruct.gguf"),
              "./granite-4.0-h-instruct.gguf"]
MODEL = next((c for c in candidates if c and os.path.exists(c)), None)
if not MODEL:
    print("    STOP model not found. Looked in:")
    for c in candidates:
        if c:
            print(f"         {c}")
    print("    FIX  put the .gguf in one of those places, or set POLYROUTE_GRANITE_GGUF=/path/to/file.gguf")
    summary(ran_model=False)
good(f"{MODEL} ({os.path.getsize(MODEL) / 1e9:.2f} GB)")
if MODEL != str(here / "granite-4.0-h-instruct.gguf") and not os.environ.get("POLYROUTE_GRANITE_GGUF"):
    print("    INFO the dashboard looks for src/explainer/granite-4.0-h-instruct.gguf (or POLYROUTE_GRANITE_GGUF)")

step("Enough free memory to load the model")
def available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        return None
avail = available_gb()
model_gb = os.path.getsize(MODEL) / 1e9
need = model_gb + 1.5  # model + working memory for the answer + Python
if avail is None:
    print("    INFO can't read free memory on this system; continuing")
else:
    print(f"    free memory {avail:.1f} GB, model {model_gb:.1f} GB, needs about {need:.1f} GB")
    if avail < need and "--force" not in sys.argv:
        print("    STOP not enough free memory: Linux would kill the process ('Killed').")
        print("    FIX  close the browser and other big apps, then run again (check with: free -h)")
        print("         or use a smaller Granite file; add --force to try anyway")
        summary(ran_model=False)
    good()

step("Where the time goes: loading the model vs writing the answer")
comp, opt = SCENARIOS["recycling full"]
prompt = make_prompt("b", comp, opt)
fb = ex.generate_fallback(opt["saved_kgco2e"], opt["plan"], opt["notes"])
# Uses explainer.py itself, so only ONE copy of the model is ever in memory.
print("    request 1 (loads the model, then writes)...", flush=True)
t = time.time()
first, _ = ex.generate_explanation(prompt, opt, model_path=MODEL)
t1 = time.time() - t
if first == fb:
    bad("Granite didn't answer (got the fallback): see the [LLM Warning] line above. Common causes: "
        "llama-cpp-python too old for this model, or (on LinuxONE) a little-endian GGUF.")
    summary()
print("    request 2 (writes only, if the model is reused)...", flush=True)
t = time.time()
second, _ = ex.generate_explanation(prompt, opt, model_path=MODEL)
t2 = time.time() - t
warns_before = len(WARNS)
if CACHED:
    load_s, gen_s = max(t1 - t2, 0.0), t2
    print(f"    load:  {load_s:6.1f} s  (first request only)")
    print(f"    write: {gen_s:6.1f} s  (every request)")
else:
    load_s, gen_s = None, t2
    print(f"    each request: {t1:.1f} s and {t2:.1f} s (model reloaded both times)")
    warn(f"explainer.py reloads the model on every request (~{t2:.0f} s each); "
         "loading it once and reusing it removes the loading part")
per_request = gen_s
print(f"    => about {per_request:.0f} s per request after the first, on this machine ({os.cpu_count()} CPU threads)")
if gen_s > 20:
    warn(f"writing an answer takes {gen_s:.0f} s here; the 3-core LinuxONE VM will be slower. "
         "A smaller Granite file would be faster and use less memory")
if first != second:
    print("    INFO the two answers differ, so a rerun for the video may not match")
if len(WARNS) == warns_before:
    good()

step("Granite explains each scenario: real numbers only, right length, right direction")
used = 0
chosen = {k: v for k, v in SCENARIOS.items()
          if not QUICK or k in ("recycling full", "landfill overflow (carbon cost)")}
for n_done, (name, (comp, opt)) in enumerate(chosen.items(), 1):
    print(f"    running {n_done}/{len(chosen)}: {name} (about {per_request:.0f} s)...", flush=True)
    prompt = make_prompt(name, comp, opt)
    fb = ex.generate_fallback(opt["saved_kgco2e"], opt["plan"], opt["notes"])
    text, secs = ex.generate_explanation(prompt, opt, model_path=MODEL)
    from_model = text != fb
    used += from_model
    print(f"\n    --- {name} ({secs} s, {'Granite' if from_model else 'fallback'})")
    print(indent(text))
    if not from_model:
        continue
    made_up = invented_numbers(text, facts_of(prompt))
    if made_up:
        warn(f"{name}: Granite wrote numbers that aren't in the results: {made_up}")
    if re.search(COMPARISONS, text, re.I):
        warn(f"{name}: adds a comparison that isn't in the results "
             f"('{re.search(COMPARISONS, text, re.I).group(0)}')")
    n = len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])
    if not 3 <= n <= 5:
        print(f"    INFO {n} sentences (asked for 3 to 5)")
    if opt["saved_kgco2e"] < 0 and re.search(r"\bsav(e|es|ed|ing)\b", text, re.I) and \
            not re.search(r"\b(more|emit|emits|increase|cost|adds)\b", text, re.I):
        warn(f"{name}: says the plan saves carbon, but it costs {-opt['saved_kgco2e']:g} kg CO2e")
    if any(tag in text for tag in ["<|", "|>"]):
        warn(f"{name}: chat tags leaked into the answer")
print()
if used == 0:
    bad("Granite never answered")
else:
    good(f"Granite answered {used}/{len(chosen)}")

step("Empty plan: 'no plastic' message, without asking the model")
comp, opt = EMPTY
print("    running...", flush=True)
t = time.time()
text, _ = ex.generate_explanation(make_prompt("b", comp, opt), opt, model_path=MODEL)
spent = time.time() - t
print(indent(f"({spent:.1f} s) {text}"))
if "No plastic" in text:
    good()
else:
    warn(f"an empty batch was sent to Granite ({spent:.0f} s) instead of the fixed 'no plastic' message")

summary()