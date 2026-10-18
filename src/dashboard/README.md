# dashboard – PolyRoute web app

Two parts:

| Part | Folder | Tech | Job |
|---|---|---|---|
| Backend | `src/dashboard/*.py` | Flask + Flask-CORS | Runs the whole chain (composition estimator → optimiser → explainer) and returns JSON |
| Frontend | `src/dashboard/frontend/` | React, TanStack Start, Vite, Tailwind CSS (scaffolded with Lovable) | The interactive page; calls the backend at `http://127.0.0.1:8501` |

The backend also serves a plain HTML version of the page (`templates/`, `static/`) at `/`, which needs no Node.js.

```
browser ─► frontend (React, :5173 or :8080) ─► POST /api/analyse ─► backend (Flask, :8501)
                                                                      ├─ comp_analyzer
                                                                      ├─ optimiser   (PuLP + GLPK, EPA WARM v16)
                                                                      └─ explainer   (IBM Granite, or template)
```

## Run it

**1. Backend** (from `src/`):
```bash
pip install -r ../requirements.txt   # once
python -m dashboard.app             # http://127.0.0.1:8501  (plain HTML page at /)
```
Batches come from the real classifier: saved outputs in `classifying_plastics/data/` (e.g. `predictions.json`, its
491 test photos), or photos uploaded on the page, which are classified with
`classifying_plastics/models/best_polymer_model_tf.keras` (in the repository).
Optional: `export POLYROUTE_GRANITE_GGUF=/path/to/granite.gguf` to use Granite for the explanation.

**2. Frontend** (in another terminal; needs [Bun](https://bun.com), e.g. `curl -fsSL https://bun.sh/install | bash`):
```bash
cd src/dashboard/frontend
bun install                          # if it fails downloading @lovable.dev/...: rm bun.lock && bun install
bun run dev                          # open the URL it prints
```
`npm install` fails on this project with an npm bug ("Cannot read properties of null (reading 'edgesOut')").
The page shows **LIVE BACKEND** when it reaches the backend, and **DEMO WORKSPACE** with illustrative values when it can't.

## On IBM LinuxONE

Run the backend on the LinuxONE VM and open it through an SSH tunnel; the frontend can run on a laptop, since it
calls `127.0.0.1:8501`, which the tunnel forwards to the VM:
```bash
# on the LinuxOne VM (inside the app container), from src/:
POLYROUTE_HOST=0.0.0.0 POLYROUTE_THREADS=3 python -m dashboard.app

ssh -L 8501:localhost:8501 linuxone
```
`/api/samples` reports `"machine": "s390x"` and the plain HTML page's footer shows "IBM LinuxONE (s390x)".

## API

`GET /api/samples` → `{"samples": [...], "defaults": {"recycle_cap_kg", "compost_cap_kg", "landfill_cap_kg"}, "machine": "s390x"}`

`POST /api/analyse` (form data or JSON):

| Field | Values |
|---|---|
| `photos` (one or more images), or `sample_file`, or `classifier_file` | photos to classify, a saved classifier output from `/api/samples`, or an uploaded classifier JSON |
| `weight_mode` | `measured` (+ `total_kg`), `volume` (+ `volume_m3`), `count` (+ `item_count`), `photos` |
| `recycle_cap_kg`, `compost_cap_kg`, `landfill_cap_kg` | facility limits in kg (defaults 40 / 10 / 15) |

Returns the view model: `lanes` (per polymer: kg, 95% range, route split), `carbon` (landfill vs plan),
`saved` (kg CO2e; **negative = the plan costs carbon**), `notes`, `explanation`, `explained_by`
(`granite` / `template`), `placeholders` (demo values in use), `result` (full estimator output).
Errors return `{"error": "..."}` with status 400.

## Files

| File | Job |
|---|---|
| `app.py` | Flask routes: JSON API (`/api/samples`, `/api/analyse`) and the plain HTML page (`/`, `/analyse`) |
| `pipeline.py` | Reads the inputs, runs comp_analyzer → optimiser → explainer, builds the view model |
| `integrations.py` | The only place that calls the other modules (classifier, optimiser, explainer); runs one Granite request at a time and reuses answers for an identical plan |
| `templates/`, `static/` | Plain HTML page |
| `frontend/` | React frontend |