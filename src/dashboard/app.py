# src/dashboard/app.py
import os
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from . import integrations, pipeline

app = Flask(__name__)
# Allow requests from your React development server (Vite default is 5173 / Lovable)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

@app.get("/api/samples")
def get_samples():
    """Returns available sample profiles and default facility limits."""
    return jsonify({
        "samples": integrations.list_sample_files(),
        "defaults": pipeline.DEFAULT_LIMITS,
        "machine": pipeline.machine()
    })

@app.post("/api/analyse")
def api_analyse():
    """Runs the PolyRoute optimizer chain and returns JSON."""
    try:
        # Determine payload type (form data vs JSON body)
        form_data = request.form if request.form else (request.get_json(silent=True) or {})
        view = pipeline.run(form_data, request.files)
        return jsonify(view)
    except (ValueError, NotImplementedError) as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(
        host=os.environ.get("POLYROUTE_HOST", "127.0.0.1"),
        port=int(os.environ.get("POLYROUTE_PORT", "8501")),
        debug=True,
    )
