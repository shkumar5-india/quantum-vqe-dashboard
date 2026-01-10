from flask import Flask, jsonify, request, render_template
from functools import lru_cache

from vqe_part1 import run_small_vqe
from vqe_nh3bh3 import run_nh3bh3_vqe

app = Flask(__name__)   # ✅ MUST be named "app"


# -----------------------
# caching (speed boost)
# -----------------------
@lru_cache(maxsize=64)
def cached_part1(molecule, maxiter, depth):
    return run_small_vqe(molecule=molecule, maxiter=maxiter, depth=depth)

@lru_cache(maxsize=16)
def cached_part2(maxiter, depth):
    return run_nh3bh3_vqe(maxiter=maxiter, circuit_depth=depth)


# -----------------------
# routes
# -----------------------
@app.get("/")
def home():
    return render_template("index.html")

@app.get("/api/status")
def status():
    return jsonify({
        "status": "Running ✅",
        "service": "Quantum VQE Dashboard",
        "endpoints": [
            "/api/vqe?molecule=H2",
            "/api/vqe?molecule=LIH",
            "/api/nh3bh3"
        ]
    })

@app.get("/api/vqe")
def api_vqe():
    molecule = request.args.get("molecule", "H2").upper()
    maxiter = int(request.args.get("maxiter", 50))
    depth = int(request.args.get("depth", 1))

    if molecule not in ["H2", "LIH"]:
        return jsonify({"error": "Supported molecules are H2 and LIH"}), 400

    return jsonify(cached_part1(molecule, maxiter, depth))

@app.get("/api/nh3bh3")
def api_nh3bh3():
    maxiter = int(request.args.get("maxiter", 60))
    depth = int(request.args.get("depth", 2))
    return jsonify(cached_part2(maxiter, depth))
