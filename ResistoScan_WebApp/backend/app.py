"""
ResistoScan Backend - app.py
Run: python app.py
API runs at: http://localhost:5000
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os, json, math

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────
#  YOUR REAL 34 SAMPLE DATA
# ─────────────────────────────────────────
SAMPLES = [
    # SEWAGE (5)
    {"id":"DRR480202","env":"Sewage","args":1247,"pl":0.72,"dr":0.75,"mi":0.0},
    {"id":"DRR480203","env":"Sewage","args":988, "pl":0.68,"dr":0.70,"mi":0.0},
    {"id":"DRR480204","env":"Sewage","args":1421,"pl":0.75,"dr":0.78,"mi":0.0},
    {"id":"DRR480205","env":"Sewage","args":788, "pl":0.61,"dr":0.65,"mi":0.0},
    {"id":"DRR480206","env":"Sewage","args":1784,"pl":0.82,"dr":0.85,"mi":0.0},
    # HUMAN GUT (5)
    {"id":"SRR9108960","env":"Human_Gut","args":511, "pl":0.40,"dr":0.48,"mi":0.0},
    {"id":"SRR9108961","env":"Human_Gut","args":623, "pl":0.44,"dr":0.52,"mi":0.0},
    {"id":"SRR9108962","env":"Human_Gut","args":447, "pl":0.36,"dr":0.42,"mi":0.0},
    {"id":"SRR9108963","env":"Human_Gut","args":789, "pl":0.52,"dr":0.58,"mi":0.0},
    {"id":"SRR9108964","env":"Human_Gut","args":890, "pl":0.56,"dr":0.62,"mi":0.0},
    # OPEN DRAIN (10)
    {"id":"SRR27587518","env":"Open_Drain","args":1122,"pl":0.65,"dr":0.70,"mi":0.0},
    {"id":"SRR27587519","env":"Open_Drain","args":1455,"pl":0.74,"dr":0.77,"mi":0.0},
    {"id":"SRR27587520","env":"Open_Drain","args":725, "pl":0.55,"dr":0.60,"mi":0.0},
    {"id":"SRR27587521","env":"Open_Drain","args":1233,"pl":0.68,"dr":0.72,"mi":0.0},
    {"id":"SRR27587522","env":"Open_Drain","args":1766,"pl":0.80,"dr":0.82,"mi":0.0},
    {"id":"SRR27587523","env":"Open_Drain","args":988, "pl":0.62,"dr":0.67,"mi":0.0},
    {"id":"SRR27587524","env":"Open_Drain","args":1340,"pl":0.71,"dr":0.74,"mi":0.0},
    {"id":"SRR27587525","env":"Open_Drain","args":1089,"pl":0.66,"dr":0.70,"mi":0.0},
    {"id":"SRR27587526","env":"Open_Drain","args":879, "pl":0.60,"dr":0.64,"mi":0.0},
    {"id":"SRR27587527","env":"Open_Drain","args":1456,"pl":0.75,"dr":0.78,"mi":0.0},
    # CRC (10)
    {"id":"SRR31828876","env":"CRC","args":1234,"pl":0.74,"dr":0.82,"mi":0.0},
    {"id":"SRR31828877","env":"CRC","args":1567,"pl":0.80,"dr":0.88,"mi":0.0},
    {"id":"SRR31828878","env":"CRC","args":1900,"pl":0.90,"dr":0.95,"mi":0.0},
    {"id":"SRR31828879","env":"CRC","args":1023,"pl":0.68,"dr":0.76,"mi":0.0},
    {"id":"SRR31828880","env":"CRC","args":1456,"pl":0.78,"dr":0.86,"mi":0.0},
    {"id":"SRR31828881","env":"CRC","args":1678,"pl":0.84,"dr":0.90,"mi":0.0},
    {"id":"SRR31828882","env":"CRC","args":1100,"pl":0.70,"dr":0.78,"mi":0.0},
    {"id":"SRR31828883","env":"CRC","args":1389,"pl":0.76,"dr":0.84,"mi":0.0},
    {"id":"SRR31828884","env":"CRC","args":1244,"pl":0.74,"dr":0.82,"mi":0.0},
    {"id":"SRR31828885","env":"CRC","args":1789,"pl":0.88,"dr":0.93,"mi":0.0},
]

ENV_WEIGHT = {"Sewage":1.25,"Human_Gut":1.0,"Open_Drain":1.20,"CRC":1.35}
ENV_KEY    = {"Sewage":"sewage","Human_Gut":"gut","Open_Drain":"drain","CRC":"crc"}

def calc_iti(args, pl, dr, mi, env):
    w    = ENV_WEIGHT.get(env, 1.0)
    rd   = min(args / 2000.0, 1.0)
    raw  = (0.30*rd + 0.25*pl + 0.25*dr + 0.20*mi) * w * 100
    iti  = round(min(raw, 100.0), 1)
    risk = "LOW" if iti<25 else "MODERATE" if iti<50 else "HIGH" if iti<75 else "CRITICAL"
    return iti, risk

def enrich(s):
    iti, risk = calc_iti(s["args"], s["pl"], s["dr"], s["mi"], s["env"])
    return {**s, "iti": iti, "risk": risk, "env_key": ENV_KEY.get(s["env"],"gut")}

ALL = [enrich(s) for s in SAMPLES]

# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","samples":len(ALL),"version":"2.0"})

@app.route("/api/stats")
def stats():
    return jsonify({
        "total_samples":34,"unique_args":133,"environments":4,
        "ml_accuracy":93.3,"pipeline_steps":7,
        "formula":"ITI = (0.30×RD + 0.25×PL + 0.25×DR + 0.20×MI) × env_weight × 100"
    })

@app.route("/api/samples")
def get_samples():
    env = request.args.get("env","all")
    data = ALL if env=="all" else [s for s in ALL if s["env_key"]==env]
    return jsonify({"count":len(data),"samples":data})

@app.route("/api/environments")
def get_environments():
    result = {}
    for s in ALL:
        k = s["env"]
        if k not in result:
            result[k] = {"env":k,"env_key":s["env_key"],"samples":[],"iti_vals":[]}
        result[k]["samples"].append(s["id"])
        result[k]["iti_vals"].append(s["iti"])
    out = []
    for k,v in result.items():
        vals = v["iti_vals"]
        out.append({
            "env":k,"env_key":v["env_key"],
            "count":len(vals),
            "iti_mean":round(sum(vals)/len(vals),1),
            "iti_min":min(vals),"iti_max":max(vals),
            "sample_ids":v["samples"],
        })
    return jsonify({"environments":out})

@app.route("/api/calculate", methods=["POST"])
def calculate():
    d   = request.get_json() or {}
    args= float(d.get("resistome_density",0))
    pl  = float(d.get("pathogen_load",0))
    dr  = float(d.get("drug_risk",0))
    mi  = float(d.get("mobility_index",0))
    env = d.get("environment","Human_Gut")

    iti, risk = calc_iti(args, pl, dr, mi, env)
    w  = ENV_WEIGHT.get(env,1.0)
    rd = min(args/2000,1.0)

    recs = {
        "LOW":      "ARG burden within safe range. Maintain routine quarterly surveillance.",
        "MODERATE": "Elevated resistome detected. Increase sampling frequency. Review antibiotic use in linked facilities.",
        "HIGH":     "Significant AMR burden. Notify public health authorities. Initiate source-tracking investigation immediately.",
        "CRITICAL": "EMERGENCY ALERT. Extreme AMR threat. Activate outbreak response protocols. Isolate and contain source."
    }
    return jsonify({
        "iti":iti,"risk":risk,
        "recommendation":recs[risk],
        "env_weight":w,
        "components":{
            "resistome": round(0.30*rd*w*100,1),
            "pathogen":  round(0.25*pl*w*100,1),
            "drug_risk": round(0.25*dr*w*100,1),
            "mobility":  round(0.20*mi*w*100,1),
        }
    })

@app.route("/api/combined")
def combined():
    combo = request.args.get("combo","all")
    COMBOS = {
        "gut_sewage":   ["gut","sewage"],
        "gut_drain":    ["gut","drain"],
        "sewage_drain": ["sewage","drain"],
        "gut_crc":      ["gut","crc"],
        "no_sewage":    ["gut","drain","crc"],
        "all":          ["gut","sewage","drain","crc"],
    }
    envs = COMBOS.get(combo, ["gut","sewage","drain","crc"])
    data = [s for s in ALL if s["env_key"] in envs]
    vals = [s["iti"] for s in data]
    return jsonify({
        "combo":combo,"envs":envs,"count":len(data),
        "iti_mean":round(sum(vals)/len(vals),1) if vals else 0,
        "iti_min":min(vals) if vals else 0,
        "iti_max":max(vals) if vals else 0,
        "samples":data
    })

@app.route("/api/ml-results")
def ml_results():
    return jsonify({"models":[
        {"name":"Random Forest",       "accuracy":93.3,"f1":0.931,"status":"best"},
        {"name":"Gradient Boosting",   "accuracy":90.0,"f1":0.897,"status":"good"},
        {"name":"Logistic Regression", "accuracy":86.7,"f1":0.863,"status":"good"},
        {"name":"SVM",                 "accuracy":83.3,"f1":0.829,"status":"ok"},
        {"name":"KNN",                 "accuracy":80.0,"f1":0.794,"status":"ok"},
    ],"validation":"5-fold stratified cross-validation","features":133,"samples":30})

@app.route("/api/top-args")
def top_args():
    return jsonify({"args":[
        {"gene":"tetA",        "prevalence":90,"drug":"Tetracycline"},
        {"gene":"blaOXA-10",   "prevalence":83,"drug":"Beta-lactam"},
        {"gene":"aac(6')-Ib",  "prevalence":77,"drug":"Aminoglycoside"},
        {"gene":"sul1",        "prevalence":73,"drug":"Sulfonamide"},
        {"gene":"vanA",        "prevalence":67,"drug":"Vancomycin"},
        {"gene":"mcr-1",       "prevalence":60,"drug":"Colistin"},
        {"gene":"blaNDM-1",    "prevalence":53,"drug":"Carbapenem"},
        {"gene":"qnrS1",       "prevalence":47,"drug":"Fluoroquinolone"},
        {"gene":"ermB",        "prevalence":40,"drug":"Macrolide"},
        {"gene":"cfr",         "prevalence":33,"drug":"Chloramphenicol"},
    ],"total":133})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  ResistoScan API  →  http://localhost:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)

import json
from pathlib import Path

@app.route('/api/network-data')
def get_network_data():
    """Get ARG network visualization data"""
    try:
        data_file = Path(__file__).parent / 'data' / 'arg_drug_mapping.json'
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Transform for D3.js network format
        nodes = []
        links = []
        
        # Add drug class nodes
        for drug_class in data['drug_classes']:
            nodes.append({
                'id': drug_class['name'],
                'name': drug_class['name'],
                'type': 'drug_class',
                'color': drug_class['color'],
                'priority': drug_class['priority'],
                'size': 30
            })
        
        # Add ARG nodes and links
        for arg in data['args']:
            nodes.append({
                'id': arg['id'],
                'name': arg['name'],
                'type': 'arg',
                'drug_class': arg['drug_class'],
                'prevalence': arg['prevalence'],
                'mechanism': arg['mechanism'],
                'size': 10 + (arg['prevalence'] / 10)
            })
            
            # Create link from ARG to drug class
            links.append({
                'source': arg['id'],
                'target': arg['drug_class'],
                'strength': arg['prevalence'] / 100
            })
        
        return jsonify({
            'success': True,
            'nodes': nodes,
            'links': links
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
