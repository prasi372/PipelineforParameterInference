"""
Usage: python3 extract_model_prob.py <output.json> [<input.db,>]
"""
import sys
import json
import pyabc
import parse
import numpy as np


MODELS = ['WMM', 'CBM', 'smoldyn']

def extract_params(inf_file):
    try:
        raw_params = parse.parse("inference_({D:f},{chi:f},{k_d:f})_All.db", inf_file).named
    except AttributeError:
        raw_params = parse.parse("{}/inference_({D:f},{chi:f},{k_d:f})_All.db", inf_file).named
    return tuple([raw_params[k] for k in ['D', 'chi', 'k_d']])

def extract_model_probs(inf_file):
    params = extract_params(inf_file)
    db_path = "sqlite:///{}".format(inf_file)
    history = pyabc.History(db_path)
    mod_prob = {MODELS[i_mod]: p
        for (i_mod, p) in history
            .get_population()
            .get_model_probabilities().itertuples()}

    if not mod_prob:
        print(f"Missing probabilities for {params}, possibly because job timed out.", file=sys.stderr)

    return (
            str(params),
            mod_prob,
            )

prob_map = dict([extract_model_probs(inf_file) for inf_file in sys.argv[2:]])

assert len(prob_map) == 256

with open(sys.argv[1], 'w') as f:
    json.dump(prob_map, f)
