"""
Usage: python3 extract_model_utility.py <output.json> [<input.json,>]
"""
import sys
import json
import pyabc
import parse
import numpy as np


MODELS = ['WMM', 'CBM', 'smoldyn']

def extract_params(inf_file):
    try:
        raw_params = parse.parse("inference_({D:f},{chi:f},{k_d:f})_log.json", inf_file).named
    except AttributeError:
        raw_params = parse.parse("{}/inference_({D:f},{chi:f},{k_d:f})_log.json", inf_file).named
    return tuple([raw_params[k] for k in ['D', 'chi', 'k_d']])

def extract_model_utility(inf_file):
    params = extract_params(inf_file)

    with open(inf_file, 'r') as f:
        utility = dict(zip(['WMM', 'CBM', 'smoldyn'], json.load(f)["cross_validation"]))

    return (
            str(params),
            utility,
            )

utility_map = dict([extract_model_utility(inf_file) for inf_file in sys.argv[2:]])

assert len(utility_map) == 256


with open(sys.argv[1], 'w') as f:
    json.dump(utility_map, f)
