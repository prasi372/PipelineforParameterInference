"""
Usage: python3 extract_model_likelihood.py <output.json> [<input.db,>]
"""
import sys
import json
import pyabc
import parse
import numpy as np

from pyabc.visualization.kde import kde_2d
from scipy.interpolate import griddata


MODELS = ['WMM', 'CBM', 'smoldyn']

base_parameters = {
    'k_a': 0.002,
    'k_d': 0.1,
    'mu': 3.0,
    'kappa': 1.0,
    'gamma': 0.04,
    'diffusion': 0.6,
    'time_step': 0.1,
    'cell_radius': 6.0,
    'nucleus_radius': 2.5,
}

def log_value(p, name=None):
    return (np.log10(base_parameters[name]) if name else 0) + p * np.log10(2)

def extract_params(inf_file):
    try:
        raw_params = parse.parse("inference_({D:f},{chi:f},{k_d:f})_{model}.db", inf_file).named
    except AttributeError:
        raw_params = parse.parse("{}/inference_({D:f},{chi:f},{k_d:f})_{model}.db", inf_file).named
    params = tuple([raw_params[k] for k in ['D', 'chi', 'k_d']])
    model = raw_params['model']
    return (model, params)

def extract_model_expvalue(inf_file):
    model, params = extract_params(inf_file)
    db_path = "sqlite:///{}".format(inf_file)
    history = pyabc.History(db_path)

    df, w = history.get_distribution()
    true_params = np.array([2**params[1], base_parameters['diffusion']*2**params[0]])
    particles = 2**df.to_numpy()\
            * np.array([[1, base_parameters['diffusion']]])

    w /= (base_parameters['diffusion']*np.log(2)**2*2**(particles.sum(axis=1)))
    w /= w.sum()

    return (
            model,
            str(params),
            np.average(
                np.linalg.norm(particles - true_params, axis=1)/np.linalg.norm(true_params),
                weights=w, axis=0).tolist(),
            )

expvalue_map = {m: {} for m in MODELS}
for model, params, expvalue in [extract_model_expvalue(inf_file) for inf_file in sys.argv[2:]]:
    expvalue_map[model][params] = expvalue

for m in expvalue_map:
    assert len(expvalue_map[m]) == 256

with open(sys.argv[1], 'w') as f:
    json.dump(expvalue_map, f)
