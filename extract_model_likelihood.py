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

def extract_params(inf_file):
    try:
        raw_params = parse.parse("inference_({D:f},{chi:f},{k_d:f})_{model}.db", inf_file).named
    except AttributeError:
        raw_params = parse.parse("{}/inference_({D:f},{chi:f},{k_d:f})_{model}.db", inf_file).named
    params = tuple([raw_params[k] for k in ['D', 'chi', 'k_d']])
    model = raw_params['model']
    return (model, params)

def extract_model_likelihood(inf_file):
    model, params = extract_params(inf_file)
    db_path = "sqlite:///{}".format(inf_file)
    history = pyabc.History(db_path)

    df, w = history.get_distribution()
    X, Y, PDF = kde_2d(df, w, "diffusion", "chi")
    XY = np.concatenate((X.reshape((-1, 1)), Y.reshape((-1, 1))), axis=1)
    lklh = float(griddata(XY, PDF.reshape((-1, 1)), params[:2], method='nearest'))
    assert lklh >= 0.

    return (
            model,
            str(params),
            lklh,
            )

lklh_map = {m: {} for m in MODELS}
for model, params, lklh in [extract_model_likelihood(inf_file) for inf_file in sys.argv[2:]]:
    lklh_map[model][params] = lklh

for m in lklh_map:
    assert len(lklh_map[m]) == 256


with open(sys.argv[1], 'w') as f:
    json.dump(lklh_map, f)
