"""
Generate data for DIYABC within the defined bounds, using a loguniform distribution

args:
    1. destination folder
    2. 0: WMM, 1: CBM
    3. number of runs

"""

import sys
import uuid

import json
import scipy.stats as st

from models import neg_feedback_WMM, neg_feedback_CBM

if len(sys.argv) <= 1:
    raise Exception("Output destination missing")
if len(sys.argv) <= 2:
    raise Exception("Model selection missing")
if len(sys.argv) <= 3:
    raise Exception("Number of runs missing")

model = [neg_feedback_WMM, neg_feedback_CBM][int(sys.argv[2])]

bounds = {
    "diffusion": (0.6*2**-8, 0.6*2**4),
    "k_d": (0.1*2**-3, 0.1*2**3),
}

filename = "{}/{}".format(sys.argv[1], str(uuid.uuid4()))

data = []
for _ in range(int(sys.argv[3])):
    params = {param: st.loguniform(p_min, p_max).rvs(size=1)[0]
              for param, (p_min, p_max) in bounds.items()}
    data.append((model.__name__, params, model(params)))

with open(filename, 'w') as f:
    print("Saving: {}".format(filename))
    json.dump(data, f)
