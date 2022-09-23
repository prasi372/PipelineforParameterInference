import sys
import json
from ast import literal_eval

with open('data.json', 'r') as f:
    params = [
            params
            for (solver, params) in [literal_eval(k) for k in json.load(f).keys()]
            if solver == 'smoldyn' and params[2] == 0.0]

for param in params:
    str_param = str(param).replace(' ', '')
    print(f'sbatch launch_inference.sh "gaussabc_default_4eps/inference_{str_param}.db" "{str_param}"')
