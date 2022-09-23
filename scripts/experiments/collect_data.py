"""
Collect data from several files into a single one (first argument).

Usage
-----
`python3 collect_data.py output.json input*.json`
"""

import sys
import json

data = {}

for filename in sys.argv[2:]:
    with open(filename, 'r') as f:
        for (model, params, partial_data) in json.load(f):
            data.setdefault(model, []).append((params, partial_data))

with open(sys.argv[1], 'w') as f:
    json.dump(data, f)
