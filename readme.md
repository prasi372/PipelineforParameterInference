# Problems with pymc3

* takes a lot of time (although accuracy seems ok)
* unclear how the distance is computed
* unclear how to use summary statistics
* unclear how to use custom distance
* unclear how to bring computation time down (maybe I don't need that much accuracy)

# pyABC?

* Good gamma estimates
* y_observed is not enough to estimate both gamma & kappa

# Rackham
* Setup python: https://www.uppmax.uu.se/support/user-guides/python-modules-guide/
* launch job: sbatch -p core -n 1 -t 72:00:00 -A snic2019-8-227 launch.sh 3.0 0.0

# How to use `collect_data.py`
```
python3 collect_data.py data/diyabc.json data/diyabc/*
```
