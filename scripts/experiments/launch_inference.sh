#!/bin/bash
#SBATCH -p core -n 2 -t 1:00:00 -A snic2019-8-227
module load gcc/9.2

python run_inference.py data.json ${1} ${2}
