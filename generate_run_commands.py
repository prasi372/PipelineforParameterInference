import sys
import parse
import numpy as np

#done = [
#    parse.parse("data/smoldyn_data({:f},{:f}).json", s).fixed
#    for s in sys.argv[1:]]

done = []

param_cd = {
        (c, d)
        for c in [0., 3.]
        for d in [-3., 0.]
        #for c in np.linspace(-8, 4, num=15)
        #for d in np.linspace(-8, 4, num=15)
        if (c, d) not in done}

for c, d in param_cd:
    print("sbatch -p core -n 1 -t 72:00:00 -A snic2019-8-227 launch.sh {} {}".format(c, d))
