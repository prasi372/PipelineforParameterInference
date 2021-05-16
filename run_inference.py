"""
Usage: python3 run_inference.py <data.json> <setup.json> <true_params>

"""

import sys

import json
import pyabc
import numpy as np
import numpy.random as npr
import scipy.stats as st

from ast import literal_eval

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


if len(sys.argv) <= 3:
    raise Exception("Input files missing")

with open(sys.argv[2], 'r') as setup_file:
    setup = json.load(setup_file)

print(f"Inference for {sys.argv[2]}, {sys.argv[3]}")

EPSILON = 0
MAXPOP = 8
POPSIZE = 250


base_parameters = {
    'k_a': 0.00166,
    'k_d': 0.1,
    'mu': 3.0,
    'kappa': 1.0,
    'gamma': 0.04,
    'diffusion': 0.6,
    'time_step': 0.32,
    'cell_radius': 6.0,
    'nucleus_radius': 2.5,
}

INIT_STOP = 200
TSTOP = 1200 # The 200 first minutes will be discarded ad burn-in
NTRAJ = 64
NPOINTS = 120
THRES = 0.1
BURNIN = 20

SPECIES = setup['species']

def neg_feedback_WMM(params):
    return {'params': [params['chi'], params['diffusion']], 'solver': 'WMM'}

def neg_feedback_CBM(params):
    return {'params': [params['chi'], params['diffusion']], 'solver': 'CBM'}

def neg_feedback_smoldyn(params):
    return {'params': [params['chi'], params['diffusion']], 'solver': 'smoldyn'}

models = [neg_feedback_WMM, neg_feedback_CBM, neg_feedback_smoldyn]

bounds = {
    "chi": (-2, 4),
    "diffusion": (-8, 4),
}

priors = [
    pyabc.Distribution(
        **{param: pyabc.RV("uniform", p_min, p_max - p_min)
        #**{param: pyabc.RV("loguniform", p_min, p_max)
          for param, (p_min, p_max) in bounds.items()}
    )
    for _ in models
]

if setup["dist"] == "ss":
    summary_statistics_raw = {
            'mean': lambda x: np.mean(x, axis=1).mean(),
            'std': lambda x: np.std(x, axis=1).mean(),
            'min': lambda x: np.min(x, axis=1).mean(),
            'max': lambda x: np.max(x, axis=1).mean(),
            }

    stat_mean = {'meanP': 1042.154285695293, 'meanRNA': 61.844218439788314, 'stdP': 255.5303176575153, 'stdRNA': 15.336083154554037, 'minP': 467.9661153157552, 'minRNA': 25.877583821614582, 'maxP': 1496.834452311198, 'maxRNA': 91.36317952473958}
    stat_std = {'meanP': 526.6189072354466, 'meanRNA': 11.768310207409813, 'stdP': 187.71013670846963, 'stdRNA': 6.797565421631525, 'minP': 443.51772696920216, 'minRNA': 20.602305740635945, 'maxP': 709.1671367023749, 'maxRNA': 4.990071149032393}


    summary_statistics = {
                k + species: lambda x: (s(x) - stat_mean[k + species])/stat_std[k + species]
                for k, s in summary_statistics_raw.items()
                for species in SPECIES
            }

    dist = lambda x,y: (
            (np.array([summary_statistics[k + species](x[species])
                       for k in summary_statistics_raw.keys()
                       for species in SPECIES])
            - (np.array([summary_statistics[k + species](y[species])
                       for k in summary_statistics_raw.keys()
                       for species in SPECIES]))
        )**2).sum()**0.5
elif setup["dist"] == "kg":
    dist = lambda x,y: (
        np.mean([
            [st.ks_2samp(x[species][:, t], y[species][:, t])[0] for t in range(x[species].shape[1])]
            for species in SPECIES
        ]))
elif setup["dist"] == "ssAdv":
    import tsfresh.feature_extraction.feature_calculators as tsfe
    summary_statistics_raw = {
            'longest_strike_below_mean': lambda x: np.apply_along_axis(tsfe.longest_strike_below_mean, 0, x).mean(),
            'longest_strike_above_mean': lambda x: np.apply_along_axis(tsfe.longest_strike_above_mean, 0, x).mean(),
            'mean_abs_change': lambda x: np.apply_along_axis(tsfe.mean_abs_change, 0, x).mean(),
            'maximum': lambda x: np.apply_along_axis(tsfe.maximum, 0, x).mean(),
            'minimum': lambda x: np.apply_along_axis(tsfe.minimum, 0, x).mean(),
            'variance': lambda x: np.apply_along_axis(tsfe.variance, 0, x).mean(),
            }

    stat_mean = {'longest_strike_below_meanP': 5.2377011138613865, 'longest_strike_below_meanRNA': 4.793258818069307, 'longest_strike_above_meanP': 5.651931208745875, 'longest_strike_above_meanRNA': 6.038321214933994, 'mean_abs_changeP': 297.93370793552566, 'mean_abs_changeRNA': 17.48370513204594, 'maximumP': 1531.3910891089108, 'maximumRNA': 91.14778645833333, 'minimumP': 411.66921281971946, 'minimumRNA': 24.466506806930692, 'varianceP': 108241.7390247367, 'varianceRNA': 297.064633632257}
    stat_std = {'longest_strike_below_meanP': 2.4667828386324206, 'longest_strike_below_meanRNA': 0.7026448914120297, 'longest_strike_above_meanP': 1.399528430327171, 'longest_strike_above_meanRNA': 1.0580060990298705, 'mean_abs_changeP': 212.0608104949187, 'mean_abs_changeRNA': 7.600031842541941, 'maximumP': 722.3805712604964, 'maximumRNA': 4.232712083773925, 'minimumP': 419.6285935725412, 'minimumRNA': 20.689287835317344, 'varianceP': 132810.87225382286, 'varianceRNA': 253.89911336323922}

    summary_statistics = {
                k + species: lambda x: (s(x) - stat_mean[k + species])/stat_std[k + species]
                for k, s in summary_statistics_raw.items()
                for species in SPECIES
            }

    dist = lambda x,y: (
            (np.array([summary_statistics[k + species](x[species])
                       for k in summary_statistics_raw.keys()
                       for species in SPECIES])
            - (np.array([summary_statistics[k + species](y[species])
                       for k in summary_statistics_raw.keys()
                       for species in SPECIES]))
        )**2).sum()**0.5
else:
    raise NotImplementedError()


with open(sys.argv[1], 'r') as f:
    raw_data = {literal_eval(k): v for k, v in json.load(f).items()}

for key in raw_data.keys():
    solver, params = key
    timeseries = raw_data[key][1]

    if solver == 'CBM':
        for species in SPECIES:
            timeseries[species] = np.array(timeseries[species + 'nuc']) + np.array(timeseries[species + 'cyt'])
            del timeseries[species + 'nuc']
            del timeseries[species + 'cyt']

    del timeseries['Gf']
    del timeseries['Gb']

    for species in timeseries:
        timeseries[species] = np.array(timeseries[species])[::setup["traj_step"], BURNIN::setup["tsamp_step"]]

# Refactor data
data = {}
for k, v in raw_data.items():
    solver, params = k
    if params[2] != 0:
        continue
    data.setdefault(params, {})[solver] = v

synthetic_params = literal_eval(sys.argv[3])


def train_surrogate_dist(target, solver, data):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RationalQuadratic, ConstantKernel, RBF, WhiteKernel, Matern
    from sklearn import preprocessing

    x_scaler = preprocessing.MinMaxScaler()
    y_scaler = preprocessing.MinMaxScaler()


    X = [k for k in data.keys() if k[2]==0.] # Ignore k_d
    y = np.array([
         dist(data[target]['smoldyn'][1], data[tuple(k)][solver][1])
         for k in X
        ])

    kernel = RationalQuadratic() + WhiteKernel()

    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
    )

    X_raw = np.array(X)[:, :2]
    x_scaler.fit(X_raw)
    X_norm = x_scaler.transform(X_raw)

    y = np.atleast_2d(y).T
    y_scaler = y_scaler.fit(y)
    y_norm = y_scaler.transform(y)

    gp.fit(X_norm, y_norm)

    return (gp, x_scaler, y_scaler)


def measure_utility(X_train, X_test, target, solver, data):
    gp, x_scl, y_scl = train_surrogate_dist(target, solver, {k: data[k] for k in X_train})
    y_test = np.atleast_2d(np.array([
         dist(data[target]['smoldyn'][1], data[tuple(k)][solver][1])
         for k in X_test
        ])).T
    mean_raw, std_raw = gp.predict(x_scl.transform(np.array(X_test)[:, :2]), return_std=True)
    mean = y_scl.inverse_transform(mean_raw.reshape(-1, 1))
    std = y_scl.inverse_transform(std_raw.reshape(-1, 1))

    return st.truncnorm(-mean/std, np.inf, loc=mean, scale=std).logpdf(y_test).mean()
    #return abs(1 - y_gp/y_test).mean()


def cross_validation(target, solver, data):
    params = [k for k in data if k[2] == 0.0]
    npr.shuffle(params)

    relative_utility = []
    for i in range(5):
        mask = np.ones(len(params), bool)
        mask[i::5] = False

        i_train = np.array(range(len(params)))[mask].tolist()
        i_test = np.array(range(len(params)))[~mask].tolist()

        X_train = [params[i] for i in i_train]
        X_test = [params[i] for i in i_test]

        if target not in X_train:
            X_train.append(target)
            X_test.remove(target)

        relative_utility.append(measure_utility(X_train, X_test, target, solver, data))
    return np.mean(relative_utility)


gp_solvers = {
        solver: train_surrogate_dist(synthetic_params, solver, data)
        for solver in ['WMM', 'CBM', 'smoldyn']
        }

cross_val_utility = [
        cross_validation(synthetic_params, solver, data)
        for solver in ['WMM', 'CBM', 'smoldyn']
        ]

print(f"Cross validation utility: {cross_val_utility}")

def get_eps(params):
    gp, x_scl, y_scl = gp_solvers['smoldyn']
    D, chi = params[:2] #OBS! original order is (D, chi, k_d)

    mean, std = y_scl.inverse_transform(gp.predict(x_scl.transform([[D, chi]]), return_std=True))
    return st.truncnorm(-mean/std, np.inf, loc=mean, scale=std).ppf(0.5)

eps = get_eps(synthetic_params)
print(f"EPS: {eps}")

output_file = f"inference_{sys.argv[3]}"

logs = {"eps": list(eps), "cross_validation": list(cross_val_utility)}
with open(f"{setup['output']}/{output_file}_log.json", 'w') as f:
    json.dump(logs, f)

#define dist function using GP (don't forget to transform and all
#OBS! Be careful about parameter ordering! The keys are ordered as (D, chi, k_d) in the data set but as (chi, D) in pyABC
def dist_gp(x, y):
    params = x['params']
    solver = x['solver']
    gp, x_scl, y_scl = gp_solvers[solver]

    chi, D = params #pyABC ordering
    mean, std = y_scl.inverse_transform(gp.predict(x_scl.transform([[D, chi]]), return_std=True))
    res = st.truncnorm(-mean/std, np.inf, loc=mean, scale=std).rvs()
    return res

abc = pyabc.ABCSMC(models, priors, dist_gp, population_size=POPSIZE)

print("All")
db_path = f"sqlite:///{setup['output']}/{output_file}_All.db"
history = abc.new(db_path, {'params':synthetic_params})
history = abc.run(minimum_epsilon=eps, max_nr_populations=MAXPOP)

model_name = {0: 'WMM', 1: 'CBM', 2: 'smoldyn'}

for i in range(len(models)):
    print(model_name[i])
    abc = pyabc.ABCSMC(models[i], priors[i], dist_gp, population_size=POPSIZE)

    db_path = f"sqlite:///{setup['output']}/{output_file}_{model_name[i]}.db"
    history = abc.new(db_path, {'params':synthetic_params})
    history = abc.run(minimum_epsilon=EPSILON, max_nr_populations=MAXPOP)
