import numpy as np

import WMM
import CBM

BASE_PARAMETERS = {
        'k_a': 0.00166,
        'k_d': 0.1,
        'mu': 3.0,
        'kappa': 1.0,
        'gamma': 0.04,
        'diffusion': 0.6,
        'cell_radius': 6.0,
        'nucleus_radius': 2.5,
}

INIT_STOP = 200
TSTOP = 1000
N_POINTS = 10
N_TRAJ = 128

def neg_feedback_WMM(parameters):
    # Fill in missing parameters (i.e. parameters which are not part of the
    # inference) with base values
    parameters = {
        **BASE_PARAMETERS,
        **parameters}

    import reactionrates as rr
    # Compute new rates, taking diffusion into account
    sigma = 0.01
    V_nucleus = 4*np.pi*parameters['nucleus_radius']**3/3
    k_a_wmmodel, k_d_wmmodel = rr.well_mixed_rates(
        parameters['k_a'],
        parameters['k_d'],
        sigma, 2*parameters['diffusion'],
        V_nucleus)

    parameters['k_a'] = k_a_wmmodel
    parameters['k_d'] = k_d_wmmodel

    model = WMM.Cell(parameters)

    _, initial_state = model.run({'Gf':1}, INIT_STOP, n_trajectories=1, n_points=2)
    return {"P": np.array(model.run(initial_state , TSTOP, n_trajectories=N_TRAJ, n_points=N_POINTS)['P']).tolist()}

def neg_feedback_CBM(parameters):
    # Fill in missing parameters (i.e. parameters which are not part of the
    # inference) with base values
    parameters = {
        **BASE_PARAMETERS,
        **parameters}

    import reactionrates as rr
    # Compute new rates, taking diffusion into account
    sigma = 0.01
    V_nucleus = 4*np.pi*parameters['nucleus_radius']**3/3
    k_a_wmmodel, k_d_wmmodel = rr.well_mixed_rates(
        parameters['k_a'],
        parameters['k_d'],
        sigma, 2*parameters['diffusion'],
        V_nucleus)

    kentry, kexit = rr.entry_exit_rates(
        parameters['nucleus_radius'],
        parameters['cell_radius'],
        parameters['diffusion'])

    parameters['k_a'] = k_a_wmmodel
    parameters['k_d'] = k_d_wmmodel

    parameters['k_cn'] = kentry
    parameters['k_nc'] = kexit

    model = WMM.Cell(parameters)

    _, initial_state = model.run({'Gf':1}, INIT_STOP, n_trajectories=1, n_points=2)
    return {"P": np.array(model.run(initial_state , TSTOP, n_trajectories=N_TRAJ, n_points=N_POINTS)['P']).tolist()}
