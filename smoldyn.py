"""
Smoldyn.py

This script's aim is to encapsulate Smoldyn and to make it easier to run many
trajectories and explore parameter space.

Usage:
    python3 smoldyn.py output.json
"""

import sys
import json
import time
import tempfile
import numpy as np
import subprocess as sp
import scipy.stats as scs
import numpy.random as npr

import parse

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.duration = self.end - self.start


class Mol:
    def __init__(self, name, x, y, z, **_):
        self.name = name
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z)

    def to_smoldyn_input(self):
        """ Generate smoldyn command to add one such molecule

        Returns
        -------
        str

        """
        return "mol 1 {name} {x} {y} {z}".format(**self.__dict__)


def fill_model(
        model, time_stop, time_step, initial_state, seed, n_points):
    """ Fill up model with output file and initial state and write it down
    in an input file

    Parameters
    ----------
    model: str
        smoldyn model description
    time_stop: Float
        Simulation duration
    time_step: Float
        Interval between two timesteps
    initial_state: [Mol]
        list of molecules at t=0
    n_points: int
        number of time samples

    """
    return model.format(
        seed=seed,
        time_stop=time_stop,
        time_step=time_step,
        freq_point=int(time_stop / (time_step * n_points)),
        initial_state="\n".join(
            [m.to_smoldyn_input() for m in initial_state]),
        )


def run_smoldyn(input_string, docker=None):
    """ Run Smoldyn through command line.

    Parameters
    ----------
    input_file: File
        input file describing the model
    docker: str
        name of docker container to be used

    """
    smoldyn = sp.Popen(
        (['docker', 'run', '-i', docker] if docker else [])
        + ['smoldyn', '/dev/stdin', '-tqw'],
        # t: no graphics, q: quiet mode, w: no warnings
        stdin=sp.PIPE, stdout=sp.PIPE)

    out, _ = smoldyn.communicate(input=input_string.encode('ascii'))

    return out.decode('utf-8')



def parse_history(raw_history):
    """ Parse smoldyn output from `molcount` command.

    Returns
    -------
    {Species: timeseries}

    """
    keys = raw_history[0].split()
    history = {key: [] for key in keys}
    for line in raw_history[1:]:
        for k, v in zip(keys, line.split()):
            history[k].append(float(v))

    return history


def parse_last_state(raw_data):
    """ Parse smoldyn output from `listmols` smoldyn command.

    Returns
    -------
    [Mol]

    """
    return [
        Mol(**parse.parse(
            "{name}({state}) {x:g} {y:g} {z:g} {id:d}", mol).named)
        for mol in raw_data]


def run_trajectory(
        model, time_stop, time_step, initial_state,
        seed, n_points=500, docker=None):
    """
    Run one trajectory using the given model and initial state

    Parameters
    ----------
    model: str
        smoldyn model description
    time_stop: float
        Simulation duration
    time_step: Float
        Interval between two timesteps
    initial_state: [Mol]
        list of molecules at t=0
    seed: int
        seed used to run smoldyn
    n_points: int
        number of time samples
    docker: str
        name of docker container to be used

    """
    input_string = fill_model(
        model, time_stop, time_step, initial_state,
        seed if seed is not None else npr.randint(10**9),
        n_points)

    raw_data = run_smoldyn(input_string, docker)

    # Collect results
    history, last_state = [
        e.strip().split("\n")
        for e in raw_data.split("--Simulation ends--\n")]

    return (parse_history(history), parse_last_state(last_state))


def run_trajectories(
        model, time_stop, n_trajectories, time_step, initial_state,
        seed, n_points=500, docker=None, client=None):
    """
    Run several trajectory (serially) using the given model and initial state

    Parameters
    ----------
    model: str
        smoldyn model description
    time_stop: float
        Simulation duration
    n_trajectories: int
        number of trajectories to simulate
    time_step: Float
        Interval between two timesteps
    initial_state: [Mol]
        list of molecules at t=0
    seed: int
        seed used to run smoldyn
    n_points: int
        number of time samples
    docker: str
        name of docker container to be used

    Returns
    -------
    {species: [trajectory]}

    """

    submit = (
        client.submit
        if client else
        lambda f, *args, **kwargs: f(*args, **kwargs)) # Identity function

    initial_state = client.scatter(initial_state)\
            if client else initial_state # Scatter data across workers to avoid big transfers

    submissions = [
        submit(
            run_trajectory,
            model,
            time_stop,
            time_step,
            initial_state,
            # Generate random seed to ensure simulations are different
            # even if they start at the same time
            seed=seed + i if seed is not None else npr.randint(10**9),
            n_points=n_points,
            docker=docker,
            )
        for i in range(n_trajectories)]

    raw_results = [s.result()[0] if client else s[0] for s in submissions]

    return {species: [trajectory[species] for trajectory in raw_results]
             for species in raw_results[0].keys()}

def kolmogorov_distance(data1, data2):
    return np.mean([scs.ks_2samp(
        np.array(data1[species]).reshape(-1),
        np.array(data2[species]).reshape(-1),)[0] for species in data1])

class Cell:
    def __init__(self, template, parameters, docker=None):
        """ Create a cell model

        Parameters
        ----------
        parameters: {str:value}
            parameters used to complete the template
        docker: str
            name of docker container to be used

        """
        self.time_step = parameters['time_step']
        # Delete 'time_step' so that it doesn't show in **parameters below
        del parameters['time_step']

        self.model = template.format(
            output_file="{output_file}",
            initial_state="{initial_state}",
            time_stop="{time_stop}",
            freq_point="{freq_point}",
            time_step="{time_step}",
            seed="{seed}",
            **parameters)
        self.docker = docker

        parameters['time_step'] = self.time_step 

    def run_threshold(
            self,
            initial_state,
            time_stop,
            n_trajectories=10,
            seed=None,
            n_points=500,
            threshold = 0.25,
            docker=None,
            client=None):

        time_step = self.time_step

        wallclock_time = []
        data = []


        start = time.time()
        with Timer() as t:
            print(time_step)
            data.append(
                    run_trajectories(
                        self.model,
                        time_stop,
                        n_trajectories,
                        time_step,
                        initial_state,
                        seed=seed,
                        n_points=n_points,
                        docker=self.docker,
                        client=client))
        wallclock_time.append(t.duration)

        with Timer() as t:
            print(time_step/2)
            data.append(
                    run_trajectories(
                        self.model,
                        time_stop,
                        n_trajectories,
                        time_step/2,
                        initial_state,
                        seed=seed,
                        n_points=n_points,
                        docker=self.docker,
                        client=client))
        wallclock_time.append(t.duration)

        time_step /= 2

        while kolmogorov_distance(data[-2], data[-1]) > threshold:
            time_step /= 2
            print(time_step, kolmogorov_distance(data[-2], data[-1]))
            with Timer() as t:
                data.append(
                    run_trajectories(
                        self.model,
                        time_stop,
                        n_trajectories,
                        time_step,
                        initial_state,
                        seed=seed,
                        n_points=n_points,
                        docker=self.docker,
                        client=client))
            wallclock_time.append(t.duration)

        return (data, wallclock_time)

    def run(
            self,
            initial_state,
            time_stop,
            n_trajectories=1,
            seed=None,
            n_points=500,
            client=None):
        """ Run smoldyn several times, using provided initial state.

        Parameters
        ----------
        initial_state: [Mol]
            list of molecules at t=0
        time_stop: float
            Simulation duration
        n_trajectories: int
            Number of trajectories to simulate
        seed: int
            seed used to run smoldyn
        n_points: int
            number of time samples
        client:
            dask client used for parallelization, leave empty if run serially

        Returns
        -------
        {Species: [timeseries]} if n_trajectories > 1
        else ({Species: timeseries}, [Mol.__dict__])

        """

        if n_trajectories > 1:
            return run_trajectories(
                    self.model,
                    time_stop,
                    n_trajectories,
                    self.time_step,
                    initial_state,
                    seed=seed,
                    n_points=n_points,
                    docker=self.docker,
                    client=client)
        else:
            return run_trajectory(
                    self.model,
                    time_stop,
                    self.time_step,
                    initial_state,
                    seed=seed,
                    n_points=n_points,
                    docker=self.docker)

if __name__ == "__main__":
    template = """random_seed {seed} \n define K_A {k_a} \n define K_D {k_d} \n define MU {mu} \n define KAPPA {kappa} \n define GAMMA {gamma} \n define OUTPUT stdout \n \n dim 3 \n boundaries x -8 8 \n boundaries y -8 8 \n boundaries z -8 8 \n time_start 0 \n time_stop {time_stop} \n time_step {time_step} \n \n species Gf Gb RNA P \n difc RNA {diffusion} \n difc P {diffusion} \n display_size all(all) 0.02 \n color P(all) lightblue \n color RNA(all) navy \n color Gf(all) scarlet \n color Gb(all) darkred \n \n graphics opengl_good \n frame_thickness 0 \n \n start_surface membrane \n action both all reflect \n color both blue \n thickness 0.01 \n panel sphere 0 0 0 {cell_radius} 30 30 \n end_surface \n \n start_surface nucleus \n action front RNA reflect \n action back P reflect \n color both red \n thickness 0.01 \n panel sphere 0 0 0 {nucleus_radius} 30 30 \n end_surface \n \n start_compartment inside \n surface nucleus \n point 0 0 0\n end_compartment \n \n start_compartment cytoplasm \n surface membrane \n point 0 0 0\n compartment andnot inside \n end_compartment \n \n reaction transcription Gf -> Gf + RNA MU \n reaction compartment=cytoplasm translation RNA -> RNA + P KAPPA \n reaction binding Gf + P <-> Gb K_A K_D \n reaction compartment=cytoplasm degradation RNA|P -> 0 GAMMA \n reaction compartment=inside degradation2 RNA|P -> 0 GAMMA \n {initial_state} \n \n output_files OUTPUT \n cmd B molcountheader OUTPUT \n cmd N {freq_point} molcount OUTPUT \n cmd A echo OUTPUT "--Simulation ends--\\n" \n cmd A listmols OUTPUT \n \n end_file \n"""

    parameters = {
        'k_a': 0.00166,
        'k_d': 0.1,
        'mu': 3.0,
        'kappa': 1.0,
        'gamma': 0.04,
        'diffusion': 0.6,
        'time_step': 0.32,
        'cell_radius': 6.0,
        'nucleus_radius': 2.5,
        'model': 'smoldyn',
        }

    initial_state = [Mol("Gf", 0.0, 0.0, 0.0)]

    p = 8
    model = Cell(template, parameters)
    result = model.run(initial_state, 1200, n_trajectories=2**p, n_points=500)
    for N in [2**i for i in range(p)]:
        print(N, kolmogorov_distance(
            {species: traj[:N] for species, traj in result.items()},
            {species: traj[N:2*N] for species, traj in result.items()}
            ), end=' ')
        print(kolmogorov_distance(
            {species: traj[:2*N:2] for species, traj in result.items()},
            {species: traj[1:2*N:2] for species, traj in result.items()}
            ))
