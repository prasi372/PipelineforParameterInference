"""
    Compartment Based Model
"""

import sys
import os.path
import numpy as np
import gillespy2
import json

def generate_partitions(n_traj, n_cores):
    return [
        n_traj//n_cores + (1 if i < n_traj % n_cores else 0)
        for i in range(n_cores)]


class Cell(gillespy2.Model):
    def __init__(self, parameters):
        gillespy2.Model.__init__(self, "Compartment Based Model")

        self.list_species = ['Gf', 'Gb', 'RNAnuc', 'RNAcyt', 'Pnuc', 'Pcyt']

        self.dict_species = {
            species: gillespy2.Species(
                name=species,
                initial_value=0)
            for species in self.list_species}

        # Force species order to follow list_species
        # Necessary to extract results correctly
        self.add_species([self.dict_species[s] for s in self.list_species])
        # TODO unit test order preservation, e.g. all rates to 0...


        #Parameters
        parameters = {name: gillespy2.Parameter(name=name, expression=value)
            for name, value in parameters.items() if isinstance(value, float)}
        self.add_parameter(list(parameters.values()))

        #Reactions
        #Degradation
        dict_reactions = {}
        for name, species in [(name, self.dict_species[name])
                              for name in [
                                  'RNAnuc', 'RNAcyt', 'Pnuc', 'Pcyt']]:

            dict_reactions["dgd"+name] = gillespy2.Reaction(
                    name = "dgd"+name,
                    reactants = {species:1}, products={},
                    rate = parameters['gamma'])

        #Transcription
        dict_reactions["tcp"] = gillespy2.Reaction(
                name = "tcp",
                reactants = {self.dict_species['Gf']:1},
                products={self.dict_species['Gf']:1, self.dict_species['RNAnuc']:1},
                rate = parameters['mu'])

        #Translation
        dict_reactions["tlt"] = gillespy2.Reaction(
                name="tlt",
                reactants={self.dict_species['RNAcyt']:1},
                products={self.dict_species['RNAcyt']:1, self.dict_species['Pcyt']:1},
                rate=parameters['kappa'])

        #Binding with P
        dict_reactions["bdg"] = gillespy2.Reaction(
                name="bdg",
                reactants={self.dict_species['Gf']:1, self.dict_species['Pnuc']:1},
                products={self.dict_species['Gb']:1},
                rate=parameters['k_a'])

        #Unbinding from her
        dict_reactions["ubdg"+name] = gillespy2.Reaction(
                name="ubdg"+name,
                reactants={self.dict_species['Gb']:1},
                products={self.dict_species['Gf']:1, self.dict_species['Pnuc']:1},
                rate=parameters['k_d'])

        #Exit and entry
        dict_reactions["exitRNA"] = gillespy2.Reaction(
                name="exitRNA",
                reactants={self.dict_species['RNAnuc']:1},
                products={self.dict_species['RNAcyt']:1},
                rate=parameters['k_nc'])

        dict_reactions["entryP"] = gillespy2.Reaction(
                name="entryP",
                reactants={self.dict_species['Pcyt']:1},
                products={self.dict_species['Pnuc']:1},
                rate=parameters['k_cn'])

        self.add_reaction(list(dict_reactions.values()))

    def run(
            self,
            initial_state,
            time_stop,
            n_trajectories,
            seed=None,
            n_points=500):
        #Species
        for name, species in self.dict_species.items():
            species.initial_value = initial_state.get(name, 0)

        self.timespan(np.linspace(0, time_stop, num=n_points + 1).tolist())

        raw_results = gillespy2.Model.run(
                self,
                number_of_trajectories=n_trajectories, seed=seed)

        results = {
            species:
            ([trajectory[species].tolist() for trajectory in raw_results]
                if n_trajectories > 1 else
                raw_results[0][species].tolist())
            for species in self.list_species}

        results['time'] = ([list(self.tspan) for _ in range(n_trajectories)]
                if n_trajectories > 1 else
                list(self.tspan))

        return (results
                if n_trajectories > 1 else
                (results, {species: results[species][-1]
                    for species in results if species != 'time'}))


if __name__ == "__main__":
    parameters = {
        'gamma': 1.,
        'mu': 1.,
        'kappa': 1.,
        'k_a': 1.,
        'k_d': 1.,
        'k_nc': 1.,
        'k_cn': 1.,
        }
    model = Cell(parameters)
    results = model.run({'Gf':1}, 100, 1, n_points=100)
    print(results)
