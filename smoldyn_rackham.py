import sys
import json
import smoldyn

template = """random_seed {seed} 
define K_A {k_a} 
define K_D {k_d} 
define MU {mu} 
define KAPPA {kappa} 
define GAMMA {gamma} 
define OUTPUT stdout 

dim 3 
boundaries x -8 8 
boundaries y -8 8 
boundaries z -8 8 
time_start 0 
time_stop {time_stop} 
time_step {time_step} 

species Gf Gb RNA P 
difc RNA {diffusion} 
difc P {diffusion} 
display_size all(all) 0.02 
color P(all) lightblue 
color RNA(all) navy 
color Gf(all) scarlet 
color Gb(all) darkred 

graphics opengl_good 
frame_thickness 0 

start_surface membrane 
action both all reflect 
color both blue 
thickness 0.01 
panel sphere 0 0 0 {cell_radius} 30 30 
end_surface 

start_surface nucleus 
action front RNA reflect 
action back P reflect 
color both red 
thickness 0.01 
panel sphere 0 0 0 {nucleus_radius} 30 30 
end_surface 

start_compartment inside 
surface nucleus 
point 0 0 0\n end_compartment 

start_compartment cytoplasm 
surface membrane 
point 0 0 0\n compartment andnot inside 
end_compartment 

reaction transcription Gf -> Gf + RNA MU 
reaction compartment=cytoplasm translation RNA -> RNA + P KAPPA 
reaction binding Gf + P <-> Gb K_A K_D 
reaction compartment=cytoplasm degradation RNA|P -> 0 GAMMA 
reaction compartment=inside degradation2 RNA|P -> 0 GAMMA 
{initial_state} 

output_files OUTPUT 
cmd B molcountheader OUTPUT 
cmd N {freq_point} molcount OUTPUT 
cmd A echo OUTPUT "--Simulation ends--\\n" 
cmd A listmols OUTPUT 

end_file 
"""

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
TSTOP = 1000
NTRAJ = 128

c, d = float(sys.argv[1]), float(sys.argv[2])

init_model = smoldyn.Cell(template, base_parameters)
# Generate initial state
initial_state = [smoldyn.Mol("Gf", 0.0, 0.0, 0.0)]
_, initial_state = init_model.run(initial_state, time_stop=1000, n_trajectories=1, n_points=2)

smoldyn_data = []
wallclock_time = []

smoldyn_parameters = {**base_parameters}
smoldyn_parameters['mu'] *= 2**c
smoldyn_parameters['kappa'] *= 2**c
smoldyn_parameters['gamma'] *= 2**c
smoldyn_parameters['diffusion'] *= 2**d
smoldyn_parameters['model'] = 'smoldyn'

model = smoldyn.Cell(template, smoldyn_parameters)

_, initial_state = model.run(initial_state, time_stop=INIT_STOP, n_trajectories=1, n_points=2)

history, time_data = model.run_threshold(
    initial_state, time_stop=TSTOP, n_trajectories=NTRAJ, n_points=500, threshold=0.05)

smoldyn_data.append((smoldyn_parameters, history))
wallclock_time.append((smoldyn_parameters, time_data))

with open('data/smoldyn_data({},{}).json'.format(c, d), 'w') as f:
    json.dump(smoldyn_data, f)
    
with open('data/smoldyn_time({},{}).json'.format(c, d), 'w') as f:
    json.dump(wallclock_time, f)
