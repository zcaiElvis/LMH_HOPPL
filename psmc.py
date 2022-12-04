from evaluator import eval, Env, standard_env, Procedure

import torch as tc
from pyrsistent import pmap, pset
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from utils import resample_using_importance_weights, check_addresses
from lmh_book import trace_update
from copy import deepcopy


def get_PSMC_samples(ast:dict, num_samples:int, run_name='start', wandb_name=None, verbose=False):
    '''
    Generate a set of samples via Sequential Monte Carlo from a HOPPL program
    '''
    particles = []
    weights = []
    logZs = []
    n_particles = num_samples
    Ds = []

    for i in range(n_particles):
        sigma =  pmap({'logW':tc.tensor(0.), 'type': None, 'address': "start", 'num_sample_state': tc.tensor(0.0), 'params': None})
        particle = eval(ast, sigma, standard_env(), verbose)("start", lambda x:x)
        logW = tc.tensor(0.)

        particles.append(particle)
        weights.append(logW)
        Ds.append(pmap({}))
    
    done = False

    while type(particles[0]) == tuple:
        for i in range(num_samples):
            particles[i] = precond(particles[i], Ds[i]) ### run until observe, construct new trace from rd sample, compare

        # check_addresses(particles)
        weights = [particle[0].sig['logW'] for particle in particles]
        particles = resample_using_importance_weights(particles, weights)
        for particle in particles:
            particle[0].sig = particle[0].sig.set('logW', tc.tensor(0.0))

    return particles

            
def precond(particle, D, num_precond = 3):
    
    ### Run it once ###
    px_old, py_old, k_old, D, names, num_sample_states_old = psmc_trace_update(particle, D)


    for _ in range(num_precond):


        ### Randomly select a sample statement
        rd_idx = random.choice(range(0, len(names))) # position of the randomly selected sample point
        target = names[rd_idx]

        ### Look up
        dist_mid, l_mid, [x_mid], k_mid, px_mid, py_mid, num_sample_states_mid = D[target]

        ### Resample sample states
        x_mid_new = dist_mid.sample()
        l_mid_new = dist_mid.log_prob(x_mid_new)

        ### Create new trace
        k_mid_new = deepcopy((k_mid[0], [x_mid_new], k_mid[2]))
        D_mid_new = D.set(target, [dist_mid, l_mid_new, [x_mid_new], k_mid_new, px_mid, py_mid, num_sample_states_mid])


        ### Rerun ###
        px_new, py_new, k_new, D_new, _, num_sample_states_new = psmc_trace_update(k_mid_new, D_mid_new, px_mid, py_mid)


        ### Rejection step

        rejection_top = (px_new+py_new) + tc.log(num_sample_states_new)
        rejection_btm = (px_old+py_old) + tc.log(num_sample_states_old)
        rejection = tc.exp(rejection_top - rejection_btm)

        if tc.rand(1) < rejection:
            k_old = k_new
            px_old = px_new
            py_old = py_new
            D = D_new
            num_sample_states_old = num_sample_states_new
        else:
            pass

    return k_old



def psmc_trace_update(k, D, px = None, py = None):
    
    ### Initialize px and py if first run
    if px == None or py == None: px = tc.tensor(0.0); py = tc.tensor(0.0)

    ### Sample names and number of sample states
    names = []
    num_sample_states = tc.tensor(0.0)

    ### Run particle:

    while isinstance(k, tuple):

        cont, args, sigma = k

        if sigma['type'] == "sample":

            ### Extract info from run
            x = args
            name = sigma['address']
            dist = sigma['dist']
            num_sample_states = sigma['num_sample_state']
            

            ### Store to D. Note px is before seeing this sample
            l = dist.log_prob(*x)
            D = D.update({name: [dist, l, x, k, px, py, num_sample_states]})

            ### Compute px, append to name
            px = px + l
            names.append(name)

            ### Run program forward
            k = cont(*args)


        elif sigma['type'] == "observe":
            
            ### Extract info from run
            y = args
            dist = sigma['dist']

            ### Calculate py
            l = dist.log_prob(*y)
            py = py + l

            return px, py, k, D, names, num_sample_states

    return k






