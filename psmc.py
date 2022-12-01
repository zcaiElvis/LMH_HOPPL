from evaluator import eval, Env, standard_env, Procedure

import torch as tc
from pyrsistent import pmap, pset
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from utils import resample_using_importance_weights, check_addresses
from lmh_book import trace_update


def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) 
    return res



def get_PSMC_samples(ast:dict, num_samples:int, run_name='start', wandb_name=None, verbose=False):
    particles = []
    weights = []
    Ds = []
    logZs = []
    n_particles = num_samples

    ### Initialize particles
    for i in range(n_particles):
        particle = eval(ast, pmap({'logW':tc.tensor(0.0), 'address':'', 'type': None, 'num_sample_state':tc.tensor(0.0)}), standard_env(), verbose)("start", lambda x:x)
        logW = tc.tensor(0.)

        particles.append(particle)
        weights.append(logW)
        Ds.append(pmap({}))

    notdone = True

    while(notdone):

        ### For each particle, do preconditioning and return the best one
        for i in range(n_particles):
            particles[i], Ds[i] = precondition(particles[i], Ds[i], precond_iter = 2)


        ### Resample
        if type(particles[0]) == tuple:
            weights = [particle[0].sig['logW'] for particle in particles]
            particles = resample_using_importance_weights(particles, weights)
            for particle in particles:
                particle[0].sig = particle[0].sig.set('logW', tc.tensor(0.0))
        else:
            notdone = False

        
    return particles


def precondition(particle, D, precond_iter):
    ### run and stop at the next observe statement
    px_old, py_old, x_old, D, names, num_sample_states_old = trace_update_psmc(particle, D)
    samples = []
    log_probs = []

    if tc.equal(num_sample_states_old, tc.tensor(0.0)): return x_old, D


    for i in range(precond_iter):
        # Pick any points along the trace
        rd_idx = random.choice(range(0, len(names))) # position of the randomly selected sample point
        target = names[rd_idx]


        ##### Creating new trace and run #####

        # Look up the target sample in dictionary
        dist_mid, l_mid, [x_mid], k_mid, px_mid, py_mid, num_sample_states_mid = D[target]

        # Resample at the target sample
        x_mid_new = dist_mid.sample()
        l_mid_new = dist_mid.log_prob(x_mid_new)

        # Create new branch
        k_mid_new = (k_mid[0], [x_mid_new], k_mid[2])
        D_mid_new = D.set(target, [dist_mid, l_mid_new, [x_mid_new], k_mid_new, px_mid, py_mid, num_sample_states_mid])

        # Run the program starting from the new x
        px_new, py_new, x_new, D_new, _, num_sample_states_new = trace_update(k_mid_new, D_mid_new, px_mid, py_mid)

        #######################################

        rejection_new = (px_new+py_new)+ l_mid + tc.log(num_sample_states_old)
        rejection_old = (px_old+py_old)+ l_mid_new + tc.log(num_sample_states_new + num_sample_states_mid)


        if tc.rand(1) < tc.exp(rejection_new - rejection_old):
            D = D_new
            px_old = px_new
            py_old = py_new
            samples.append(x_new)
            x_old = x_new
            num_sample_states_old = num_sample_states_mid + num_sample_states_new
            log_probs.append(px_new+py_new)
        else:
            samples.append(x_old)
            log_probs.append(px_old+py_old)

    if tc.is_tensor(x_old):
        # If no acceptance, then take x_old as the input particle
        x_old = particle

    return x_old, D

def trace_update_psmc(k, D = pmap({}), px=None, py = None):
    if px == None or py == None: px = tc.tensor(0.0); py = tc.tensor(0.0)
    
    names = []
    num_sample_states = tc.tensor(0.0) # number of samples for this run, total num samples recorded in lmh_sampler instead

    while isinstance(k, tuple):

        cont, args, sigma = k

        if k[2]['type'] == "sample":
            
            x_k = args
            name = sigma['address']
            names.append(name)
            dist_k = sigma['dist']
            l = dist_k.log_prob(*x_k)
            D = D.update({name: [dist_k,l, x_k, k, px, py, num_sample_states]})
            px = px + l
            num_sample_states = num_sample_states + 1
            k = cont(*args)

        elif k[2]['type'] == "observe": ### If "observe", evaluate and return
            dist_k = sigma['dist']
            y_k = args
            l = dist_k.log_prob(*y_k)
            py = py + l
            k = cont(*args) ### what happen if two samples happen
            return px, py, k, D, names, num_sample_states

            # if isinstance(k, tuple):
            #     if k[2]['type'] != "observe":
            #         return px, py, k, D, names, num_sample_states

        else:
            k = cont(*args)

    return px, py, k, D, names, num_sample_states



