from evaluator import eval, Env, standard_env, Procedure

import torch as tc
import numpy as np
from pyrsistent import pmap, pset
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from utils import resample_using_importance_weights, check_addresses, calculate_effective_sample_size
from lmh_book import trace_update
from copy import deepcopy
import time


def get_rejSMC_samples(ast:dict, num_samples:int, num_rej:int,  run_name='start', wandb_name=None, verbose=False):
    
    particles = []
    weights = []
    logZs = []
    n_particles = num_samples
    Ds = []
    checkpoints = [None]*num_samples
    num_observe = 0
    ess = []
    plot_files = []

    for i in range(n_particles):
        sigma =  pmap({'values': '', 'logW':tc.tensor(0.), 'type': None, 'address': "start", 'num_sample_state': tc.tensor(0.0)})
        particle = eval(ast, sigma, standard_env(), verbose)("start", lambda x:x)
        logW = tc.tensor(0.)

        particles.append(particle)
        weights.append(logW)
        Ds.append(pmap({}))


    while type(particles[0]) == tuple:
        num_sample_original = 0
        for i in range(num_samples):
            ### Run program and stop at the observe. Store output in checkpoints
            checkpoints[i] = rejsmc_trace_update(particles[i], Ds[i])
            num_sample_original += checkpoints[i][6]
        
        ### If not tuple, then program finished
        if type(checkpoints[i]) != tuple: particles = [checkpoint for checkpoint in checkpoints]; break


        ### Record # of observe
        num_observe += 1


        ### Resample particles, Ds and checkpoints, based on weights
        particles = [checkpoint[2] for checkpoint in checkpoints]
        weights = [checkpoints[1] for checkpoints in checkpoints]
        particles, Ds, checkpoints, ESS_org, N = resample_rejsmc(particles, Ds, checkpoints, weights)


        ### Rejunenation step
        rej_start_time = time.time()
        num_sample_rej_total = 0
        for i in range(num_samples):
            particles[i], weights[i], num_sample_rej = rejuvenate(checkpoints[i], num_rej)
            num_sample_rej_total += num_sample_rej
        rej_time = time.time()-rej_start_time
        ESS_rej = 0

        _ = summary_iter(num_observe, ESS_org, ESS_rej, rej_time,  N, num_sample_original, num_sample_rej_total)

        ess.append(ESS_org/N)
        

        ### Zero out weights
        for i in range(num_samples):
            particles[i][0].sig = particles[i][0].sig.set('logW', tc.tensor(0.0)) ### TODO: check which weight to reset
            cont, args, sig = particles[i]
            particles[i] = cont(*args) ### At 'observe', push to run

    


    return particles, ess

def summary_iter(num_observe, ESS_org, ESS_rej, rej_time,  N, num_sample_original, num_sample_rej_total):
    print('')
    print('Observe', num_observe)
    print('Sample size:', N)
    print('SMC ESS', ESS_org)
    print('SMC Fractional ESS', ESS_org/N)
    print('Rejuvenated ESS:', ESS_rej)
    print('Rejuvenated Fractional SS:', ESS_rej/N)
    print('Rejuvenation time:', rej_time)
    print('num_sample total smc:',  num_sample_original)
    print('num_sample total rejuvenated', num_sample_rej_total)
    return None

def summary_non_rej_iter(num_observe, ESS_org, N, num_sample_original):
    print('')
    print('Observe', num_observe)
    print('Sample size:', N)
    print('SMC ESS', ESS_org)
    print('SMC Fractional ESS', ESS_org/N)
    print('Rejuvenation not needed')
    print('num_sample smc:',  num_sample_original)
    return None

            
def rejuvenate(checkpoint, num_rej):
    
    ### Run it once ###
    px_old, py_old, k_old, D, names, num_sample_states_old, _ = checkpoint

    ### If no sample statement ###
    if len(names) == 0:
        return k_old, px_old + py_old, 0

    num_sample_visited = 0

    for _ in range(num_rej):

        ### Randomly select a sample statement
        rd_idx = random.choice(range(0, len(names))) # position of the randomly selected sample point
        target = names[rd_idx]
        num_sample_visited  = num_sample_visited + len(names)- (rd_idx)

        ### Look up
        dist_mid, l_mid, [x_mid], k_mid, px_mid, py_mid, num_sample_states_mid = D[target]

        ### Resample sample states
        x_mid_new = dist_mid.sample()
        l_mid_new = dist_mid.log_prob(x_mid_new)

        ### Create new trace
        # k_mid_new = deepcopy((k_mid[0], [x_mid_new], k_mid[2]))
        k_mid_new = (k_mid[0], [x_mid_new], k_mid[2])
        D_mid_new = D.set(target, [dist_mid, l_mid_new, [x_mid_new], k_mid_new, px_mid, py_mid, num_sample_states_mid])


        ### Rerun ###
        px_new, py_new, k_new, D_new, _, num_sample_states_new, _ = rejsmc_trace_update(k_mid_new, D_mid_new, px_mid, py_mid)


        ### Rejection step

        rejection_top = (px_new+py_new) + l_mid + tc.log(num_sample_states_new)
        rejection_btm = (px_old+py_old) + l_mid_new + tc.log(num_sample_states_old)

        # rejection_top = (px_new+py_new) + tc.log(num_sample_states_new)
        # rejection_btm = (px_old+py_old)  + tc.log(num_sample_states_old)

        rejection = tc.exp(rejection_top - rejection_btm)

        if tc.rand(1) < rejection:
            k_old = k_new
            px_old = px_new
            py_old = py_new
            D = D_new
            num_sample_states_old = num_sample_states_new
        else:
            pass

    return k_old, py_old, num_sample_visited



def rejsmc_trace_update(k, D, px = None, py = None):
    
    ### Initialize px and py if first run
    if px == None or py == None: px = tc.tensor(0.0); py = tc.tensor(0.0)

    ### Sample names and number of sample states
    names = []
    num_sample_states = tc.tensor(0.0)
    num_sample_visited = 0

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
            num_sample_visited += 1

            ### Run program forward
            k = cont(*args)


        elif sigma['type'] == "observe":
            
            ### Extract info from run
            y = args
            dist = sigma['dist']

            ### Calculate py
            l = dist.log_prob(*y)
            py = py + l

            ### Run pass this observe
            # k = cont(*args)

            return px, py, k, D, names, num_sample_states, num_sample_visited

    return k






def calculate_effective_sample_size_rejsmc(weights:tc.Tensor, rej_time: int, verbose=True):
    '''
    Calculate the effective sample size via the importance weights
    '''
    weights = tc.exp(tc.tensor(weights)).type(tc.float64)
    weights = weights/weights.sum()

    N = len(weights)
    weights /= weights.sum()
    ESS = 1./(weights**2).sum()
    ESS = ESS.type(tc.float)
    if verbose:
        print('')
        print('Rejuvenation step')
        print('Sample size:', N)
        print('Rejuvenated Effective sample size:', ESS)
        print('Rejuvenated Fractional sample size:', ESS/N)
        print('Sum of weights:', weights.sum())
        print('Rejuvenation time:', rej_time)
        print('')
    return ESS, N

def resample_rejsmc(samples:list, Ds:list, checkpoints:list, log_weights:list, normalize=True, wandb_name=None):
    '''
    Use the (log) importance weights to resample so as to generate posterior samples 
    '''
    nsamples = len(samples)
    weights = tc.exp(tc.tensor(log_weights)).type(tc.float64)
    weights = weights/weights.sum()
    ESS = calculate_effective_sample_size(weights, verbose=False)
    indices = np.random.choice(nsamples, size=nsamples, replace=True, p=weights)
    new_samples = [samples[index] for index in indices]
    new_Ds = [Ds[index] for index in indices]
    new_checkpoints = [checkpoints[index] for index in indices]
    new_weights = tc.tensor([weights[index] for index in indices])
    # ESS = calculate_effective_sample_size(new_weights, verbose=False)
    return new_samples, new_Ds, new_checkpoints, ESS, len(weights)