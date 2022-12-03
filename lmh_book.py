from evaluator import eval, Env, standard_env, Procedure

import torch as tc
from pyrsistent import pmap, pset
from copy import deepcopy
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


def get_LMH_samples(ast: dict, num_samples: int, wandb_name:str,  verbose:bool,  run_name = "start"):
    env = standard_env()
    sig = pmap({'logW':tc.tensor(0.), 'type': None, 'address': "start", 'num_sample_state': tc.tensor(0.0), 'params': None})
    # sig = {'logW':tc.tensor(0.), 'type': None, 'address': "start", 'num_sample_state': tc.tensor(0.0)}
    exp = eval(ast, sig, env, verbose)(run_name, lambda x : x) ### First run
    # exp = eval(ast, sig, env, verbose)

    D = pmap({}) ### Database for storing continuation and loglikelihood
    samples = lmh_sampler(exp, num_samples, D)
    
    return samples



def trace_update(k, D = pmap({}), px=None, py = None):
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

        elif k[2]['type'] == "observe":
            dist_k = sigma['dist']
            y_k = args
            l = dist_k.log_prob(*y_k)
            py = py + l
            k = cont(*args)

        else:
            k = cont(*args)

    return px, py, k, D, names, num_sample_states
            



def lmh_sampler(k, num_samples, D):

    px_old, py_old, x_old, D, names, num_sample_states_old = trace_update(k, D)

    samples = []

    log_probs = []

    num_accept = 0

    for i in range(num_samples):
        
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
        k_cont = k_mid[0]
        k_mid_new = (k_cont, [x_mid_new], k_mid[2])
        D_mid_new = D.set(target, [dist_mid, l_mid_new, [x_mid_new], k_mid_new, px_mid, py_mid, num_sample_states_mid])

        # Run the program starting from the new x
        px_new, py_new, x_new, D_new, _, num_sample_states_new = trace_update(k_mid_new, D_mid_new, px_mid, py_mid)

        #######################################

        rejection_new = (px_new+py_new) + tc.log(num_sample_states_old)
        rejection_old = (px_old+py_old) + tc.log(num_sample_states_new + num_sample_states_mid)

        rejection = tc.exp(rejection_new - rejection_old)

        if tc.rand(1) < rejection:
            D = D_new
            px_old = px_new
            py_old = py_new
            samples.append(x_new)
            x_old = x_new
            num_sample_states_old = num_sample_states_mid + num_sample_states_new
            log_probs.append(px_new+py_new)
            num_accept +=1
        else:
            samples.append(x_old)
            log_probs.append(px_old+py_old)

    # samples = samples[math.floor(0.2*len(samples)):]

    # plt.plot(log_probs)
    # plt.savefig('log_probs.png')
    # plt.close()

    # plt.plot(samples)
    # plt.savefig('samples.png')
    # plt.close()

    print("number of acceptance: ", num_accept)


    return samples

        




