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



def run_until_observe_or_end(k, D):

    if px == None or py == None: px = tc.tensor(0.0); py = tc.tensor(0.0)
    
    names = []
    num_sample_states = tc.tensor(0.0) # number of samples for this run, total num samples recorded in lmh_sampler instead


    while isinstance(k, tuple):

        cont, args, sigma = k

        if sigma['type'] == "sample":
            
            x = args
            name = sigma['address']
            names.append(name)
            dist = sigma['dist']
            num_sample_states = sigma['num_sample_state']

            l = dist.log_prob(*x)
            D = D.update({name: [dist, l, x, k, px, py, num_sample_states]})

            px = px + l
            k = cont(*args)

        elif sigma['type'] == "observe":
            return k, D


        else:
            k = cont(*args)

    return px, py, k, D, names, num_sample_states

def precond(D):

    return

def get_SMC_samples(ast:dict, num_samples:int, run_name='start', wandb_name=None, verbose=False):
    '''
    Generate a set of samples via Sequential Monte Carlo from a HOPPL program
    '''
    particles = []
    weights = []
    logZs = []
    n_particles = num_samples
    Ds = []

    for i in range(n_particles):
        particle = eval(ast, pmap({'logW':tc.tensor(0.0), 'address':'', 'type': None, 'num_sample_state':tc.tensor(0.0)}),standard_env(), verbose)("start", lambda x:x)
        logW = tc.tensor(0.)

        particles.append(particle)
        weights.append(logW)
        Ds.append(pmap({}))
    
    done = False
    smc_cnter = 0
    while not done:
        for i in range(n_particles): 
            res = run_until_observe_or_end(particles[i], Ds[i])
            if type(res) != tuple:
                particles[i] = res[0]
                if i == 0:
                    done = True 
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                particles[i], Ds[i] = res

        particles = precond(Ds)        


        if not done:
            check_addresses(particles)
            weights = [particle[0].sig['logW'] for particle in particles]
            particles = resample_using_importance_weights(particles, weights)
            for particle in particles:
                particle[0].sig = particle[0].sig.set('logW', tc.tensor(0.0))


            logZs.append(weights)
        smc_cnter += 1
    logZ = tc.tensor(logZs).sum(dim=0)
        
    return particles

