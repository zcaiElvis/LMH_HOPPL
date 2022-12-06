# Standard imports
from time import time
import torch as tc
from pyrsistent import pmap

# Project imports
from evaluator import eval, evaluate, standard_env
from utils import log_sample_to_wandb, log_samples_to_wandb
from utils import resample_using_importance_weights, check_addresses
from lmh_book import get_LMH_samples
from psmc import get_PSMC_samples
from rej_smc import get_rejSMC_samples

def get_samples(ast:dict, num_samples:int, num_preconds:int, tmax=None, inference=None, wandb_name=None, verbose=False):
    '''
    Get some samples from a HOPPL program
    '''
    if inference is None:
        samples = get_prior_samples(ast, num_samples, tmax, wandb_name, verbose)
    elif inference == 'IS':
        samples = get_importance_samples(ast, num_samples, tmax, wandb_name, verbose)
    elif inference == 'SMC':
        samples = get_SMC_samples(ast, num_samples, wandb_name, verbose)
    elif inference == "LMH":
        samples = get_LMH_samples(ast, num_samples, wandb_name, verbose)
    elif inference =="PSMC":
        samples = get_PSMC_samples(ast, num_samples, num_preconds, wandb_name, verbose)
    elif inference == "rejSMC":
        samples = get_rejSMC_samples(ast, num_samples, num_preconds, wandb_name, verbose)
    else:
        print('Inference scheme:', inference, type(inference))
        raise ValueError('Inference scheme not recognised')
    return samples


def get_prior_samples(ast:dict, num_samples:int, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a HOPPL program
    '''
    samples = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample, _ = evaluate(ast, verbose=verbose)
        if wandb_name is not None: log_sample_to_wandb(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        if (tmax is not None) and (time() > max_time): break
    return samples


def get_importance_samples(ast:dict, num_samples:int, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of importamnce samples from a HOPPL program
    '''
    samples = []
    log_weights = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        # sigma = pmap({'logW':tc.tensor(0.), 'address':'', 'num_sample_state': 0})
        # sigma = {'logW':tc.tensor(0.), 'address':'', 'num_sample_state': 0}
        sample, sigma = evaluate(ast, sig = None, verbose=verbose)
        if wandb_name is not None: log_sample_to_wandb(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        log_weights.append(sigma['logW'])
        if (tmax is not None) and (time() > max_time): break

    resamples = resample_using_importance_weights(samples, log_weights)
    
    return resamples



def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, weights, n_particles):
    weights = tc.exp(tc.tensor(weights))
    normalized_weights = tc.div(tc.tensor(weights), sum(weights))
    particle_idx = tc.distributions.Categorical(normalized_weights).sample(tc.Size([n_particles]))
    new_part = [particles[i.item()] for i in particle_idx]

    logZn = tc.log(normalized_weights)

    for par in new_part:
        par[2]['logW'] = 0


    return new_part, logZn


def get_SMC_samples(ast:dict, num_samples:int, run_name='start', wandb_name=None, verbose=False):
    '''
    Generate a set of samples via Sequential Monte Carlo from a HOPPL program
    '''
    particles = []
    weights = []
    logZs = []
    n_particles = num_samples

    for i in range(n_particles):
        particle = eval(ast, pmap({'logW':tc.tensor(0.0), 'address':'', 'type': None, 'num_sample_state':tc.tensor(0.0)}),standard_env(), verbose)("start", lambda x:x)
        logW = tc.tensor(0.)

        particles.append(particle)
        weights.append(logW)
    
    done = False
    smc_cnter = 0
    while not done:
        for i in range(n_particles): 
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]:
                particles[i] = res[0]
                if i == 0:
                    done = True 
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                particles[i] = res


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

