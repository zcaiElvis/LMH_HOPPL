from evaluator import eval, Env, standard_env

import torch as tc
from pyrsistent import pmap, pset
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


def get_LMH_samples(ast: dict, num_samples: int, wandb_name:str,  verbose:bool,  run_name = "start"):
    env = standard_env()
    sig = pmap({'logW':tc.tensor(0.), 'type': None, 'address': "start", 'num_sample_state': tc.tensor(0.0)})
    output = lambda x: x
    exp = eval(ast, sig, env, verbose)(run_name, output) ### First run
    # exp = eval(ast, sig, env, verbose)

    D = pmap({}) ### Database for storing continuation and loglikelihood
    samples = lmh_sampler(exp, num_samples, D)
    

    return samples


def run_until_vars_or_end(res, num_sample_states):
    cont, args, sigma = res
    if tc.equal(sigma['num_sample_state'], tc.tensor(1.)) and tc.equal(num_sample_states, tc.tensor(1.)): # If this is the first run
        return [{'status':'sample'}, res]
    res = cont(*args)
    while type(res) is tuple:
        # print(args)
        if res[2]['type'] == 'sample':
            return [{'status': 'sample'}, res]
        elif res[2]['type'] == 'observe':
            return [{'status': 'observe'}, res]

        cont, args, sigma = res
        res = cont(*args)

    res = [{'status': 'done'}, res]
    return res

def trace_update(k, D = pmap({}), px=None, py = None):
    if px == None or py == None: px = tc.tensor(0.0); py = tc.tensor(0.0)
    
    names = []

    done = False

    num_sample_states = k[2]['num_sample_state']


    while(done == False):
        status, k = run_until_vars_or_end(k, num_sample_states)
        # skip = skip + 1

        if status['status'] == 'done':
            done = True
            sample = k

        else:
            # k = {cont, args, sigma}
            if status['status'] == 'sample':
                x_k = k[1]
                name = k[2]['address']
                names.append(name)
                dist_k = k[2]['dist']
                num_sample_states = num_sample_states + 1
                if name in D.keys(): # If already in dictionary
                    disk_d, l_d, _, _, px, py, _ = D[name]
                    if dist_k.params ==  disk_d.params:
                        px = px + l_d
                    else:
                        l = dist_k.log_prob(*x_k)
                        
                        px = px + l
                        # store px, py until and include seeing x_k
                        D = D.update({name: [dist_k, l, x_k, k, px, py, num_sample_states]})
                        
                else: # If not in dictionary
                    l = dist_k.log_prob(*x_k)
                    px = px + l
                    D = D.update({name: [dist_k,l, x_k, k, px, py, num_sample_states]}) # storing px, py

            elif status['status'] == 'observe':
                dist_k = k[2]['dist']
                y_k = k[1]
                l = dist_k.log_prob(*y_k)
                py = py + l

            else:
                raise Exception("Status not recognized:"+status['status'])

    return px, py, sample, D, names, num_sample_states



def lmh_sampler(k, num_samples, D):
    px_old, py_old, sample_old, D, names, num_sample_states_old = trace_update(k, D)

    samples = []
    prev_sample = sample_old

    for i in range(num_samples):
        # Pick any points along the trace
        rd_idx = random.choice(range(0, len(names))) # position of the randomly selected sample point
        target = names[rd_idx]

        # Look up the target sample in dictionary
        dist_old, l_old, [x_old], k_old, px_mid, py_mid, num_sample_states_old = D[target]

        # Resample at the target sample
        x_new = dist_old.sample()
        px_mid = px_mid - dist_old.log_prob(x_old) + dist_old.log_prob(x_new)

        # Create new branch
        k_new = [k_old[0], [x_new], k_old[2]]
        D_new = D.set(target, [dist_old, l_old, [x_new], k_new, px_mid, py_mid, num_sample_states_old])

        # Run the program starting from the new x
        px_new, py_new, sample_new, D_new, _, num_sample_states_new = trace_update(k_new, D_new, px_mid, py_mid)

        ### TODO:Get number of samples in each trace

        print(num_sample_states_old - num_sample_states_new)
        
        rejection_new = (px_new+py_new)+ dist_old.log_prob(x_new) + tc.log(num_sample_states_old)
        rejection_old = (px_old+py_old)+ dist_old.log_prob(x_old) + tc.log(num_sample_states_new)

        # print(tc.exp(rejection_new-rejection_old))


        if tc.rand(1) < tc.exp(rejection_new - rejection_old):
            D = D_new
            px_old = px_new
            py_old = py_new
            samples.append(sample_new)
            prev_sample = sample_new
        else:
            pass
            samples.append(prev_sample)

    samples = samples[math.floor(0.2*len(samples)):]

    plt.plot(samples)
    plt.savefig('samples.png')

    return samples

        




