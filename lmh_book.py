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


def run_until_vars_or_end(res, skip):
    cont, args, sigma = res
    if tc.equal(sigma['num_sample_state'], tc.tensor(1.)) and tc.equal(skip, tc.tensor(1.)): # If this is the first run
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

def trace_update(k, D = pmap({}), px=0, py = 0):
    
    names = []

    done = False

    skip = k[2]['num_sample_state']

    while(done == False):
        status, k = run_until_vars_or_end(k, skip)
        skip = skip + 1

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
                if name in D.keys(): # If already in dictionary
                    disk_d, l_d, x_d, k_d, px, py = D[name]
                    if dist_k.params ==  disk_d.params:
                        px = px + l_d
                    else:
                        l = dist_k.log_prob(*x_k)
                        
                        px = px + l
                        D = D.update({name: [dist_k, l, x_k, k, px, py]})
                        
                else: # If not in dictionary
                    l = dist_k.log_prob(*x_k) # This was in logW but now gone, so recalculate
                    px = px + l
                    D = D.update({name: [dist_k,l, x_k, k, px, py]}) # record px, py before seeing the new sample

            elif status['status'] == 'observe':
                dist_k = k[2]['dist']
                y_k = k[1]
                l = dist_k.log_prob(*y_k)
                py = py + l

            else:
                raise Exception("Status not recognized:"+status['status'])

    return px, py, sample, D, names



def lmh_sampler(k, num_samples, D):
    px, py, sample, D, names = trace_update(k, D)
    # print([D[d][0] for d in D.keys()])
    # print([D[d][3][2]['num_sample_state'] for d in D.keys()])
    samples = []
    prev_sample = sample

    for i in range(num_samples):
        # Pick any points along the trace
        rd_idx = random.choice(range(0, len(names))) # position of the randomly selected sample point
        target = names[rd_idx]


        disk_d, l_d, [x_d], k_d, px_d, py_d = D[target]
        x_p = disk_d.sample()

        k_p = [k_d[0], [x_p], k_d[2]]
        px_p, py_p, sample, D_p, names_p = trace_update(k_p, D, px_d, py_d)

        ### Get number of samples in each trace
        

        if tc.rand(1) < (px_p+py_p)+ disk_d.log_prob(x_p) - (px+py) - disk_d.log_prob(x_d):
            D = D_p
            px = px_p
            py = py_p
            samples.append(sample)
            prev_sample = sample
        else:
            pass
            samples.append(prev_sample)

    samples = samples[math.floor(0.2*len(samples)):]

    plt.plot(samples)
    plt.savefig('samples.png')

    return samples

        




