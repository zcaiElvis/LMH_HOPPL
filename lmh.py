from evaluator import eval, Env, standard_env

import torch as tc
from pyrsistent import pmap, pset
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_LMH_samples(ast: dict, num_samples: int, wandb_name:str,  verbose:bool,  run_name = "start"):
    env = standard_env()
    sig = pmap({'logW':tc.tensor(0.), 'type': None, 'address': "start"})
    output = lambda x: x
    exp = eval(ast, sig, env, verbose)(run_name, output) ### First run

    D = pmap({}) ### Database for storing continuation and loglikelihood
    samples = lmh_sampler(exp, num_samples, D)
    

    return samples


def run_until_sample_or_end(res):
    ### We want to stop before sample, not after sample statement
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'sample':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = [res, None, {'done' : True}, sigma]
    return res


def trace_update(k, D = pmap({})):
    ll = 0
    done = False
    names = []

    while not done:
        res = run_until_sample_or_end(k)
        if 'done' in res[2]: # If the program ends
            sample = res[0]
            sigma = res[3]
            done = True
            # k contains [res, None, {}, sigma]
        else:
            k = res
            # k contains [cont, [x:sample from distribution], sigma={address, logW, distribution}]

        if not done:
            n = k[0].sig['address']
            d_c = k[0].sig['dist']
            theta_c = vars(d_c)
            l_c = k[0].sig['logW']
            if n not in names: names.append(n)

            if n in D.keys(): 
                k_d = D[n]
                if type(k_d[2]['dist']) == type(d_c):
                    if vars(k_d[2]['dist']) == theta_c: 
                        ll = ll + l_c
                    else: 
                        D.update({n:k}) 
                        l = k[0].sig['dist'].log_prob(*k[1])
                        ll = ll+l
            
            else: 
                D = D.update({n:k})
                l = k[0].sig['dist'].log_prob(*k[1])
                ll = ll+l

    # ll = sigma['logW']

    return [ll, D, names, sample]


def lmh_sampler(k, num_samples = 100, D = pmap({})):
    ll, D, names, sample = trace_update(k, D)
    samples = []
    acceptance = 0
    last_sample = sample
    
    for i in tqdm(range(num_samples)):
        target = random.choice(names)
        k_d = D[target]
        cont_d, [x_d], sig_d = k_d
        d_d = sig_d['dist']
        x = d_d.sample()
        
        F = d_d.log_prob(x)
        R = d_d.log_prob(x_d)

        k_copy = (k_d[0], [x], k_d[2])
        D_copy = D
        D_copy.update({target: k_copy})

        ll_p, D_p, _, sample_current = trace_update(k_copy, D_copy)

        if tc.rand(1) < ll_p - ll + R - F:
            D = D_p
            acceptance +=1
            samples.append(sample_current)
            last_sample = sample_current
        else:
            samples.append(last_sample)

    print(acceptance)
    plt.plot(samples)
    plt.savefig('samples.png')


    return samples


# TODO: 1) count number of samples on a trace
#       2) Parameters of samples (is it important?)