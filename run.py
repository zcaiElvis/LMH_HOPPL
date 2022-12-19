import numpy as np
import torch as tc
from time import time
import wandb
import hydra
import sys
import pandas as pd
import pickle
import os
from time import time, strftime

# Project imports
from daphne import load_program
from evaluator import evaluate, eval
from sampling import get_samples
from tests import is_tol, run_probabilistic_test, load_truth
from utils import wandb_plots_homework5, wandb_plots_homework6

def run_tests(tests, test_type, base_dir, daphne_dir, num_samples=int(1e4), max_p_value=1e-4, compile=True, verbose=False):

    # File paths
    # NOTE: This path should be with respect to the daphne path
    test_dir = base_dir+'/programs/tests/'+test_type+'/'
    daphne_test = lambda i: test_dir+'test_%d.daphne'%(i)
    json_test = lambda i: test_dir+'test_%d.json'%(i)
    truth_file = lambda i: test_dir+'test_%d.truth'%(i)

    # Loop over tests
    print('Running '+test_type+' tests:', tests)
    for i in tests:

        # Start
        print('Test %d starting'%i)
        ast = load_program(daphne_dir, daphne_test(i), json_test(i), mode='desugar-hoppl-cps', compile=compile)
        truth = load_truth(truth_file(i))
        if verbose: print('Test truth:', truth)

        # Deterministic tests
        if test_type in ['deterministic', 'hoppl-deterministic']:
            result, _ = evaluate(ast, verbose=verbose)
            if verbose: 
                print('Result:', result)
                print('Truth:', truth)
            try:
                assert(is_tol(result, truth))
            except:
                if not verbose:
                    print('Result:', type(result), result)
                    print('Truth:', type(truth), truth)
                raise AssertionError('Return value is not equal to truth')

        # Probabilistic tests
        elif test_type == 'probabilistic':
            samples = get_samples(ast, num_samples)
            p_val = run_probabilistic_test(samples, truth)
            print('p value:', p_val)
            assert(p_val > max_p_value)

        else:
            raise ValueError('Test type not recognised')

        # Finish
        print('Test %d passed'%i, '\n')
    print('All '+test_type+' tests passed\n')


def run_programs(programs, prog_set, base_dir, daphne_dir, num_samples=int(1e3), num_rej_run = [int(1e0)], num_samples_run = [int(1e3)], tmax=None, inference=None, compile=True, wandb_run=False, verbose=False):

    # File paths
    prog_dir = base_dir+'/programs/'+prog_set+'/'
    daphne_prog = lambda i: prog_dir+'%d.daphne'%(i)
    json_prog = lambda i: prog_dir+'%d.json'%(i)
    if inference is None:
        results_file = lambda i: 'data/%s/%d.dat'%(prog_set, i)
    else:
        results_file = lambda i: 'data/%s/%d_%s.dat'%(prog_set, i, inference)


    timestr = strftime("%m%d-%H%M")

    num_samples_run = (int(float(x)) for x in num_samples_run)
    num_samples_run = list(num_samples_run)
    # num_samples_run = num_samples_run * 20
    # num_rej_run = (int(float(x)) for x in num_rej_run)
    # num_rej_run = list(num_rej_run)
    # num_rej_run = num_rej_run*10

    results = np.zeros((len(programs), len(inference), len(num_samples_run), len(num_rej_run)), dtype=object)

    for p in range(len(programs)):
        for i in range(len(inference)):
            for j in range(len(num_samples_run)):
                for k in range(len(num_rej_run)):
                    ast = load_program(daphne_dir, daphne_prog(programs[p]), json_prog(programs[p]), mode='desugar-hoppl-cps', compile=compile)
                    if inference[i] == "rejSMC" or inference[i] == "SMC":
                        start_time = time()
                        try:
                            samples, ess = get_samples(ast, num_samples_run[j], num_rej_run[k], tmax=tmax, inference=inference[i], folder=timestr, program = programs[p], verbose=verbose)
                        except:
                            samples, ess = get_samples(ast, num_samples_run[j], num_rej_run[k], tmax=tmax, inference=inference[i], folder=timestr, program = programs[p], verbose=verbose)
                        samples = tc.stack(samples).type(tc.float)
                        print('Sample mean:', samples.mean(axis=0))
                        print('Sample standard deviation:', samples.std(axis=0))
                        end_time = time()-start_time
                        print(end_time)
                        results[p,i,j,k] = [samples, ess, end_time, num_samples_run[j], num_rej_run[k]]

    os.mkdir('data/{}'.format(timestr))
    with open('data/{}/results.pkl'.format(timestr), 'wb') as f:
        pickle.dump(results, f)


@hydra.main(version_base=None, config_path='', config_name='config')
def run_all(cfg):

    # Configuration
    wandb_run = cfg['wandb_run']
    num_samples = int(cfg['num_samples'])
    num_samples_run = cfg['num_samples_run']
    num_rej_run = cfg['num_rej_run']
    tmax = cfg['tmax']
    compile = cfg['compile']
    base_dir = cfg['base_dir']
    daphne_dir = cfg['daphne_dir']
    seed = cfg['seed']
    recursion_limit = cfg['recursion_limit']
    inference = cfg['inference']
    if inference == 'None': inference = None

    # Calculations
    sys.setrecursionlimit(recursion_limit)
    if (seed != 'None'): tc.manual_seed(seed)

    # W&B init
    # if wandb_run: wandb.init(project='HW6', entity='cs532-2022', name = 'elvis-SMC-{:.0e}'.format(Decimal(num_samples)))
    # if wandb_run: wandb.init(project='HW6', entity='cs532-2022', name = 'elvis-IS')
    if wandb_run: wandb.init(project='project', entity='elvis_cai', name = 'elvis-project')


    # Homework 6
    programs = cfg['homework6_programs']
    run_programs(programs, prog_set='homework_6', base_dir=base_dir, daphne_dir=daphne_dir,
        num_samples=num_samples, num_rej_run = num_rej_run, num_samples_run = num_samples_run, tmax=tmax, inference=inference, compile=compile, wandb_run=wandb_run, verbose=False)

    # W&B finalize
    if wandb_run: wandb.finish()

if __name__ == '__main__':
    run_all()