# Standard imports
import torch as tc
from pyrsistent import pmap
from copy import deepcopy

# Project imports
from primitives import primitives

# Parameters
use_pyrsistent = True

if use_pyrsistent:
    class Env(object):
        'An environment: a persistent map of {var: val} pairs, with an outer environment'
        def __init__(self, params=(), args=(), outer=None):
            self.env = pmap()
            self.update(dict(zip(params, args)))
            self.outer = outer
        def __str__(self):
            return self.env.__str__()
        def update(self, dictionary:dict):
            for key, value in dictionary.items():
                self.env = self.env.set(key, value)
        def find(self, var:str):
            'Find the innermost Env where var appears'
            if var in self.env.keys():
                result = self.env
            else:
                if self.outer is None:
                    print('Not found in any environment:', var)
                    raise ValueError('Outer limit of environment reached')
                else:
                    result = self.outer.find(var)
            return result
else:
    class Env(dict):
        'An environment: a dict of {var: val} pairs, with an outer environment'
        def __init__(self, parms=(), args=(), outer=None):
            self.update(zip(parms, args))
            self.outer = deepcopy(outer)
        def find(self, var):
            'Find the innermost Env where var appears'
            if var in self:
                result = self
            else:
                if self.outer is None:
                    print('Not found in any environment:', var)
                    raise ValueError('Outer limit of environment reached')
                else:
                    result = self.outer.find(var)
            return result


class Procedure(object):
    'A user-defined HOPPL procedure'
    def __init__(self, params:list, body:list, sig:dict, env:Env):
        self.params, self.body, self.sig, self.env = params, body, sig, env
    def __call__(self, *args):
        return eval(self.body, self.sig, Env(self.params, args, self.env))


def standard_env():
    'An environment with some standard procedures'
    env = Env()
    env.update(primitives)
    return env


def eval(e, sig:dict, env:Env, trampolining=True, verbose=False):
    '''
    The eval routine
    @params
        e: expression
        sig: side-effects
        env: environment
    '''
    if verbose: print('Expression (before):', e)

    if type(e) in [float, int, bool]: # 'case c' with constants (float, int, bool)
        result = tc.tensor(e, dtype=tc.float)

    elif type(e) is str: # Strings
        if (e[0] == '"' and e[-1] == '"'): # 'case c' with string
            result = str(e[1:-1])
        elif e[0:4] == 'addr': # Addressing
            result = e
        else: # 'case v' look-up variable in environment
            result = env.find(e)[e]

    elif type(e) is list: # Otherwise e should be a list

        if e[0] == 'if': # 'if' case needs to do lazy evaluation (if e1 e2 e3)
            condition = eval(e[1], sig, env)
            if condition:
                consequent = eval(e[2], sig, env)
                result = consequent
            else:
                alternative = eval(e[3], sig, env)
                result = alternative

        elif e[0] == 'sample': # Probabilistic program sample
            dist = eval(e[2], sig, env)
            cont = eval(e[3], sig, env)
            sample = dist.sample()
            if 'logP' in sig.keys():
                sig['logP'] = log_prob + sig['logP']
            if 'address' in sig.keys(): sig['address'] = eval(e[1], sig, env)
            if 'dist' in sig.keys(): sig['dist'] =  dist
            if 'type' in sig.keys(): sig['type'] = "sample"
            result = cont, [sample], sig

        elif e[0] == 'observe': # Probabilistic program observe
            dist = eval(e[2], sig, env)
            obs = eval(e[3], sig, env)
            cont = eval(e[4], sig, env)
            if ('logP' in sig.keys()) or ('logW' in sig.keys()):
                log_prob = dist.log_prob(obs)
                # logp = sig['logP']
                # logw = sig['logW']
                if 'logP' in sig.keys(): sig['logP'] = log_prob + sig['logP']
                if 'logW' in sig.keys(): sig['logW'] = log_prob + sig['logW']
            if 'address' in sig.keys(): sig['address'] = eval(e[1], sig, env)
            if 'dist' in sig.keys(): sig['dist'] =  dist
            if 'type' in sig.keys(): sig['type'] = "observe"
            result = cont, [obs], sig

        # elif e[0] == 'sample': # Probabilistic program sample
        #     dist = eval(e[2], sig, env)
        #     cont = eval(e[3], sig, env)
        #     sample = dist.sample()
        #     if 'logP' in sig.keys():
        #         log_prob = dist.log_prob(sample)
        #         logp = sig['logP']
        #         sig = sig.set('logP', logp+log_prob)
        #     if 'address' in sig.keys(): sig = sig.set('address', eval(e[1], sig, env))
        #     if 'dist' in sig.keys(): sig = sig.set('dist', dist)
        #     if 'type' in sig.keys(): sig = sig.set('type', "sample")
        #     result = cont, [sample], sig

        # elif e[0] == 'observe': # Probabilistic program observe
        #     dist = eval(e[2], sig, env)
        #     obs = eval(e[3], sig, env)
        #     cont = eval(e[4], sig, env)
        #     if ('logP' in sig.keys()) or ('logW' in sig.keys()):
        #         log_prob = dist.log_prob(obs)
        #         # logp = sig['logP']
        #         # logw = sig['logW']
        #         if 'logP' in sig.keys(): sig = sig.set('logP', log_prob + sig['logP'])
        #         if 'logW' in sig.keys(): sig = sig.set('logW',  log_prob + sig['logW'])
        #     if 'address' in sig.keys(): sig = sig.set('address', eval(e[1], sig, env))
        #     if 'dist' in sig.keys(): sig = sig.set('dist', dist)
        #     if 'type' in sig.keys(): sig = sig.set('type', "sample")
        #     result = cont, [obs], sig

        elif e[0] == 'fn': # Function definition
            _, params, body = e
            # if params == ['alpha', 'dontcare0', 'k50']:
            #     print("at function statement")
            result = Procedure(params, body, sig, env) # NOTE: This returns an object

        else: # Function
            func = eval(e[0], sig, env)
            args = []
            for arg in e[1:]:
                # if arg == "k50":
                #     print("here is k50")
                result = eval(arg, sig, env)
                args.append(result)
            # args = [eval(arg, sig, env) for arg in e[1:]]
            if 'location' in sig.keys(): sig['location'] = 'function'
            result = (func, args, sig) if trampolining else func(*args)

    else:
        print('Expression not recognised:', e)
        raise ValueError('Expression not recognised')

    if verbose: 
        print('Expression (after):', e)
        print('Result:', result, type(result))

    return result


def evaluate(ast:dict, sig=None, run_name='start', verbose=False):
    '''
    Evaluate a HOPPL program as desugared by daphne
    Args:
        ast: abstract syntax tree
    Returns: The return value of the program
    '''
    if sig is None: sig = {}
    env = standard_env()
    output = lambda x: x # Identity function, so that output value is identical to output
    exp = eval(ast, sig, env, verbose)(run_name, output) # NOTE: Must run as function with a continuation
    while type(exp) is tuple: # If there are continuations the exp will be a tuple and a re-evaluation needs to occur
        func, args, sig = exp
        exp = func(*args)
    return exp, sig