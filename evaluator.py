# # Standard imports
# import torch as tc
# from pyrsistent import pmap
# from time import time
# import sys

# # Project imports
# from primitives import primitives

# class Env():
#     "An environment: a dict of {'var': val} pairs, with an outer Env."
#     def __init__(self, parms=(), args=(), outer=None):
#         self.data = pmap(zip(parms, args))
#         self.outer = outer
#         if outer is None:
#             self.level = 0
#         else:
#             self.level = outer.level+1

#     def __getitem__(self, item):
#         return self.data[item]

#     def find(self, var):
#         "Find the innermost Env where var appears."
#         if (var in self.data):
#             return self
#         else:
#             if self.outer is not None:
#                 return self.outer.find(var)
#             else:
#                 raise RuntimeError('var "{}" not found in outermost scope'.format(var))

#     def print_env(self, print_lowest=False):
#         print_limit = 1 if print_lowest == False else 0
#         outer = self
#         while outer is not None:
#             if outer.level >= print_limit:
#                 print('Scope on level ', outer.level)
#                 if 'f' in outer:
#                     print('Found f, ')
#                     print(outer['f'].body)
#                     print(outer['f'].parms)
#                     print(outer['f'].env)
#                 print(outer,'\n')
#             outer = outer.outer



# class Procedure(object):
#     'A user-defined HOPPL procedure'
#     def __init__(self, params:list, body:list, sig:dict, env:Env):
#         self.params, self.body, self.sig, self.env = params, body, sig, env
#     def __call__(self, *args):
#         return eval(self.body, self.sig, Env(self.params, args, self.env))

# ### I set k.sig because self.sig is used here. I want self.sig = sig

# def standard_env():
#     "An environment with some Scheme standard procedures."
#     env = Env(primitives.keys(), primitives.values())
#     return env

# def eval(e, sig:dict, env:Env, verbose=False):


#     if isinstance(e, bool):
#         t = tc.tensor(e).float()
#         return t
    
#     elif isinstance(e, int) or isinstance(e, float):
#         return tc.tensor(e).float()

#     elif tc.is_tensor(e):
#         return e

#     elif isinstance(e, str):
#         try:
#             ### If already defined, return corresponding value
#             return env.find(e)[e]

#         except:
#             # sig['address'] = sig['address'] + "-" + e
#             addr = sig['address'] + "-" + e
#             sig.set('address', addr)
#             ### If not, evaluate it below
#             return e

#     op, *args = e

#     if op == "if":
#         (test, conseq, alt) = args
#         if eval(test, sig, env):
#             exp = conseq
#         else:
#             exp = alt

#         return eval(exp, sig, env)

#     elif op == "sample" or op == "sample*":
#         d = eval(args[1], sig, env)
#         s = d.sample()
#         k = eval(args[2], sig, env)
#         addr = eval(args[0], sig, env)
#         num_sample_statement = sig['num_sample_state']

#         sig = sig.update({'address': addr})
#         sig = sig.update({'dist':d})
#         sig = sig.update({'type':"sample"})
#         sig = sig.update({'num_sample_state': num_sample_statement+tc.tensor(1.)})

#         # k.sig = sig

#         return k, [s], sig

#     elif op == "observe" or op == "observe*":
#         d = eval(args[1], sig, env)
#         v = eval(args[2], sig, env)
#         k = eval(args[3], sig, env)
#         logp = d.log_prob(v)
#         addr = eval(args[0], sig, env)
#         logW = sig['logW']

#         sig = sig.set('address', addr)
#         sig = sig.set('logW', logW+logp)
#         sig = sig.set('type', 'observe')
#         sig = sig.set('dist', d)
#         # k.sig = sig

      
#         return k, [v], sig


#     elif op == "fn":
#         (parms, body) = args
#         return Procedure(parms, body, sig, env)

    
#     else:
#         proc = eval(op, sig, env)
#         args = []
#         for arg in e[1:]:
#             args.append(eval(arg, sig, env))

#         if isinstance(proc, str):
#             raise Exception("{} is not a procedure".format(proc))

#         return proc(*args)


# def evaluate(ast:dict, sig=None, run_name='start', verbose=False):
#     '''
#     Evaluate a HOPPL program as desugared by daphne
#     Args:
#         ast: abstract syntax tree
#     Returns: The return value of the program
#     '''
#     if sig is None: sig = pmap({'logW':tc.tensor(-10.0), 'type': None, 'address': "start", 'num_sample_state': tc.tensor(0.0)})
#     env = standard_env()
#     output = lambda x: x 
#     exp = eval(ast, sig, env, verbose)(run_name, output)
#     while type(exp) is tuple: 
#         func, args, sig = exp
#         exp = func(*args)
#     return exp, sig 


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
                log_prob = dist.log_prob(sample)
                sig['logP'] += log_prob
            if 'location' in sig.keys(): sig['location'] = 'sample'
            if 'address' in sig.keys(): sig['address'] = eval(e[1], sig, env)
            if 'dist' in sig.keys(): sig['dist'] = dist
            if 'type' in sig.keys(): sig['type'] =  "sample"
            result = cont, [sample], sig

        elif e[0] == 'observe': # Probabilistic program observe
            dist = eval(e[2], sig, env)
            obs = eval(e[3], sig, env)
            cont = eval(e[4], sig, env)
            if ('logP' in sig.keys()) or ('logW' in sig.keys()):
                log_prob = dist.log_prob(obs)
                if 'logP' in sig.keys(): sig['logP'] += log_prob
                if 'logW' in sig.keys(): sig['logW'] += log_prob
            if 'location' in sig.keys(): sig['location'] = 'observe'
            if 'address' in sig.keys(): sig['address'] = eval(e[1], sig, env)
            if 'dist' in sig.keys(): sig['dist'] = dist
            if 'type' in sig.keys(): sig['type'] =  "observe"
            result = cont, [obs], sig

        elif e[0] == 'fn': # Function definition
            _, params, body = e
            result = Procedure(params, body, sig, env) # NOTE: This returns an object

        else: # Function
            func = eval(e[0], sig, env)
            args = [eval(arg, sig, env) for arg in e[1:]]
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