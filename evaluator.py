# Standard imports
import torch as tc
from pyrsistent import pmap
from time import time
import sys

# Project imports
from primitives import primitives

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
class Env(object):
        'An environment: a persistent map of {var: val} pairs, with an outer environment'
        def __init__(self, params=(), args=(), outer=None):
            self.env = pmap()
            # if params == ['alpha', 'dontcare0', 'k50']:
            #     print("creating k50 dictionary, check args")
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
                    raise ValueError('Outer limit of environment reached')
                else:
                    result = self.outer.find(var)
            return result


class Procedure(object):
    'A user-defined HOPPL procedure'
    def __init__(self, params:list, body:list, sig:dict, env:Env):
        self.params, self.body, self.sig, self.env = params, body, sig, env
    def __call__(self, *args):
        newenv = Env(self.params, args, self.env)
        result = eval(self.body, self.sig, newenv)
        return result

### I set k.sig because self.sig is used here. I want self.sig = sig

def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env(primitives.keys(), primitives.values())
    # env= Env()
    # env.update(primitives)
    return env

def eval(e, sig:dict, env:Env, verbose=False):

    # if e == ['conj', 'cps51', 'states', 'state', 'k50']:
    #     print("now at conjugate step")


    if isinstance(e, bool):
        t = tc.tensor(e).float()
        return t
    
    elif isinstance(e, int) or isinstance(e, float):
        return tc.tensor(e).float()

    elif tc.is_tensor(e):
        return e

    elif type(e) is str: # Strings
        if (e[0] == '"' and e[-1] == '"'): # 'case c' with string
            return str(e[1:-1])
        elif e[0:4] == 'addr': # Addressing
            return e
        else: # 'case v' look-up variable in environment
            lookup = env.find(e)[e]
            return lookup

    op, *args = e

    if op == "if":
        (test, conseq, alt) = args
        if eval(test, sig, env):
            exp = conseq
        else:
            exp = alt

        return eval(exp, sig, env)

    elif op == "sample" or op == "sample*":
        d = eval(args[1], sig, env)
        s = d.sample()
        k = eval(args[2], sig, env)
        addr = eval(args[0], sig, env)
        num_sample_statement = sig['num_sample_state']
        logw = sig['logW']

        sig = sig.update({'address': addr})
        sig = sig.update({'dist':d})
        sig = sig.update({'type':"sample"})
        sig = sig.update({'num_sample_state': num_sample_statement+tc.tensor(1.)})
        sig = sig.update({'logW': logw})

        k.sig = sig

        return k, [s], sig

    elif op == "observe" or op == "observe*":
        d = eval(args[1], sig, env)
        v = eval(args[2], sig, env)
        k = eval(args[3], sig, env)
        logp = d.log_prob(v)
        addr = eval(args[0], sig, env)
        logW = sig['logW']

        sig = sig.set('address', addr)
        sig = sig.set('logW', logW+logp)
        sig = sig.set('type', 'observe')
        sig = sig.set('dist', d)
        k.sig = sig

      
        return k, [v], sig


    elif op == "fn":
        parms, body = args
        # print(parms)
        # if parms == ['alpha', 'dontcare0', 'k50']:
        #     print("creating k50 dictionary, check args")
        return Procedure(parms, body, sig, env)

    
    else:
        proc = eval(op, sig, env)
        args = []
        for arg in e[1:]:
            if arg == "k50":
                print("heres k50")
            arg_result = eval(arg, sig, env)
            args.append(arg_result)

        if isinstance(proc, str):
            raise Exception("{} is not a procedure".format(proc))

        result = proc(*args)

        return result


def evaluate(ast:dict, sig=None, run_name='start', verbose=False):
    '''
    Evaluate a HOPPL program as desugared by daphne
    Args:
        ast: abstract syntax tree
    Returns: The return value of the program
    '''
    if sig is None: sig = pmap({'logW':tc.tensor(0.0), 'type': None, 'address': "start", 'num_sample_state': tc.tensor(0.0)})
    env = standard_env()
    output = lambda x: x 
    exp = eval(ast, sig, env, verbose)(run_name, output)
    while type(exp) is tuple: 
        func, args, sig = exp
        exp = func(*args)
    return exp, sig 





# change alex code to persistance