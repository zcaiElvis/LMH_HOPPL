import torch
import torch.distributions as dist


def p_C(c:torch.Tensor)->torch.Tensor:
    probs = torch.tensor([0.5,0.5])
    d = dist.Categorical(probs)
    return torch.exp(d.log_prob(c))


def p_S_given_C(s: torch.Tensor,c: torch.Tensor) -> torch.Tensor:
    probs = torch.tensor([[0.5, 0.9], [0.5, 0.1]])
    d = dist.Categorical(probs.t())
    lp = d.log_prob(s)[c.detach()]
    return torch.exp(lp)


def p_R_given_C(r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    probs = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    d = dist.Categorical(probs.t())
    lp = d.log_prob(r)[c.detach()]
    return torch.exp(lp)

def p_W_given_S_R(w: torch.Tensor, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    probs = torch.tensor([
        [[1.0, 0.1], [0.1, 0.01]],  # w = False
        [[0.0, 0.9], [0.9, 0.99]],  # w = True
    ])
    return probs[w.detach(), s.detach(), r.detach()]


def getprob(args):
    cloudy = args[0]
    rainy = args[1]
    p = torch.zeros((2,2,2,2))
    for c in range(2):
        for s in range(2):
            for r in range(2):
                for w in range(2):
                    p[c, s, r, w] = p_C(torch.tensor(c))*p_S_given_C(torch.tensor(s), torch.tensor(c))*p_R_given_C(torch.tensor(r), torch.tensor(c))*p_W_given_S_R(torch.tensor(w), torch.tensor(s), torch.tensor(r))
    return p[cloudy, 1, rainy, 1]