import torch
import util as util

# Funzione di binarizzazione (-1, 1)
class Binarize11(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.where(torch.ge(x, 0), 1., -1.)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)
        
binarize11 = Binarize11.apply
# -----------------------------------------------------

# Funzione di binarizzazione (0, 1)
class Binarize01(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.where(torch.ge(x, 0), 1., 0.)

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)
        
binarize01 = Binarize01.apply
# -----------------------------------------------------

# Funzione di binarizzazione (-1, 0, 1)
class Binarize101(torch.autograd.Function):

    threshold = 0

    @staticmethod
    def forward(ctx, x):
        out = torch.where(torch.ge(torch.abs(x), Binarize101.threshold), x, 0.)
        out = torch.where(torch.gt(out, 0), 1., out)
        out = torch.where(torch.lt(out, 0), -1., out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)
        
binarize101 = Binarize101.apply

# Funzione per approssimare alla potenza di 2 più vicina
def AP2(x):
    return torch.sign(x) * torch.pow(2, torch.round(torch.log2(torch.abs(x))))
# -----------------------------------------------------

# Livello di Batch Normalization
# gamma e beta: parametri di traning
# inference: False fase di traning | True fase di inferenza
# E: media della popolazione
# Var: varianza della popolazione
def batchNorm(x, gamma, beta, inference = False, E = None, Var = None):
    
    eps = 1e-5
    
    if inference:

        b_num = E * gamma
        b_den = torch.sqrt(Var + eps).pow_(-1)
        b = beta - b_num * b_den
        w = gamma * torch.sqrt(Var + eps).pow_(-1) 
        y = AP2(w) * x + b            # Moltiplicazione -> shift register (mul per potenza 2)

    else:
        
        mu = torch.mean(x, dim=0)   # Media per colonna (media delle uscite di un percettrone sul mini-batch)
        var = torch.var(x, dim=0)
        z = (x - mu) / torch.sqrt(var + eps)
        y = gamma * z + beta
        
    return y
# -----------------------------------------------------
