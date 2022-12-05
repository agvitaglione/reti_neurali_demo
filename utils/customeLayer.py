import torch
import binarize as bin
import util

class BinarizeLayer(torch.nn.Module):

    def __init__(self, binFunction):

        super(BinarizeLayer, self).__init__()
        self.binFunction = binFunction

    def forward(self, x):
        return self.binFunction(x)


class LinearBin(torch.nn.Module):

    def __init__(self, weight, bias=None, threshold=0, device="cpu"):

        super(LinearBin, self).__init__()

        self.weight = torch.nn.Parameter(weight)
        self.bias = bias
        self.threshold = threshold

        # Layers
        self.bl = bin.binarize101


    def forward(self, x):

        # Binarizzazione dei pesi
        old = bin.Binarize101.threshold
        bin.Binarize101.threshold = self.threshold
        wb = self.bl(self.weight)
        bin.Binarize101.threshold = old

        #---------- FOR REPORT
        # weights = [[x for x in lista if x != 0] for lista in wb.int().tolist()]
        # util.report(f"------\nWeights exit\n{weights}\n")

        # Calcolo output
        return torch.nn.functional.linear(x, wb, self.bias)

    def setThreshold(self, threshold):
        self.threshold = threshold


class BatchNorm1D(torch.nn.Module):

    def __init__(self, input_size, device="cpu"):

        super(BatchNorm1D, self).__init__()

        self.device = device

        # Liste che contengono la media e la varianza di ogni feature per ogni batch
        self.EPList = []        
        self.VarPList = []

        self.gamma = torch.empty(1, device=device)
        torch.nn.init.uniform_(self.gamma)
        self.beta = torch.zeros(input_size, device=device)
        self.gamma = torch.nn.Parameter(self.gamma)
        self.beta = torch.nn.Parameter(self.beta)

        # Statistiche della popolazione da settare per l'inferenza
        self.EP = torch.nn.Parameter(torch.empty(input_size, device=device), requires_grad=False)
        self.VarP = torch.nn.Parameter(torch.empty(input_size, device=device), requires_grad=False)       

        # Settare self.inference = True prima della fase di inferenza in modo da utilizzare i parametri
        # della popolazione. Reimpostarlo a False prima della fase di training.    
        self.inference = False

    def forward(self, x):

        # Algoritmo del paper
        eps = 1e-5
        
        if self.inference:

            b_num = self.EP * self.gamma
            b_den = torch.sqrt(self.VarP + eps).pow_(-1)
            b = self.beta - b_num * b_den
            w = self.gamma * torch.sqrt(self.VarP + eps).pow_(-1) 
            b = b.floor_()
            y = (bin.AP2(w) * x).floor_() + b            # Moltiplicazione -> shift register (mul per potenza 2)
            #y = w * x + b

        else:
                
            mu = torch.mean(x, dim=0)   # Media per colonna (media delle uscite di un percettrone sul mini-batch)
            var = torch.var(x, dim=0)
            z = (x - mu) / torch.sqrt(var + eps)
            y = self.gamma * z + self.beta

            # Salvataggio dei valori per il calcolo della statistica della popolazione
            self.EPList.append(mu.detach_())
            self.VarPList.append(var.detach_())
            
        return y

    def frozeParameters(self, batch_size):

        # Calcola i parametri della popolazione a partire dai parametri dei batch memorizzati durante il forward 
        # nelle variabili EPList e VarPList.
        # Va invocato dopo aver terminato la fase di traning.

        del self.EP
        del self.VarP

        eP = torch.stack(self.EPList, dim=0).to(self.device)
        varP = torch.stack(self.VarPList, dim=0).to(self.device)

        self.EP = torch.nn.Parameter(torch.mean(eP, dim=0), requires_grad=False)
        self.VarP = torch.nn.Parameter(torch.mean(varP, dim=0).mul_(batch_size / (batch_size - 1)), requires_grad=False)

        self.EPList.clear()
        self.VarPList.clear()

    def getParameter(self):

        # Restituzione di a e b del livello di batchnorm.
        # a è una potenza di 2.
        # b è un intero approssimato per difetto. 

        eps = 1e-5
        b_num = self.EP * self.gamma
        b_den = torch.sqrt(self.VarP + eps).pow_(-1)
        b = self.beta - b_num * b_den
        w = self.gamma * torch.sqrt(self.VarP + eps).pow_(-1) 
        b = b.floor_()
        return bin.AP2(w), b            # Moltiplicazione -> shift register (mul per potenza 2)

# ------------------------------------------------
# For concolutional model

class ConvBinLayer(torch.nn.Module):

    def __init__(self, weights, device="cpu", binfunction=bin.binarize11):

        super(ConvBinLayer, self).__init__()

        self.device = device

        # weights (size_out x size_in x kernel_size)
        self.weights = torch.nn.Parameter(weights)

        self.bl = BinarizeLayer(binfunction)

    def forward(self, x):

        # Livello di binarizzazione dei pesi
        wb = self.bl(self.weights)

        # Convoluzione
        return torch.nn.functional.conv2d(x, wb)

class LazyBatchNorm2D(torch.nn.Module):

    def __init__(self, device="cpu"):

        super(LazyBatchNorm2D, self).__init__()

        self.device = device

        # Liste che contengono la media e la varianza di ogni feature per ogni batch
        self.EPList = []
        self.VarPList = []

        self.inference = False
        self.__firstForward = True

    # Da invocare al primo forward per il calcolo dei parametri
    def lazyInit(self, x):

        self.output_shape = x.shape[1:]
        self.output_depth , self.output_height, self.output_width = self.output_shape

        # Parametri di training
        self.gamma = torch.empty(self.output_depth, device=self.device)
        torch.nn.init.uniform_(self.gamma)
        self.beta = torch.zeros(self.output_shape, device=self.device)
        self.gamma = torch.nn.Parameter(self.gamma)
        self.beta = torch.nn.Parameter(self.beta)

        self.__firstForward = False

    def forward(self, x):

        if self.__firstForward:
            self.lazyInit(x)

        eps = 1e-5
        
        if self.inference:

            b_num = self.EP * self.gamma
            b_den = torch.sqrt(self.VarP + eps).pow_(-1)
            b = self.beta.reshape(len(self.beta), -1) - (b_num * b_den).unsqueeze_(1)
            b = b.reshape(len(b), self.output_height, self.output_width)
            w = self.gamma * torch.sqrt(self.VarP + eps).pow_(-1) 
            y = bin.AP2(w).unsqueeze_(1) * x.reshape(len(x), self.output_depth, -1) 
            y = y.reshape(len(y), self.output_depth, self.output_height, self.output_width) + b           # Moltiplicazione -> shift register (mul per potenza 2)

        else:

            # Calcolo della media per ogni filtro (su tutto il filtro) su tutto il batch
            x_resh = x.permute(1,0,2,3).reshape(self.output_depth, -1)
            mu =  torch.mean(x_resh, dim=1)  
            var = torch.var(x_resh, dim=1)

            x_resh = x.reshape(len(x), self.output_depth, -1)
            z = (x_resh - mu.unsqueeze_(1)) / torch.sqrt(var.unsqueeze_(1) + eps)
            y = torch.unsqueeze(self.gamma, 1) * z
            y = y.reshape(len(x), self.output_depth, self.output_height, self.output_width)
            y = y + self.beta

            self.EPList.append(mu.detach_())
            self.VarPList.append(var.detach_())
            
        return y

    def frozeParameters(self, batch_size):
        eP = torch.stack(self.EPList, dim=0).to(self.device)
        varP = torch.stack(self.VarPList, dim=0).to(self.device)

        #TODO EP e VarP Parametri
        self.EP = torch.mean(eP, dim=0).view(-1)
        self.VarP = torch.mean(varP, dim=0).mul_(batch_size / (batch_size - 1)).view(-1)

        self.EPList.clear()
        self.VarPList.clear()