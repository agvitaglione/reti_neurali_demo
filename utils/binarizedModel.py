import torch
import customeLayer as cl
import binarize as bin
import util as util


# Classe per la binarizzazione di un modello lineare
class MNISTBinarizedModel(torch.nn.Module):
    
    # Model: MNIST
    def __init__(self, model=None, device="cpu"):

        super(MNISTBinarizedModel, self).__init__()

        self.device = device
        self.num_layers = 5
        self.weights = []

        # Parametri del modello dal binarizzare
        if model is not None:
            for g in model.parameters():
                self.weights.append(g)
        else:
            self.weights.append(torch.empty(256,784, device=self.device))
            self.weights.append(torch.empty(256,256, device=self.device))
            self.weights.append(torch.empty(256,256, device=self.device))
            self.weights.append(torch.empty(256,256, device=self.device))
            self.weights.append(torch.empty(10, 256, device=self.device))

        # Layers
        # Struttura della rete:
        # 1 - livello di binarizzazione delle feature di input
        # 2 - livello fully connected
        # 3 - livello di batch normalizazion
        self.binLayers = []
        self.fcLayers = []
        self.bnLayers = []

        for k in range(self.num_layers):
            binLayers = cl.BinarizeLayer(bin.binarize11)
            fcLayers = cl.LinearBin(self.weights[k], device=device, binFunction=bin.binarize11)
            bnLayers = cl.BatchNorm1D(len(self.weights[k]), device=device)
            self.binLayers.append(binLayers)
            self.fcLayers.append(fcLayers)
            self.bnLayers.append(bnLayers)

        self.binLayers = torch.nn.ModuleList(self.binLayers)
        self.fcLayers = torch.nn.ModuleList(self.fcLayers)
        self.bnLayers = torch.nn.ModuleList(self.bnLayers)

        self.softMax = torch.nn.Softmax(dim=0)
        # --------------------------------------------------


    def setInference(self, inference: bool):
        for b in self.bnLayers:
            b.inference = inference

    def frozeParameter(self, batch_size):
        for b in self.bnLayers:
            b.frozeParameters(batch_size)

    def forward(self, x_vect):

        out = x_vect - 0.5                  # IMPORTANTE LA NORMALIZZAZIONE
        #util.report("##### FORWARD PHASE #####\n\n")
        for k in range(self.num_layers):
            #util.report(f"Level\t{k}\nInput\t{out.tolist()}\n")
            out = self.binLayers[k](out)
            #util.report(f"------\nBin exit\n{((out + 1)/2).int().tolist()}\n")
            out = self.fcLayers[k](out)
            #util.report(f"------\nFC exit\t{out.int().tolist()}\n")
            out = self.bnLayers[k](out)
            #util.report(f"------\nBN exit\n{out.int().tolist()}\n---------------------------\n\n")
    
        return self.softMax(out)

    def trainModel(self, training_loader, test_loader, epochs=50, lr=0.01, writer = None, PATH = None):
        
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        maxAcc = 0

        for epoch in range(epochs):

            loss = None

            for i, (images, labels) in enumerate(training_loader):
                
                images = images.reshape(len(images), -1)

                images = images.to(self.device)
                labels = labels.to(self.device)

                self.setInference(False)
                predictions = self(images)

                labels_expand = torch.zeros(len(labels), 10, device=self.device)
                for j in range(len(labels)):
                    labels_expand[j][labels[j]] = 1
                labels = labels_expand

                loss = loss_function(predictions, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i + 1)% 100 == 0:
                    print(f"epoch = {epoch+1}/{epochs}, step = {i + 1}/{len(training_loader)}, loss = {loss}")

            else:
                self.frozeParameter(len(images))
                self.setInference(True)

            # Evaluate accuracy on test dat 
            accVal = self.getAccuracy(test_loader)
            print(f"Validation Accuracy = {accVal}%")

            if PATH is not None and accVal > maxAcc:
                maxAcc = accVal
                util.save_model(self, PATH)

            # Write result on tensorboard
            if writer is not None:
                writer.add_scalar("Training loss", loss, epoch)
                writer.add_scalar("Validation accuracy", accVal, epoch) 

            # Update learning rate
            for g in optimizer.param_groups:
                g['lr'] = lr * 0.95

    def prune(self, threshold, training_loader, test_loader, epochs=50, lr=0.01, writer = None, PATH = None):

        for layer in self.fcLayers:
            bin.Binarize101.threshold = threshold
            layer.setBinFunction(bin.binarize101)

        self.trainModel(training_loader, test_loader, epochs, lr, writer, PATH)

        for layer in self.fcLayers:
            bin.Binarize101.threshold = 0
            layer.setBinFunction(bin.binarize11)


    def getAccuracy(self, test_loader, pruned=False, threshold=0):
        
        self.setInference(True)

        if pruned:
            for layer in self.fcLayers:
                bin.Binarize101.threshold = threshold
                layer.setBinFunction(bin.binarize101)

        acc = util.getAccuracy(self, test_loader, self.device, dim=1)

        if pruned:
            for layer in self.fcLayers:
                bin.Binarize101.threshold = 0
                layer.setBinFunction(bin.binarize11)
        
        return acc 

    def prediction(self, x, pruned=False, threshold=0):
        
        self.setInference(True)

        if pruned:
            for layer in self.fcLayers:
                bin.Binarize101.threshold = threshold
                layer.setBinFunction(bin.binarize101)

        out = self(x)
        _, pred = torch.max(out, -1)

        if pruned:
            for layer in self.fcLayers:
                bin.Binarize101.threshold = 0
                layer.setBinFunction(bin.binarize11)
        return out, pred

# -----------------------------------------------------


# ------------------------------------------------------------------------------------------------------
# NOT USED 


# Classe per la binarizzazione di un modello lineare
class BinarizedModelPRIMO(torch.nn.Module):
    
    def __init__(self, model, device="cpu"):
        super(BinarizedModelPRIMO, self).__init__()
        self.device = device
        self.total_weights = 0              # Numero di pesi totali nella rete
        self.hidden_layer = -1              # Numero di livelli nascosti
        self.number_perceptron_layer = []   # Numero di percettroni in ogni livello
        self.in_perceptron_layer = []       # Numero di ingressi del percettrone del livello i

        for x in model.parameters():
            
            # Non vengono considerati livelli intermedi (solo info ingresso, nessuna uscita)
            if len(x.shape) > 1:  
                self.in_perceptron_layer.append(x.shape[1])
                self.number_perceptron_layer.append(x.shape[0])
                self.total_weights += x.shape[0] * x.shape[1]
                self.hidden_layer += 1

        self.L = self.hidden_layer + 1     # numero di livelli della rete

        self.weights = []
        for x in model.parameters():
            if len(x.shape) > 1:
                weights = torch.empty(x.shape[1], x.shape[0], device=device)  # Numero di Pesi x Numero di Percettroni
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        weights[j][i] = x[i][j].item()
                self.weights.append(weights)

        # Paramatri per la backnormalization
        self.gamma = torch.empty(self.L, device=device)
        self.beta = torch.zeros(self.L, device=device)
        torch.nn.init.uniform_(self.gamma)

        # Parametri trainable
        self.weights = torch.nn.ParameterList(self.weights)       # Pesi della rete
        self.beta = torch.nn.Parameter(self.beta)                 # Beta e Gamma per la backnorm
        self.gamma = torch.nn.Parameter(self.gamma)

        # Parametri non trainable
        # Lista delle medie sui batch delle uscite dei percettroni per ogni livello
        self.EP = [torch.empty(self.number_perceptron_layer[k], device=device) for k in range(self.L)]
        self.EP = torch.nn.ParameterList(self.EP).requires_grad_(False)
        # Lista delle varianze delle uscite dei percettroni per ogni livello
        self.VarP = [torch.empty(self.number_perceptron_layer[k], device=device) for k in range(self.L)]
        self.VarP = torch.nn.ParameterList(self.VarP).requires_grad_(False)
                
        # Fuzione di attivazione
        self.softMax = torch.nn.Softmax(dim=0)

        self.init() 
    
    # Funzione per la inizializzazione dei parametri
    # Da invocare prima di iniziare il traning
    def init(self): 
        # Variabili utili per il forward                  
        self.W = self.weights
        self.Wb = [None] * (self.L)
        self.s = [None] * (self.L)
        self.a = [None] * (self.L + 1)
        self.ab = [None] * (self.L + 1)

        # Traccia media popolazione traning
        self.EPList = [[] for i in range(self.L)]
        self.VarList = [[] for i in range(self.L)]

        # inference = false
        #   calcolo delle statistiche della popolazione
        #   da impostare prima della fase di traning
        # inference = true
        #   utilizzo delle statistiche della popolazione calcolate
        #   da impostare prima della fase di inferenza
        self.inference = False 

    def forward(self, x_vect):

        # Normalizzazione dei dati di input [-1; 1]
        self.a[-1] = x_vect - 0.5    

        # Binarizzazione dei dati di input
        self.ab[-1] = bin.binarize11(self.a[-1])    
        ab = self.ab[-1]
        
        for k in range(self.L):

            # --------------- Paper

            # Binarizzazione dei pesi
            self.Wb[k] = bin.binarize11(self.W[k])

            # Prodotto scalare pesi x input    
            self.s[k] = torch.matmul(ab, self.Wb[k])

            # Livello di batch normalization
            if not self.inference:
                self.a[k] = bin.batchNorm(self.s[k], self.gamma[k], self.beta[k])
            else:
                self.a[k] = bin.batchNorm(self.s[k], self.gamma[k], self.beta[k], True, self.EP[k], self.VarP[k])

            # Livello di binarizzazione (l'ultimo livello non è binarizzato)
            if k < self.L - 1:
                self.ab[k] = bin.binarize11(self.a[k])
            ab = self.ab[k]
            # ---------------

            # ------------------- traccia media popolazione traning
            if not self.inference:
                self.EPList[k].append(torch.mean(self.s[k], dim=0))
                self.VarList[k].append(torch.var(self.s[k], dim=0))
            # -------------------

        return self.softMax(self.a[self.L - 1])

    # Salvataggio dei parametri della popolazione
    def frozeParameter(self, batch_size):

        for k in range(self.L):
            eP = torch.stack(self.EPList[k], dim=0).to(self.device)
            varP = torch.stack(self.VarList[k], dim=0).to(self.device)
            self.EP[k] = torch.mean(eP, dim=0)
            self.VarP[k] = torch.mean(varP, dim=0).mul_(batch_size / (batch_size - 1))

        # Cencellazione dei parametri di traning utilizzati per il calcolo delle statistiche
        # della popolazione
        self.EPList = [[] for i in range(self.L)]
        self.VarList = [[] for i in range(self.L)]


    def trainModel(self, training_loader, test_loader, epochs=100, lr=0.01, writer = None, PATH = None):
        
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.init()
        maxAcc = 0
        

        for epoch in range(epochs):

            losscount = 0

            loss = None

            for i, (images, labels) in enumerate(training_loader):

                # Forward phase
                images = images.reshape(len(images), -1).to(self.device)
                labels = labels.to(self.device)
                self.inference = False
                predictions = self(images)

                labels_pred = torch.zeros(len(labels), 10, device=self.device)
                for j in range(len(labels)):
                    labels_pred[j][labels[j]] = 1

                loss = loss_function(predictions, labels_pred)
                losscount += loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0:
                    print(f"epoch = {epoch+1}/{epochs}, step = {i}/{len(training_loader)}, loss = {loss}")

            else:
                # Evaluate accuracy on test dat 
                self.inference = True
                self.frozeParameter(len(images))
                acc = util.getAccuracy(self, test_loader, self.device)
                print("acc = ", acc ,"%")

            if PATH != None and acc > maxAcc:
                maxAcc = acc
                torch.save(self.state_dict(), PATH)

            # Update learning rate
            for g in optimizer.param_groups:
                g['lr'] = lr * 0.95

            # Write result on tensorboard
            if writer != None:
                writer.add_scalar("Training loss", losscount / len(training_loader), epoch)
                writer.add_scalar("Validation accuracy", acc, epoch)  
# -----------------------------------------------------