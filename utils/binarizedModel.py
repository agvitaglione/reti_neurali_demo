import torch
import customeLayer as cl
import binarize as bin
import util


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
            fcLayers = cl.LinearBin(self.weights[k], device=device)
            bnLayers = cl.BatchNorm1D(len(self.weights[k]), device=device)
            self.binLayers.append(binLayers)
            self.fcLayers.append(fcLayers)
            self.bnLayers.append(bnLayers)

        self.binLayers = torch.nn.ModuleList(self.binLayers)
        self.fcLayers = torch.nn.ModuleList(self.fcLayers)
        self.bnLayers = torch.nn.ModuleList(self.bnLayers)

        self.softMax = torch.nn.Softmax(dim=1)
        # --------------------------------------------------


    def setInference(self, inference: bool):
        for b in self.bnLayers:
            b.inference = inference

    def frozeParameter(self, batch_size):
        for b in self.bnLayers:
            b.frozeParameters(batch_size)

    def forward(self, x_vect):

        out = x_vect - 0.5                  # IMPORTANTE LA NORMALIZZAZIONE
        # util.report("##### FORWARD PHASE #####\n\n")
        for k in range(self.num_layers):
            # util.report(f"Level\t{k}\nInput\t{out.tolist()}\n")
            out = self.binLayers[k](out)
            # util.report(f"------\nBin exit\n{((out + 1)/2).int().tolist()}\n")
            out = self.fcLayers[k](out)
            # util.report(f"------\nFC exit\t{out.int().tolist()}\n")
            out = self.bnLayers[k](out)
            # util.report(f"------\nBN exit\n{out.int().tolist()}\n---------------------------\n\n")
    
        return self.softMax(out)

    # Da usare per forward su modelli binarizzati pruned
    def prediction(self, x, pruned=False, threshold_list=[0., 0., 0., 0., 0.]):
        
        self.setInference(True)

        if pruned:
            for i, layer in enumerate(self.fcLayers):
                layer.setThreshold(threshold_list[i])

        out = self(x)
        _, pred = torch.max(out, -1)

        if pruned:
            for layer in self.fcLayers:
                layer.setThreshold(0)
                
        return out, pred

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

    def prune(self, threshold_list, training_loader, test_loader, epochs=50, lr=0.01, writer = None, PATH = None):

        for i, layer in enumerate(self.fcLayers):
            layer.setThreshold(threshold_list[i])

        self.trainModel(training_loader, test_loader, epochs, lr, writer, PATH)

        for layer in self.fcLayers:
            layer.setThreshold(0)

    def getAccuracy(self, test_loader, pruned=False, threshold_list=[0.,0.,0.,0.,0.]):
        
        self.setInference(True)

        if pruned:
            for i, layer in enumerate(self.fcLayers):
                layer.setThreshold(threshold_list[i])

        acc = util.getAccuracy(self, test_loader, self.device, dim=1)

        if pruned:
            for layer in self.fcLayers:
                layer.setThreshold(0)
        
        return acc 

    def getPruningPercentage(self):

        countZero = 0
        total = 0

        for g in self.parameters():
            if(len(g.shape) > 1):
                total += g.shape[0] * g.shape[1]
                r = torch.where(torch.eq(g, 0), 1., 0.)
                countZero += torch.sum(r).item()

        return countZero / total * 100

# -----------------------------------------------------