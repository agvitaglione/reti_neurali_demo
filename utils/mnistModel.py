import torch, util

class MnistNetwork(torch.nn.Module):

    def __init__(self, input_size=784, output_size=10, hidden_size=256, device="cpu"):

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device

        super(MnistNetwork, self).__init__()

        self.l1 = torch.nn.Linear(input_size, hidden_size, bias=False, device=device)
        self.a1 = torch.nn.ReLU()

        self.l2 = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.a2 = torch.nn.ReLU()

        self.l3 = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.a3 = torch.nn.ReLU()

        self.l4 = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.a4 = torch.nn.ReLU()

        self.l5 = torch.nn.Linear(hidden_size, output_size, bias=False, device=device)
        self.a5 = torch.nn.ReLU()
    
    def forward(self, x):

        out = self.l1(x)
        out = self.a1(out)

        out = self.l2(out)
        out = self.a2(out)

        out = self.l3(out)
        out = self.a3(out)

        out = self.l4(out)
        out = self.a4(out)

        out = self.l5(out)
        out = self.a5(out)

        return out

    def trainModel(self, training_loader, test_loader, epochs=100, lr=0.01, writer = None, PATH = None):

        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), momentum=0.9, lr=lr)

        maxAcc = 0

        for epoch in range(epochs):

            loss = None

            for i, (images, labels) in enumerate(training_loader):
                
                images = images.reshape(len(images), -1)

                images = images.to(self.device)
                labels = labels.to(self.device)

                # FORWARD
                predictions = self(images)
                loss = loss_function(predictions, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100 == 0:
                    print(f"epoch = {epoch+1}/{epochs}, step = {i + 1}/{len(training_loader)}, loss = {loss}")

            # Evaluate accuracy on test dat 
            accVal = self.getAccuracy(test_loader, device=self.device)
            print(f"Validation accuracy = {accVal}%")

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

    def getAccuracy(self, test_loader, device="cpu"):

        with torch.no_grad():

            acc = 0 
            tot = 0 # number of image processed 

            for images, labels in test_loader:
                
                images = images.reshape(len(images), -1)
                
                images = images.to(device)
                labels = labels.to(device)

                output = self(images)
                _, predictions = torch.max(output, 1)

                acc += (predictions == labels).sum().item()
                tot += len(test_loader)

            acc = 100 * acc / tot         
            return acc