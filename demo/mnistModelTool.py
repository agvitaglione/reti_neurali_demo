import sys, getopt, os
sys.path.append(f"{os.path.dirname(__file__)}/../utils")
import mnistModel as Model
import util
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# Variabili globali
LOAD_PATH = None
SAVE_PATH = None
writer = None
epochs = 50
lr = 0.1
job = None
device_str = "cpu"

def help():
    help_str = f"""
Usage: python mnistModelTool.py <mode> [options] [<model>]

Mode:
  -A\t\t\t(Accuracy) Evaluate <model> accuracy.
  -T\t\t\t(Training) Generate and train a new MNIST Model.
  
Training options:
  -e <epochs>\t\tSet the number of epochs (default 50).
  -l <learning rate>\tSet the learning rate (default 0.1).
  -s <file>\t\tSave the model in <file>.
  -w <file>\t\tStore tensorboard data into <file>. 
  
General options:
  -h, --help\t\tDisplay this information. 
  --gpu\t\t\tPerform operations on GPU if available. 
"""
    print(help_str)
    quit()

def parsing():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "TAhs:w:e:l:", ["help", "gpu"])
        for opt, arg in opts:

            if opt in ["-h", "--help"]:
                help()

            elif opt in ['-w']:
                global writer
                writer = SummaryWriter(arg)

            elif opt in ['-e']:
                global epochs
                epochs = int(arg)
                if epochs <= 0:
                    print("Error: epochs must be a positive number")
                    quit()

            elif opt in ['-s']:
                global SAVE_PATH
                SAVE_PATH = arg

            elif opt in ["-l"]:
                global lr
                lr = float(arg)
                if lr <= 0:
                    print("Error: learning rate must be a positive number")
                    quit()

            elif opt in ["-T"]:
                global job
                job = training

            elif opt in ["-A"]:
                if len(args) > 0:
                    global LOAD_PATH
                    LOAD_PATH = args[0]
                else:
                    print("Specify a model to load.")
                    quit()
                job = accuracy

            elif opt in ["--gpu"]:
                global device_str
                device_str = "cuda"

    except Exception as e:
        help()


def training():
    
    device = torch.device(device_str)
    model = Model.MnistNetwork(device=device)

    # Caricamento dataset
    training_data = datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST("./data", train=False, transform=transforms.ToTensor(), download=True)

    training_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100, shuffle=False)

    # Training
    print("Model training...")
    try:
        model.trainModel(training_loader, test_loader, epochs, lr, writer, SAVE_PATH)
    except KeyboardInterrupt:
        print("Model training interrupted.")

def accuracy():
    device = torch.device(device_str)
    model = Model.MnistNetwork(device=device)

    # Caricamento dataset
    test_data = datasets.MNIST("./data", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100, shuffle=False)
    
    # Load Model
    try:
        util.load_model(model, LOAD_PATH)
    except:
        print(f"Error: {LOAD_PATH} is not a valid model")
        quit()
    acc = model.getAccuracy(test_loader, device)
    print(f"{LOAD_PATH} accuracy = {acc}%")


def main():
    parsing()
    if job is None:
        help()
    job()

if __name__ == "__main__":
    main()