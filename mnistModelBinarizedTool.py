import sys, getopt, os
sys.path.append(f"{os.path.dirname(__file__)}/utils/")
import utils.mnistModel as Model
import utils.binarizedModel as BModel
import utils.util as util
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils.prune as prune

# Variabili globali
LOAD_PATH = None
SAVE_PATH = None
writer = None
epochs = 50
lr = 0.01
job = None
device_str = "cpu"
threshold = 0
pruned = False


def help():
    help_str = """
Usage: python mnistModelBinarizedTool.py <mode> [options] <model>

Mode:
  -A\t\t\t(Accuracy) Evaluate <model> accuracy.
  -P\t\t\t(Pruning)  Prune the binarized model <model> with specified <threshold>.
  -T\t\t\t(Training) Generate and train a new MNIST Binarized Model from <model>.

Accuracy options:
  -t <threshold>\tSimulate accuracy with a specified pruning threshold value.
  -p\t\t\tEvaluate accuracy on a pruned model. Use this rather than -t if the model is already pruned.

Pruning options:
  -t <threshold>\tSet threshold for pruning (default 0).
  
Training options:
  -e <epochs>\t\tSet the number of epochs (default 50).
  -l <learning rate>\tSet the learning rate (default 0.01).
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
        opts, args = getopt.getopt(sys.argv[1:], "TAPhs:w:e:l:t:p", ["help", "gpu"])

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

            elif opt in ["-l"]:
                global lr
                lr = float(arg)
                if lr <= 0:
                    print("Error: learning rate must be a positive number")
                    quit()

            elif opt in ['-p']:
                global pruned
                pruned = True

            elif opt in ['-s']:
                global SAVE_PATH
                SAVE_PATH = arg

            elif opt in ['-t']:
                global threshold
                
                threshold = float(arg)
                if(threshold > 0):
                    pruned = True

            elif opt in ["-T"]:
                global job
                job = training

                if len(args) > 0:
                    global LOAD_PATH
                    LOAD_PATH = args[0]
                else:
                    print("Specify a model to load.")
                    quit()

            elif opt in ["-A"]:
                job = accuracy

                if len(args) > 0:
                    LOAD_PATH = args[0]
                else:
                    print("Specify a binarized model to load.")
                    quit()

            elif opt in ["-P"]:
                job = pruning

                if len(args) > 0:
                    LOAD_PATH = args[0]
                else:
                    print("Specify a binarized model to load.")
                    quit()
            

            elif opt in ["--gpu"]:
                global device_str
                device_str = "cuda"

    except Exception as e:
        # print(e)
        help()

def training():

    # Dataset
    training_data = datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST("./data", train=False, transform=transforms.ToTensor(), download=True)

    training_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100, shuffle=False)

    # Caricamento modello non binarizzato
    device = torch.device(device_str)
    model = Model.MnistNetwork(device=device)
    try:
        util.load_model(model, LOAD_PATH, device)
    except:
        print(f"Error: {LOAD_PATH} is not a valid model")
        quit()

    # Definizione modello binarizzato
    bmodel = BModel.MNISTBinarizedModel(model, device)
    try:
        print("Trining model...")
        bmodel.trainModel(training_loader, test_loader, epochs, lr, writer=writer, PATH=SAVE_PATH)
    except KeyboardInterrupt:
        print("Model training interrupted.")

def accuracy():

    device = torch.device(device_str)

    # Dataset
    test_data = datasets.MNIST("./data", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100, shuffle=False)

    # Definizione modello binarizzato
    bmodel = BModel.MNISTBinarizedModel(device=device)
    try:
        util.load_model(bmodel, LOAD_PATH, device)
    except Exception as e:
        print(e)
        print(f"Error: {LOAD_PATH} is not a valid model.")
        quit()

    acc = bmodel.getAccuracy(test_loader, pruned, threshold)
    print(f"{LOAD_PATH} accuracy = {acc}%")

def pruning():

    # Dataset
    training_data = datasets.MNIST("./data", train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST("./data", train=False, transform=transforms.ToTensor(), download=True)

    training_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100, shuffle=False)

    # Caricamento modello non binarizzato
    device = torch.device(device_str)
    bmodel = BModel.MNISTBinarizedModel(device=device)
    try:
        util.load_model(bmodel, LOAD_PATH, device)
    except:
        print(f"Error: {LOAD_PATH} is not a valid model")
        quit()

    # Training
    try:
        print("Trining model...")
        bmodel.prune(threshold, training_loader, test_loader, epochs, lr, writer, SAVE_PATH)
    except KeyboardInterrupt:
        print(f"Training interrupted. Saving the best model in {SAVE_PATH}.")

    # Pruning definitivo
    util.load_model(bmodel, SAVE_PATH, device)  # Caricamento modello migliore pruned

    parameters_to_prune = [(bmodel.fcLayers[i], "weight") for i in range(len(bmodel.fcLayers))]
    prune.global_unstructured(
        parameters_to_prune, pruning_method=util.ThresholdPruning, threshold=threshold
    )

    for child in bmodel.fcLayers:
        prune.remove(child, "weight")

    # Salvataggio modello pruned
    util.save_model(bmodel, SAVE_PATH)

    # Calcolo della percentuale dei pesi pruned
    tot = 0
    size = 0
    for g in bmodel.parameters():
        if(len(g.shape) > 1):
            size += g.shape[0] * g.shape[1]
            r = torch.where(torch.eq(g, 0), 1., 0.)
            tot += torch.sum(r).item()

    print("Percentage of pruned weights = " + "{:.3f}".format(tot / size * 100) + "%")


def main():
    parsing()
    if job is None:
        help()
    job()

if __name__ == "__main__":
    main()
