import sys, os
sys.path.append(f"{os.path.dirname(__file__)}/../../../utils/")
import torch
from torchvision import datasets
from torchvision import transforms
import util
import binarize as Bin
import binarizedModel as BModel


# Files
python_prediction_file = f"{os.path.dirname(__file__)}/python_prediction"
model_path = f"{os.path.dirname(__file__)}/../../../models/MnistNetworkPruned_binarized_model"
test_data_directory = f"{os.path.dirname(__file__)}/data"

# Data una predizione vhdl in esadecimale, restituisce una lista di predizioni in formato int
def getOutputList(x):
    bin_pred = util.hex2bin(x, 90)
    output_network = [bin_pred[0:9], bin_pred[9:18], bin_pred[18:27], bin_pred[27:36], bin_pred[36:45], bin_pred[45:54], bin_pred[54:63], bin_pred[63:72], bin_pred[72:82], bin_pred[82:91]]
    output_network = list(map(util.bin2int, output_network))
    return output_network

def main():

    # Caricamento dati
    test_data = datasets.MNIST(test_data_directory, train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100, shuffle=False)

    # Model
    model = BModel.MNISTBinarizedModel(device=torch.device("cpu"))
    model.softMax = torch.nn.Identity() # Disabilitaazione softmax dal modello

    util.load_model(model, model_path)

    # Generate file
    print(f"Generating python prediction in {python_prediction_file}...")
    with open(python_prediction_file, 'w+') as f:

        for images, labels in test_loader:

            images = images.reshape(len(images), -1)
            outputs, predictions = model.prediction(images, pruned=True)
            
            # Scrittura immagini su file
            for out in outputs:
                line = str(out.int().tolist()).replace(' ', '').replace('[', '').replace(']', '')
                f.write(line + '\n')
    print("Operation completed!")

if __name__ == "__main__":
    main()