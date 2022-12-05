import sys, os
sys.path.append(f"{os.path.dirname(__file__)}/../../../utils/")
import torch
from torchvision import datasets
from torchvision import transforms
import util
import binarize as Bin
import binarizedModel as BModel

# test_data = datasets.MNIST(f"{os.path.dirname(__file__)}/data", train=False, transform=transforms.ToTensor(), download=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100, shuffle=False)

# Files
images_file = f"{os.path.dirname(__file__)}/images"
labels_file = f"{os.path.dirname(__file__)}/labels"
output_bound_file = f"{os.path.dirname(__file__)}/output_bound"
architecture_predictions_file = f"{os.path.dirname(__file__)}/../vhdl_src/vhdl_prediction.txt"

# Ottenimento bound per il parsing
# bound_list contiene le informazioni per il parsing dell'output del modello
# Ogni riga contiene il range di bit della predizione
bound_list = []
try:
    with open(output_bound_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            bound = list(map(int,line.split(',')))
            bound_list.append((bound[0], bound[1] + 1))
except:
    print(f"Error: bound list file not found or incorrect format! {output_bound_file}")
    quit()

# Get prediction from vhdl tb
def architecturePrediction():
    command = f'ghdl -r --std=08 rete_mnist_tb'
    os.system(command)
# ------------------------------------------

def getPrediction(x):
    num_of_bits = bound_list[-1][1]
    bin_pred = util.hex2bin(x, num_of_bits)
    output_network = [bin_pred[bound_list[i][0]:bound_list[i][1]] for i in range(len(bound_list))]
    output_network = list(map(util.bin2int, output_network))
    prediction = torch.argmax(torch.tensor(output_network)).item()
    return prediction


# Generate file
def main():
    # print("Generating images and labels file...")
    # with open(images_file, 'w+') as f_image, open(labels_file, 'w+') as f_labels:

        # for images, labels in test_loader:

            # images = images.reshape(len(images), -1)

            # # Normalizzazione immagini e binarizzazione
            # images = images - 0.5
            # images = Bin.binarize01(images).int()
            
            # # Scrittura immagini su file
            # for img in images:
                # img_str = str(img.tolist()).replace('[', '').replace(']', '').replace(', ', '')
                # f_image.write(img_str + "\n")
                
            # for label in labels.tolist():
                # f_labels.write(str(label) + '\n')

    # print("Operation completed!")

    # Test architecture
    print("Test architecture...")
    with open(architecture_predictions_file, 'r') as f_arch_pred, open(labels_file, 'r') as f_label:
        
        arch_predictions_lines = f_arch_pred.readlines()
        label_lines = f_label.readlines()

    tot_correct_pred = 0

    for i in range(len(label_lines)):
        arch_pred = arch_predictions_lines[i].replace('\n', '')
        pred = getPrediction(arch_pred)
        label = int(label_lines[i].replace('\n', ''))

        if pred == label:
            tot_correct_pred += 1

    accuracy = tot_correct_pred / (len(label_lines)) * 100
    print(str(accuracy) + "%")

if __name__ == "__main__":
    main()
 
