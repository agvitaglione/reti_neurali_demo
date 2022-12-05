import os, sys
sys.path.append(f"{os.path.dirname(__file__)}/../../../utils")
import pandas as pd
import numpy as np
import binarizedModel as BModel
import util
import torch

number_of_levels = 5

# Files
mnist_model_path = f"{os.path.dirname(__file__)}/../../models/mnist_binarized_model"
csv_file = f"{os.path.dirname(__file__)}/weight_distribution.csv"

# CSV List
weight_per_level = [[] for i in range(number_of_levels)]


# Get model
model = BModel.MNISTBinarizedModel()
util.load_model(model, mnist_model_path)

# Get weight parameter
for i, g in enumerate(model.parameters()):
    weight_per_level[i].extend(g.reshape(-1).abs().tolist())
    if i == 4:
        break

for list in weight_per_level:
    list.extend([None for i in range(784*256 - len(list))])

weight_per_level = np.array(weight_per_level)
weight_per_level = np.transpose(weight_per_level)

# Pandas dataframe
df = pd.DataFrame(weight_per_level, columns=[f"layer_{i+1}" for i in range(number_of_levels)])
df.to_csv(csv_file, index=False)