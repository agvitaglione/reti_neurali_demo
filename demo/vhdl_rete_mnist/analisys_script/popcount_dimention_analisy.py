import os
import torch
import pandas as pd

number_of_levels = 5

weight_files = [f"{os.path.dirname(__file__)}/../model_parameters/weight_{i+1}" for i in range(number_of_levels)]
csv_file = f"{os.path.dirname(__file__)}/popcount_dimention_8443.csv"
weight_list = [[] for i in range(256)]

count = 0

for file in weight_files:
    with open(file, 'r') as f:
        lines = f.readlines()
        for perceprton, line in enumerate(lines):
            line = line.replace('\n', '')
            perceptron_in = torch.tensor(list(map(float, line.split(','))))
            perceptron_in = torch.abs(perceptron_in)
            perceptron_number_of_input = int(torch.sum(perceptron_in).item())
            weight_list[perceprton].append(perceptron_number_of_input)
        
# Pandas dataframe
df = pd.DataFrame(weight_list, columns=[f"layer_{i+1}" for i in range(number_of_levels)])
df.to_csv(csv_file, index=False)