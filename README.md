This repositoy allows you to generate a neural network model, a binarized model and a pruned model, with a specify threshold value, for MNIST dataset classification. Futhermore, given a model, it is possible to generate a VHDL implementation. 

## Dependencies
Install the following dependencies:
````
pip install numpy
pip install torch
pip install tensorboard
pip install torchvision
````

## Demo

### Generate models
Generate a MNIST model:
````
mkdir models
python mnistModelTool.py -T -s ./models/mnist_model
````
Evaluate *mnist_model accuracy*:
````
python mnistModelTool.py -A ./models/mnist_model
````
Binarize *mnist_model*:
````
python mnistModelBinarizedTool.py -T -s ./models/mnist_binarized_model ./models/mnist_model
````
Evaluate *mnist_binarized_model* accuracy:
````
python mnistModelBinarizedTool.py -A ./models/mnist_binarized_model
````
Prune *mnist_binarized_model* with threasheld set to 4 and retrain to gain accuracy:
````
python mnistModelBinarizedTool.py -P -t 4 -s ./models/mnist_binarized_pruned_model ./models/mnist_binarized_model
````
Evaluate *mnist_binarized_pruned_model* accuracy
````
python mnistModelBinarizedTool.py -A -p ./models/mnist_binarized_pruned_model 
````

### Generate VHDL code
Get model parameters:
````
cd vhdl_rete_mnist
python generate_vhdl_parameters.py ../models/mnist_binarized_pruned_model 
````
Generate vhdl code:
````
python generate_architecture.py
````

## Tools information
You can custome training phase, by setting epochs and learning rate, and visualize machine learning workflow with tensorboard. Moreover, the operations can be performed on GPU. 
For further details about scripts' flag, use *--help*.

