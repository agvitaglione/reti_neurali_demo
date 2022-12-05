import sys, os
sys.path.append(f"{os.path.dirname(__file__)}/../../utils")
sys.path.append(f"{os.path.dirname(__file__)}/../../models")
import util
import mnistModel as Model
import torch
import binarizedModel
import getopt

LOAD_PATH = None
MODEL_PARAMETERS = f"{os.path.dirname(__file__)}/model_parameters"

def help():
    help_str = f"""
Usage: python generate_vhdl_parameters.py [options] <model>

<model> is a MNIST binarized model.
The parameters are stored in {MODEL_PARAMETERS}

Options:
  -h, --help\t\tDisplay this information. 
  --gpu\t\t\tPerform operations on GPU if available. 
"""
    print(help_str)
    quit()

def parsing():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "gpu"])

        if len(args) > 0:
            global LOAD_PATH
            LOAD_PATH = args[0]

        for opt, arg in opts:

            if opt in ["-h", "--help"]:
                help()

            elif opt in ["--gpu"]:
                global device_str
                device_str = "cuda"
        
        if LOAD_PATH is None:
            raise Exception

    except Exception as e:
        help()

def main():

    # IMPORT MODELLO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bmodel = binarizedModel.MNISTBinarizedModel(device=device)
    util.load_model(bmodel, LOAD_PATH)

    # PESI
    for k, g in enumerate(bmodel.fcLayers):
        r = torch.where(torch.gt(g.weight, 0), 1, g.weight)
        r = torch.where(torch.lt(r, 0), -1, r)
    
        level = ""
        for i in range(r.shape[0]):
            line = ""
            for j in range(r.shape[1]):
                line += str(int(r[i][j].item())) + ','
            line = line[:-1]            # Rimozione ultima virgola
            level += line + "\n"

        with open(f"{MODEL_PARAMETERS}/weight_{k+1}", 'w') as f:
            f.write(level)
        
    # BN
    for k, g in enumerate(bmodel.bnLayers):
        a, b = g.getParameter()

        level = ""
        for elem in a:
            level += str(elem.item()) + ','
        level = level[:-1]
        
        with open(f"{MODEL_PARAMETERS}/a_{k+1}", 'w') as f:
            f.write(level)

        level = ""
        for elem in b:
            level += str(elem.item()) + ','
        level = level[:-1]

        with open(f"{MODEL_PARAMETERS}/b_{k+1}", 'w') as f:
            f.write(level)

if __name__ == "__main__":

    parsing()

    # Preparazione ambiente
    try: 
        os.mkdir(MODEL_PARAMETERS)
        print(f"Directory {MODEL_PARAMETERS} created")
    except:
        pass

    main()