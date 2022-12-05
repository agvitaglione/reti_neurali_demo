import sys, os
import math
sys.path.append(f"{os.path.dirname(__file__)}/../utils")
sys.path.append(f"{os.path.dirname(__file__)}/../models")

MODEL_PARAMETERS = f"{os.path.dirname(__file__)}/model_parameters"


# utils function 
# Genera la tabella di verità template dato il numero di imput
def getTable(number_of_input):
    rows = [i for i in range(2 ** number_of_input)]
    table = []
    for i in range(len(rows)):
        table_row = ("{0:0" + str(number_of_input) + "b}").format(rows[i])
        table.append(table_row)
    return table


def int2bit(x, num_of_bits):
    val = x if x >= 0 else (1 << num_of_bits) + x
    val = int(val)
    bit_str =  ("{0:0" + str(num_of_bits) + "b}").format(val)
    return bit_str

# -----------------------------------------------------------------------

# Lettura template
with open(f"{os.path.dirname(__file__)}/template/layer_template.vhd", 'r') as f:
    template = f.read()

# Template percettrone
# bit di segno deve essere negato (not) poiché 1 bit di segno corrisponde a -1 che nella nostra logica è indicano col bit 0
perceptron_template = """
    p%INDEX%_in%SINGLE_INPUT% <= %x%;
    y(%INDEX%) <= not(p%INDEX%_out);
    p%INDEX% : entity work.perceptron(LUT6)
        Generic map (
            WEIGHTS => "%WEIGHTS%",
            TABLE => "%TABLE%",
            SHIFT => %SHIFT%,
            B => %B%,
            BATCH_POSITIVE => %BATCH_POSITIVE%
        )
        Port map (
            x => p%INDEX%_in,
            y(0) => p%INDEX%_out,
            y(1 to %YLENGTH% - 1) => dummy%INDEX%
        );"""
perceptron_last_level_template = """
    p%INDEX%_in%SINGLE_INPUT% <= %x%;
    p%INDEX% : entity work.perceptron(LUT6)
        Generic map (
            WEIGHTS => "",
            TABLE => "%TABLE%",
            SHIFT => %SHIFT%,
            B => %B%,
            BATCH_POSITIVE => %BATCH_POSITIVE%
        )
        Port map (
            x => p%INDEX%_in,
            y => y(%LB% to %UB%)
        );"""
signal_template = """signal p%INDEX%_in : STD_LOGIC_VECTOR(0 to %SIGNAL_LENGTH% - 1);\n\tsignal p%INDEX%_out : STD_LOGIC;\n\t"""
dummy_template = """signal dummy%INDEX% : STD_LOGIC_VECTOR(0 to %YLENGTH% - 2);\n\t"""

# Util variable
ub_array = []
bound_str = ""      # Stringa che contiene i bound per il parsing dell'uscita della rete

# Per ogni livello
for k in range(5):

    body = ""
    signals = ""
    dummies = ""

    # Lettura pesi
    with open(f"{MODEL_PARAMETERS}/weight_{k+1}", 'r') as f:
        weight_lines = f.readlines()

    # Lettura a
    with open(f"{MODEL_PARAMETERS}/a_{k+1}", 'r') as f:
        a_array = f.read().split(',')

    # Lettura b
    with open(f"{MODEL_PARAMETERS}/b_{k+1}", 'r') as f:
        b_array = f.read().split(',')
    

    # upper bound e lower bound per i percettroni dell'ultimo livello
    lb = 0
    ub =-1
    # ---------------------------

    # Per ogni percettrone del livello
    for p_index, weight_line in enumerate(weight_lines):

        perceptron = perceptron_template if k < 4 else perceptron_last_level_template      # template diverso per i percettroni dell'ultimo livello

        # Lettura dei pesi
        weight_array = weight_line.replace("\n", "").split(',')
        
        # AGGIUNTA PESI
        in_index = []                           # Indice degli ingressi associati ai pesi. Gli 0 sono potati
        weight_str = ""                         # Stringa dei pesi, dove -1 -> 0, 1 -> 1
        for i, w in enumerate(weight_array):
            if w == '-1' or w == '1':
                if w == '-1':
                    weight_str += '0'
                elif w == '1':
                    weight_str += '1'  
                in_index.append(i)

        # Se il percettrone non ha ingressi, deve essere prunato
        if len(in_index) == 0:
            continue
        
        x_str = [f"x({i})" for i in in_index]
        x_str = ' & '.join(x_str)
        if len(weight_str) == 1:
            perceptron = perceptron.replace("%SINGLE_INPUT%", "(0)")
        elif len(weight_str) > 1:
            perceptron = perceptron.replace("%SINGLE_INPUT%", "")

        perceptron = perceptron.replace("%INDEX%", str(p_index))
        perceptron = perceptron.replace("%WEIGHTS%", weight_str)
        perceptron = perceptron.replace("%x%", x_str)

        # Generazione tabella
        num_6_luts = int(len(weight_str) / 6)
        num_rest_inputs = len(weight_str) % 6

        xnor_pop_count = []
        if num_6_luts != 0:

            table = getTable(6)

            for i in range(num_6_luts):
                # Lettura stringa pesi
                weight_list = weight_str[6 * i : 6 * (i + 1)]
                weight_list = [x for x in weight_list]

                for row in table:
                    xor = [int(weight_list[j]) ^ int(row[j]) for j in range(len(row))]
                    xnor = [abs(x - 1) for x in xor] # flip
                    xnor = [2 * x - 1 for x in xnor]
                    xnor_pop = sum(xnor)
                    # Rappresentazione complementi a due
                    xnor_pop = int2bit(xnor_pop / 2, 3)
                    xnor_pop_count.append(xnor_pop)


        if num_rest_inputs != 0:

            table = getTable(num_rest_inputs)
            weight_list = weight_str[-num_rest_inputs:]
            weight_list = [x for x in weight_list]
            for row in table:
                    xor = [int(weight_list[j]) ^ int(row[j]) for j in range(len(row))]
                    xnor = [abs(x - 1) for x in xor] # flip
                    xnor = [2 * x - 1 for x in xnor]
                    xnor_pop = sum(xnor)
                    # Rappresentazione complementi a due

                    if num_rest_inputs == 1:
                        num_of_bits = 2

                    elif num_rest_inputs == 2 or num_rest_inputs == 3:
                        num_of_bits = 3

                    elif num_rest_inputs == 4 or num_rest_inputs == 5:
                        num_of_bits = 4 

                    xnor_pop = int2bit(xnor_pop, num_of_bits)
                    xnor_pop_count.append(xnor_pop)

        # Costruzione degli INIT del percettrone

        # LUT6
        init = ""
        for i in range(num_6_luts):
            o2 = ""
            o1 = ""
            o0 = ""
            for j in range(64):
                o2 = xnor_pop_count[64 * i + j][0] + o2
                o1 = xnor_pop_count[64 * i + j][1] + o1
                o0 = xnor_pop_count[64 * i + j][2] + o0
            init = init + o2 + o1 + o0 

        # Ultima LUT
        if num_rest_inputs == 1:

            o1 = xnor_pop_count[-1][0] + xnor_pop_count[-2][0]
            o0 = xnor_pop_count[-1][1] + xnor_pop_count[-2][1]
            init = init + o1 + o0
            
        elif num_rest_inputs == 2 or num_rest_inputs == 3:

            o2 = ""
            o1 = ""
            o0 = ""

            for i in range(64 * num_6_luts, len(xnor_pop_count)):

                o2 = xnor_pop_count[i][0] + o2
                o1 = xnor_pop_count[i][1] + o1
                o0 = xnor_pop_count[i][2] + o0

            init = init + o2 + o1 + o0

        elif num_rest_inputs == 4 or num_rest_inputs == 5:
            
            o3 = ""
            o2 = ""
            o1 = ""
            o0 = ""

            for i in range(64 * num_6_luts, len(xnor_pop_count)):
                
                o3 = xnor_pop_count[i][0] + o3
                o2 = xnor_pop_count[i][1] + o2
                o1 = xnor_pop_count[i][2] + o1
                o0 = xnor_pop_count[i][3] + o0

            init = init + o3 + o2 + o1 + o0

        perceptron = perceptron.replace("%TABLE%", init)

        # if p_index == 1:
        #     with open("./log_tabella_verita", 'w+') as f_t:
        #         f_t.write(weight_str[6:12] + "\n\n")
        #         table = getTable(6)
        #         init0 = init[64 * 3 : 64 * 4]
        #         init1 = init[64 * 4: 64 * 5]
        #         init2 = init[64 * 5 : 64 * 6]
        #         count = 0
        #         for k in reversed(range(len(table))):
        #             f_t.write("".join(table[count]) + "\t" + str(init0[k]) + str(init1[k]) + str(init2[k]) + "  " + xnor_pop_count[count + 64] + '\n')
        #             count += 1
        #     quit()

        # ------------------------ Costruzione degli INIT del percettrone

        # Shift
        a = float(a_array[p_index])
        shift = int(math.log2(abs(a)))
        perceptron = perceptron.replace("%SHIFT%", str(shift))
        perceptron = perceptron.replace("%BATCH_POSITIVE%", str(a >= 0))

        # B
        b = int(float(b_array[p_index]))
        perceptron = perceptron.replace("%B%", str(b))

        # Signal concatenazioni
        signal_length = len(weight_str)
        signals += signal_template.replace("%INDEX%", str(p_index)).replace("%SIGNAL_LENGTH%", str(signal_length))

         # OUTPUT LENGTH
        sum_length = math.ceil(math.log2(2*len(x_str)+1)) + shift
        b_length = 0 if b == 0 else math.ceil(math.log2(abs(b))) + 1
        y_length = max(sum_length, b_length) + 1

        # Modifica template non ultimo livello
        if k < 4:           
            perceptron = perceptron.replace("%YLENGTH%", str(y_length))
            # DUMMY
            dummies += dummy_template.replace("%INDEX%", str(p_index)).replace("%YLENGTH%", str(y_length))
        
        # Modifica template ultimo livello
        else:

            lb = ub + 1
            ub = ub + y_length
            perceptron = perceptron.replace("%LB%", str(lb))
            perceptron = perceptron.replace("%UB%", str(ub))
            ub_array.append(ub)

            bound_str += f"{lb},{ub}\n"

        body += perceptron + "\n\n"

    # Scrittura body per ogni livello
    template = template.replace(f"%BODY_{k+1}%", body).replace(f"%SIGNALS_{k+1}%", signals).replace(f"%DUMMIES_{k+1}%", dummies)

# Scrittura template
with open(f"{os.path.dirname(__file__)}/vhdl_src/layer.vhd", 'w') as f:
    f.write(template)


# Generazione template rete mnist
with open(f"{os.path.dirname(__file__)}/template/rete_mnist_template.vhd", 'r') as f:
    rete_mnist_template = f.read()

rete_mnist_template = rete_mnist_template.replace("%OUTPUT_UB%", str(ub_array[-1]))

parsing = ""
lb = 0
for i, ub in enumerate(ub_array):
    parsing += f"-- Prediction {i}: y({lb} to {ub})\n\t"
    lb = ub + 1

rete_mnist_template = rete_mnist_template.replace("%PARSING%", parsing)

with open(f"{os.path.dirname(__file__)}/vhdl_src/rete_mnist.vhd", 'w') as f:
    f.write(rete_mnist_template)

# Scrittura file bound del parsing
with open(f"{os.path.dirname(__file__)}/test_vhdl_architecture/output_bound", 'w') as f:
    f.write(bound_str)


# Elaborazione del template di rete_mnist_tb
with open(f"{os.path.dirname(__file__)}/template/rete_mnist_tb_template.vhd", 'r') as f: 
    rete_mnist_tb_template = f.read()

rete_mnist_tb_template = rete_mnist_tb_template.replace("%OUT_UB%", str(ub_array[-1]))

with open(f"{os.path.dirname(__file__)}/vhdl_src/rete_mnist_tb.vhd", 'w') as f:
    f.write(rete_mnist_tb_template)