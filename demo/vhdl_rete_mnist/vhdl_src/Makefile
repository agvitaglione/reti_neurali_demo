unisim_dir = D:/Programmi/Xilinx/Vivado/2022.2/data/vhdl/src/unisims
GHDL_OPT = --std=08 -frelaxed -fsynopsys

ALL:
	ghdl --clean
	
	ghdl -a --std=08 batchNorm.vhd
	
	
	ghdl -a $(GHDL_OPT) --work=unisim $(unisim_dir)/unisim_VCOMP.vhd
	ghdl -a $(GHDL_OPT) --work=unisim $(unisim_dir)/unisim_VPKG.vhd
	ghdl -a $(GHDL_OPT) --work=unisim $(unisim_dir)/primitive/LDCE.vhd 
	ghdl -a $(GHDL_OPT) --work=unisim $(unisim_dir)/primitive/RAM64X1D.vhd
	ghdl -a $(GHDL_OPT) $(unisim_dir)/primitive/LUT6.vhd
	ghdl -a $(GHDL_OPT) $(unisim_dir)/primitive/LUT5.vhd
	ghdl -a $(GHDL_OPT) $(unisim_dir)/primitive/LUT4.vhd
	ghdl -a $(GHDL_OPT) $(unisim_dir)/primitive/LUT3.vhd
	ghdl -a $(GHDL_OPT) $(unisim_dir)/primitive/LUT2.vhd
	ghdl -a $(GHDL_OPT) $(unisim_dir)/primitive/LUT1.vhd
	
	ghdl -a --std=08 -frelaxed perceptron.vhd
	ghdl -a --std=08 layer.vhd 
	ghdl -a --std=08 popcount.vhd
	ghdl -a --std=08 popcountTile.vhd
	ghdl -a --std=08 rete_mnist.vhd
	ghdl -a --std=08 -frelaxed rete_mnist_tb.vhd
	ghdl -e --std=08 -frelaxed rete_mnist_tb
	ghdl -r --std=08 -frelaxed rete_mnist_tb