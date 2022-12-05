----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 09.11.2022 10:05:04
-- Design Name: 
-- Module Name: rete_mnist_tb - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
use IEEE.NUMERIC_STD.ALL;
use std.textio.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity rete_mnist_tb is
--	Generic(
--		IMAGE : STD_LOGIC_VECTOR
--	);
--  Port ( );
end rete_mnist_tb;

architecture Behavioral of rete_mnist_tb is

    Component rete_mnist 
    Port (
        x : in STD_LOGIC_VECTOR(0 to 783);
        y : out STD_LOGIC_VECTOR(0 to %OUT_UB%)  
    );
    end component;
    
    -- Inputs
    signal x : STD_LOGIC_VECTOR(0 to 783) := (others => '0');
    
    -- Outputs
    signal y : STD_LOGIC_VECTOR(0 to %OUT_UB%) := (others => '0');
    
   -- Clock period definitions
   constant clk_period : time := 10 ns;

begin

    -- Istanza del componente da testare
    rm : rete_mnist
        Port map (
            x => x,
            y => y
        );

    -- Process di simulazione
	 sim_proc : process
		variable input_line : line;
		variable output_line : line; 
		variable image : STD_LOGIC_VECTOR(0 to 783);
		variable count : Integer := 0;
		file input_file : text open read_mode is "images"; 
		file output_file : text open write_mode is "vhdl_prediction.txt";
		begin
			
			while not endfile(input_file) loop
				
				wait for clk_period;
				readline(input_file, input_line);
				read(input_line, image);
				x <= image;
				wait for clk_period;
				hwrite(output_line, y);
				writeline(output_file, output_line);
				-- x -> 1
				-- x <= "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000100000000000000000000000000110000000000000000000000000011000000000000000000000000001000000000000000000000000001100000000000000000000000000110000000000000000000000000011000000000000000000000000011000000000000000000000000001100000000000000000000000000110000000000000000000000000110000000000000000000000000011000000000000000000000000001100000000000000000000000001110000000000000000000000000110000000000000000000000000011000000000000000000000000011100000000000000000000000001100000000000000000000000000110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
				-- x -> 9
				-- x <= IMAGE;
				-- wait for clk_period;
				-- hwrite(l, y);
				-- writeline(output, l);
				-- x <= "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111000000000000000000000011111111110000000000000000111111111111000000000000001111110001011110000000000000111100000100111000000000000111000000001111100000000000011101111111111100000000000001111111111111100000000000000001111111111100000000000000000000001111000000000000000000000000111000000000000000000000000111100000000000000000000000111100000000000000000000000111100000000000000000000000011110000000000000000000000011110000000000000000000000001110000000000000000000000001111000000000000000000000000111000000000000000000000000001000000000000000000000000000000000000000000000";       
				
				count := count + 1;
				if (count mod 100) = 0 then
					write(output_line, count);
					writeline(output, output_line);
				end if;
			
			end loop;
			wait;
    end process;
    
end Behavioral;
