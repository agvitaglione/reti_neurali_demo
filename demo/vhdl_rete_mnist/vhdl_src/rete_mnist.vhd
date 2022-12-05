----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 05.11.2022 12:55:26
-- Design Name: 
-- Module Name: rete_mnist - Behavioral
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
use IEEE.math_real.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity rete_mnist is
    Port (
        x : in STD_LOGIC_VECTOR(0 to 783);
        y : out STD_LOGIC_VECTOR(0 to 89)
    );
end rete_mnist;

architecture Behavioral of rete_mnist is

    COMPONENT layer
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
    );
    END COMPONENT;

    signal l1_out : STD_LOGIC_VECTOR(0 to 255);
    signal l2_out : STD_LOGIC_VECTOR(0 to 255);
    signal l3_out : STD_LOGIC_VECTOR(0 to 255);
    signal l4_out : STD_LOGIC_VECTOR(0 to 255);

begin

    l1 : entity work.layer(layer_1)
        Port map ( 
            x => x,
            y => l1_out
        );
        
    l2 : entity work.layer(layer_2)
        Port map ( 
            x => l1_out,
            y => l2_out
        );
     
     l3 : entity work.layer(layer_3)
        Port map ( 
            x => l2_out,
            y => l3_out
        );
    
    l4 : entity work.layer(layer_4)
        Port map ( 
            x => l3_out,
            y => l4_out
        );
        
    l5 : entity work.layer(layer_5)
        Port map ( 
            x => l4_out,
            y => y
        );
		
	-- PARSING
	-- Prediction 0: y(0 to 8)
	-- Prediction 1: y(9 to 17)
	-- Prediction 2: y(18 to 26)
	-- Prediction 3: y(27 to 35)
	-- Prediction 4: y(36 to 44)
	-- Prediction 5: y(45 to 53)
	-- Prediction 6: y(54 to 62)
	-- Prediction 7: y(63 to 71)
	-- Prediction 8: y(72 to 80)
	-- Prediction 9: y(81 to 89)
	
   
end Behavioral;
