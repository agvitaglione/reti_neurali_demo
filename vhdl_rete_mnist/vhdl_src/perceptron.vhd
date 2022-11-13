----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 05.11.2022 12:07:05
-- Design Name: 
-- Module Name: perceptron - Behavioral
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

entity perceptron is
    Generic (
        WEIGHTS : STD_LOGIC_VECTOR;
        SHIFT : Integer;     -- number of shift (left + or right -)
        B : Integer;          -- bias of batch norm
        BATCH_POSITIVE : Boolean := True  
    );
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
   );
end perceptron;

architecture Behavioral of perceptron is

    -- popcount del percettrone
    COMPONENT popcount
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
    );  
    END COMPONENT;
    
    COMPONENT batchNorm
    Generic (
        SHIFT : Integer;     -- number of shift (left + or right -)
        B : Integer          -- bias of batch norm   
    );
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
    );
    END COMPONENT;
    
   -- Signals
   signal xor_result : STD_LOGIC_VECTOR(x'range) := (others => '0');
   signal y_popcount: STD_LOGIC_VECTOR(0 to integer(ceil(log2(real(2*x'length + 1)))) - 1);
   
begin
    
    -- Fully connected phase
    xor_result <= WEIGHTS xnor x;
    
    pop : popcount 
    Port map(
        x => xor_result,
        y => y_popcount
    );
    
    ---------------------------
    
    -- Batchnorm phase
    bnpg : if (BATCH_POSITIVE = True) generate
        bn : entity work.batchNorm(positiveBatchNorm)
        Generic map (
            SHIFT => SHIFT,
            B => B
        )
        Port map ( 
            x => y_popcount,
            y => y
        );
    end generate; 
    
    bnng : if (BATCH_POSITIVE = False) generate
        bn : entity work.batchNorm(negativeBatchNorm)
        Generic map (
            SHIFT => SHIFT,
            B => B
        )
        Port map ( 
            x => y_popcount,
            y => y
        );
    end generate;
    --------------------
   
end Behavioral;
