----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 05.11.2022 12:07:05
-- Design Name: 
-- Module Name:  perceptron - Behavioral
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
library UNISIM;
use UNISIM.VComponents.all;

entity perceptron is
    Generic (
        table : bit_vector;
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

architecture LUT6 of perceptron is

    Component LUT1
    Generic  (
        INIT : bit_vector
    );
    Port (
        O : out STD_LOGIC;
        I0 : in STD_LOGIC
    );
    end component;
    
    Component LUT2
    Generic  (
        INIT : bit_vector
    );
    Port (
        O : out STD_LOGIC;
        I0 : in STD_LOGIC;
        I1 : in STD_LOGIC
    );
    end component;
    
    Component LUT3
    Generic  (
        INIT : bit_vector
    );
    Port (
        O : out STD_LOGIC;
        I0 : in STD_LOGIC;
        I1 : in STD_LOGIC;
        I2 : in STD_LOGIC
    );
    end component;
    
    Component LUT4
    Generic  (
        INIT : bit_vector
    );
    Port (
        O : out STD_LOGIC;
        I0 : in STD_LOGIC;
        I1 : in STD_LOGIC;
        I2 : in STD_LOGIC;
        I3 : in STD_LOGIC
    );
    end component;
    
    Component LUT5
    Generic  (
        INIT : bit_vector
    );
    Port (
        O : out STD_LOGIC;
        I0 : in STD_LOGIC;
        I1 : in STD_LOGIC;
        I2 : in STD_LOGIC;
        I3 : in STD_LOGIC;
        I4 : in STD_LOGIC
    );
    end component;

    Component LUT6
    Generic  (
       INIT : bit_vector
      );
    Port (
       O : out STD_LOGIC;
       I0 : in STD_LOGIC;
       I1 : in STD_LOGIC;
       I2 : in STD_LOGIC;
       I3 : in STD_LOGIC;
       I4 : in STD_LOGIC;
       I5 : in STD_LOGIC
    );
    end component;
    
    Component batchNorm
    Generic (
        SHIFT : Integer;     -- number of shift (left + or right -)
        B : Integer          -- bias of batch norm   
    );
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
    );
    end component;

    constant num_6_luts : integer := integer(floor(real(x'length) / 6.0));
    constant rest_input : integer := x'length mod 6;
    signal partial_pop_count : STD_LOGIC_VECTOR(0 to 3 * num_6_luts - 1 + integer(ceil(log2(2.0 * real(rest_input) + 1.0)))) := (others => '0');  -- è calcolato il numero di output della lut resto
    signal pop_count_result : STD_LOGIC_VECTOR(0 to integer(ceil(log2(real(2*x'length + 1)))) - 1) := (others => '0');

begin

    luts6gen : for i in 0 to num_6_luts - 1 generate
        
        lu6_1 : LUT6
            Generic map (
                INIT => table(64 * (3 * i + 2) to 64 * (3 * i + 3) - 1)
            )
            -- BIT X DAL PIU' SIGNIFICATIVO
            Port map (
               O => partial_pop_count(3 * i + 2),
               I0 => x(6 * i + 5),
               I1 => x(6 * i + 4),
               I2 => x(6 * i + 3),
               I3 => x(6 * i + 2),
               I4 => x(6 * i + 1),
               I5 => x(6 * i)
            ); 
            
        lu6_2 : LUT6
            Generic map (
                INIT => table(64 * (3* i + 1) to 64 * (3 * i + 2) - 1)
            )
            -- BIT X DAL PIU' SIGNIFICATIVO
            Port map (
               O => partial_pop_count(3 * i + 1),
               I0 => x(6 * i + 5),
               I1 => x(6 * i + 4),
               I2 => x(6 * i + 3),
               I3 => x(6 * i + 2),
               I4 => x(6 * i + 1),
               I5 => x(6 * i)
            );
               
        lu6_3 : LUT6
            Generic map (
                INIT => table(64 * 3*i to 64 * (3 * i + 1) - 1)
            )
            -- BIT X DAL PIU' SIGNIFICATIVO
            Port map (
               O => partial_pop_count(3 * i),
               I0 => x(6 * i + 5),
               I1 => x(6 * i + 4),
               I2 => x(6 * i + 3),
               I3 => x(6 * i + 2),
               I4 => x(6 * i + 1),
               I5 => x(6 * i)
            );     
                
    end generate;
    
    cond1 : if rest_input = 1 generate
        lu1_1 : LUT1
            Generic map(
                INIT => table(table'right - 3 to table'right - 2)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right - 1),
                I0 => x(x'right)
            );
            
        lu1_2 : LUT1
            Generic map(
                INIT => table(table'right - 1 to table'right)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right),
                I0 => x(x'right)
            );
            
    end generate;
    
    cond2 : if rest_input = 2 generate
    
        l2_1 : LUT2
            Generic map(
                INIT => table(table'right - 11 to table'right - 8)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right - 2),
                I0 => x(x'right),
                I1 => x(x'right - 1)
            );
            
        l2_2 : LUT2
            Generic map(
                INIT => table(table'right - 7 to table'right - 4)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right - 1),
                I0 => x(x'right),
                I1 => x(x'right - 1)
            );
            
        l2_3 : LUT2
            Generic map(
                INIT => table(table'right - 3 to table'right)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right),
                I0 => x(x'right),
                I1 => x(x'right - 1)
            );  
                
    end generate;
    
    cond3 : if rest_input = 3 generate
    
        l3_1 : LUT3
            Generic map (
                INIT => table(table'right - 23 to table'right - 16)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right - 2),
                I0 => x(x'right),
                I1 => x(x'right - 1),
                I2 => x(x'right - 2)
            );
            
        l3_2 : LUT3
            Generic map (
                INIT => table(table'right - 15 to table'right - 8)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right - 1),
                I0 => x(x'right),
                I1 => x(x'right - 1),
                I2 => x(x'right - 2)
            );  
            
        l3_3 : LUT3
            Generic map (
                INIT => table(table'right - 7 to table'right)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right),
                I0 => x(x'right),
                I1 => x(x'right - 1),
                I2 => x(x'right - 2)
            );        
   
    end generate;
    
    cond4 : if rest_input = 4 generate
    
        l4_1 : LUT4
            Generic map (
                INIT => table(table'right - 63 to table'right - 48)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right - 3),
                I0 => x(x'right),
                I1 => x(x'right - 1),
                I2 => x(x'right - 2),
                I3 => x(x'right - 3)
            );
            
        l4_2 : LUT4
            Generic map (
                INIT => table(table'right - 47 to table'right - 32)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right - 2),
                I0 => x(x'right),
                I1 => x(x'right - 1),
                I2 => x(x'right - 2),
                I3 => x(x'right - 3)
            );  
            
        l4_3 : LUT4
            Generic map (
                INIT => table(table'right - 31 to table'right - 16)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right - 1),
                I0 => x(x'right),
                I1 => x(x'right - 1),
                I2 => x(x'right - 2),
                I3 => x(x'right - 3)
            );        
          
        l4_4 : LUT4
            Generic map (
                INIT => table(table'right - 15 to table'right)
            )
            Port map (
                O => partial_pop_count(partial_pop_count'right),
                I0 => x(x'right),
                I1 => x(x'right - 1),
                I2 => x(x'right - 2),
                I3 => x(x'right - 3)
            );
       
       end generate;    
       
       cond5 : if rest_input = 5 generate
       
            l5_1 : LUT5
                Generic map (
                    INIT => table(table'right - 127 to table'right - 96)
                )
                Port map (
                    O => partial_pop_count(partial_pop_count'right - 3),
                    I0 => x(x'right),
                    I1 => x(x'right - 1),
                    I2 => x(x'right - 2),
                    I3 => x(x'right - 3),
                    I4 => x(x'right - 4)
                ); 
                
            l5_2 : LUT5
                Generic map (
                    INIT => table(table'right - 95 to table'right - 64)
                )
                Port map (
                    O => partial_pop_count(partial_pop_count'right - 2),
                    I0 => x(x'right),
                    I1 => x(x'right - 1),
                    I2 => x(x'right - 2),
                    I3 => x(x'right - 3),
                    I4 => x(x'right - 4)
                );
                
            l5_3 : LUT5
                Generic map (
                    INIT => table(table'right - 63 to table'right - 32)
                )
                Port map (
                    O => partial_pop_count(partial_pop_count'right - 1),
                    I0 => x(x'right),
                    I1 => x(x'right - 1),
                    I2 => x(x'right - 2),
                    I3 => x(x'right - 3),
                    I4 => x(x'right - 4)
                );
                
            l5_4 : LUT5
                Generic map (
                    INIT => table(table'right - 31 to table'right)
                )
                Port map (
                    O => partial_pop_count(partial_pop_count'right),
                    I0 => x(x'right),
                    I1 => x(x'right - 1),
                    I2 => x(x'right - 2),
                    I3 => x(x'right - 3),
                    I4 => x(x'right - 4)
                );
                       
       end generate; 
        
        sum : process(partial_pop_count)
        variable var_pop_count_result : signed(pop_count_result'range) := (others => '0');
        begin
        
            var_pop_count_result := (others => '0');
            
            -- Calcolo delle 6 lut
            for i in 0 to num_6_luts - 1 loop
                var_pop_count_result := var_pop_count_result + resize(signed(partial_pop_count(3 * i to 3 * (i + 1) - 1)), var_pop_count_result'length);
            end loop;
            
            -- Il risultato va moltiplicato per la codifica dei risultati delle lut con
            -- numero di ingressi pari
            var_pop_count_result := resize(2 * var_pop_count_result, var_pop_count_result'length);
            
            -- Calcolo delle ultime lut rimanenti
            if rest_input = 1 then
                var_pop_count_result := var_pop_count_result + resize(signed(partial_pop_count(partial_pop_count'right - 1 to partial_pop_count'right)), var_pop_count_result'length);
            elsif rest_input = 2 then
                var_pop_count_result := var_pop_count_result + resize(signed(partial_pop_count(partial_pop_count'right - 2 to partial_pop_count'right)), var_pop_count_result'length);
            elsif rest_input = 3 then
                var_pop_count_result := var_pop_count_result + resize(signed(partial_pop_count(partial_pop_count'right - 2 to partial_pop_count'right)), var_pop_count_result'length);
            elsif rest_input = 4 then
                var_pop_count_result := var_pop_count_result + resize(signed(partial_pop_count(partial_pop_count'right - 3 to partial_pop_count'right)), var_pop_count_result'length);
            elsif rest_input = 5 then
                var_pop_count_result := var_pop_count_result + resize(signed(partial_pop_count(partial_pop_count'right - 3 to partial_pop_count'right)), var_pop_count_result'length);
            end if;
            
            pop_count_result <= STD_LOGIC_VECTOR(var_pop_count_result);
            
        end process;    

        ---------------------------
    
        -- Batchnorm phase
        bnpg : if (BATCH_POSITIVE = True) generate
            bn : entity work.batchNorm(positiveBatchNorm)
            Generic map (
                SHIFT => SHIFT,
                B => B
            )
            Port map ( 
                x => pop_count_result,
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
                x => pop_count_result,
                y => y
            );
        end generate;
        --------------------            

end LUT6;
