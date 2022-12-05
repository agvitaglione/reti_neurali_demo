----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 05.11.2022 10:59:01
-- Design Name: 
-- Module Name: popcount - Behavioral
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

entity popcount is
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
    );  
end popcount;

architecture Behavioral of popcount is

    Component popcountTile
        Port ( 
            x : in STD_LOGIC_VECTOR;
            y : out STD_LOGIC_VECTOR
        );  
    end component;
    
    type vect is array(0 to (x'LENGTH - x'LENGTH mod 16) / 16 - 1) of STD_LOGIC_VECTOR(0 to 5);
    signal y_temp : vect := (others => (others => '0'));
    signal y_temp_mod : STD_LOGIC_VECTOR(0 to 4) := (others => '0');
     
begin

    cond1 : if x'LENGTH >= 16 generate
        gen : for i in y_temp'RANGE generate
            popcnt : popcountTile
                Port map (
                    x => x(16 * i to 16 * (i + 1) - 1),
                    y => y_temp(i)
                );
        end generate;
    end generate;
    
    cond2 : if x'LENGTH mod 16 /= 0 generate
        popcnt : popcountTile
            Port map (
                x => x(x'LENGTH - x'LENGTH mod 16 to  x'LENGTH - 1),
                y => y_temp_mod
            );
    end generate;
    
    sum : process(y_temp, y_temp_mod)
    variable result : signed(y'RANGE) := (others => '0'); 
    begin
        result := (others => '0');
        if x'LENGTH >= 16  then
            for i in y_temp'RANGE loop
                result := result + resize(signed(y_temp(i)), y'LENGTH);
            end loop;
         end if;
        result := result + resize(signed(y_temp_mod), y'LENGTH);
        y <= STD_LOGIC_VECTOR(result);
    end process;

end Behavioral;
