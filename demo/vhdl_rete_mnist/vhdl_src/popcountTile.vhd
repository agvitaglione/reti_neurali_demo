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

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity popcountTile is
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
    );  
end popcountTile;

architecture Behavioral of popcountTile is

    signal y_temp : STD_LOGIC_VECTOR(y'range) := (others => '0');

begin

    y <= y_temp;

    sum: process (x)
        variable count : signed(y'range) := (others => '0');
        begin
            count := (others => '0');
            for i in x'range loop
                if x(i) = '0' then
                    count := count - 1;
                else
                    count := count + 1;
                end if;
            end loop;
            y_temp <= STD_LOGIC_VECTOR(count); 
    end process; 
    
end Behavioral;
