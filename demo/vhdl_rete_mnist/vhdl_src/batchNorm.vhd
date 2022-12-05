----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 07.11.2022 16:53:32
-- Design Name: 
-- Module Name: batchNorm - Behavioral
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

entity batchNorm is
    Generic(
        SHIFT : Integer;     -- number of shift (positive left shift, negative right shift)
        B : Integer          -- bias of batch norm   
    );
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
    );
end batchNorm;

architecture positiveBatchNorm of batchNorm is

begin

    op : process(x)
    variable x_shift : signed(y'range) := (others => '0');
    variable y_temp : signed(y'range) := (others => '0');
    begin
        if SHIFT > 0 then  
            x_shift := shift_left(resize(signed(x), x_shift'length), SHIFT);
        else
            x_shift := resize(shift_right(signed(x), -SHIFT), x_shift'length);
        end if;
        y_temp := x_shift + to_signed(B, y_temp'length);
        y <= std_logic_vector(y_temp);
    end process;
    
    
end positiveBatchNorm;

architecture negativeBatchNorm of batchNorm is

begin

    op : process(x)
    variable x_shift : signed(x'range) := (others => '0');
    variable y_temp : signed(y'range) := (others => '0');
    begin
        if SHIFT > 0 then
            x_shift := shift_left(resize(signed(x), x_shift'length), SHIFT);
        else
            x_shift := resize(shift_right(signed(x), -SHIFT), x_shift'length);
        end if;
        y_temp := not(x_shift) + to_signed(B+1, y_temp'length);
        y <= std_logic_vector(y_temp);
    end process;
    
end negativeBatchNorm;
