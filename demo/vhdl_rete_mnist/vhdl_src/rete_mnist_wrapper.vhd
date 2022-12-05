----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 18.11.2022 14:41:11
-- Design Name: 
-- Module Name: rete_mnist_wrapper - Behavioral
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

entity rete_mnist_wrapper is
    Port ( 
        x : in STD_LOGIC;
        input_shift : in STD_LOGIC;
        output_shift : in STD_LOGIC;
        start : in STD_LOGIC;
        clk : in STD_LOGIC;
        y : out STD_LOGIC
    );
end rete_mnist_wrapper;

architecture Behavioral of rete_mnist_wrapper is

    Component rete_mnist
        Port (
            x : in STD_LOGIC_VECTOR(0 to 783);
            y : out STD_LOGIC_VECTOR(0 to 89)
        );
    end component;

signal input : STD_LOGIC_VECTOR(0 to 783) := (others => '0');
signal image : STD_LOGIC_VECTOR(0 to 783) := (others => '0');
signal y_shift_register : STD_LOGIC_VECTOR(0 to 89) := (others => '0');
signal rete_mnist_output : STD_LOGIC_VECTOR(0 to 89) := (others => '0');
signal shift_output_pressed : STD_LOGIC := '0';
signal shift_input_pressed : STD_LOGIC := '0';

signal computing_prediction : STD_LOGIC := '0';
signal counter : unsigned(0 to 3) := (others => '0');

begin

    y <= y_shift_register(0);

    rm : rete_mnist
        Port map (
            x => input,
            y => rete_mnist_output
        );
       
  
     -- Raccolta dati input mediante shift register
     process_input_shift : process(clk)
     begin
        if rising_edge(clk) then
            if input_shift = '1' AND shift_input_pressed = '0' then
                shift_input_pressed <= '1';
                image <= image(1 to image'RIGHT) & x;
            elsif input_shift = '0' AND shift_input_pressed = '1' then
                shift_input_pressed <= '0';
            end if;
        end if;
     end process;
     
     process_work : process(clk)
     begin
        if rising_edge(clk) then
        
            -- Avvio inizio predizione
            if start = '1' AND computing_prediction = '0' then
                input <= image;
                computing_prediction <= '1';
            end if;
        
            -- Se la rete sta predicendo, attendere la predizione
            if computing_prediction = '1' then
                if counter < 7 then
                    counter <= counter + 1;
                else
                    counter <= (others => '0');
                    computing_prediction <= '0';
                    y_shift_register <= rete_mnist_output; 
                end if;
             
             -- Se la rete non sta predicendo, è possibile eseguire lo shift dell'ouput
             elsif start = '0' AND computing_prediction = '0' then
                 if output_shift = '1' AND shift_output_pressed = '0' then
                    shift_output_pressed <= '1';
                    y_shift_register <= y_shift_register(1 to y_shift_register'RIGHT) & '0';
                elsif output_shift = '0' AND shift_output_pressed = '1' then
                    shift_output_pressed <= '0';
                end if;
             end if;     
        end if; -- clock rising
     end process;
     
end Behavioral;
