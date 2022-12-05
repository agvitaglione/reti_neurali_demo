----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 07.11.2022 19:01:24
-- Design Name: 
-- Module Name: layer_template - Behavioral
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
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity layer is
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
   );
end layer;

architecture layer_1 of layer is

    Component perceptron
    Generic (
        WEIGHTS : STD_LOGIC_VECTOR;
        SHIFT : Integer;     -- number of shift (left + or right -)
        B : Integer;          -- bias of batch norm
        BATCH_POSITIVE : Boolean    
    );
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
   );
    end component;
    
    -- Signals
    %SIGNALS_1%
    
    -- Dummies
    %DUMMIES_1% 

begin

    %BODY_1%

--    p1 : perceptron
--        Generic map (
--            WEIGHTS => "",
--            SHIFT => "",
--            B => "",
--            BATCH_POSITIVE => ""
--        )
--        Port map (
--            x => x(0) & x(2) & ...,
--            y => ""
--        );

end layer_1;

architecture layer_2 of layer is

    Component perceptron
    Generic (
        WEIGHTS : STD_LOGIC_VECTOR;
        SHIFT : Integer;     -- number of shift (left + or right -)
        B : Integer;          -- bias of batch norm
        BATCH_POSITIVE : Boolean    
    );
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
   );
    end component;
    
    -- Signals
    %SIGNALS_2%
    
    -- Dummies
    %DUMMIES_2% 
    
begin

    %BODY_2%

end layer_2;

architecture layer_3 of layer is

    Component perceptron
    Generic (
        WEIGHTS : STD_LOGIC_VECTOR;
        SHIFT : Integer;     -- number of shift (left + or right -)
        B : Integer;          -- bias of batch norm
        BATCH_POSITIVE : Boolean    
    );
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
   );
    end component;

    -- Signals
    %SIGNALS_3%
    
    -- Dummies
    %DUMMIES_3% 

begin

    %BODY_3%

end layer_3;

architecture layer_4 of layer is

    Component perceptron
    Generic (
        WEIGHTS : STD_LOGIC_VECTOR;
        SHIFT : Integer;     -- number of shift (left + or right -)
        B : Integer;          -- bias of batch norm
        BATCH_POSITIVE : Boolean    
    );
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
   );
    end component;
    
    -- Signals
    %SIGNALS_4%
    
    -- Dummies
    %DUMMIES_4% 

begin

    %BODY_4%

end layer_4;

architecture layer_5 of layer is

    Component perceptron
    Generic (
        WEIGHTS : STD_LOGIC_VECTOR;
        SHIFT : Integer;     -- number of shift (left + or right -)
        B : Integer;          -- bias of batch norm
        BATCH_POSITIVE : Boolean    
    );
    Port ( 
        x : in STD_LOGIC_VECTOR;
        y : out STD_LOGIC_VECTOR
   );
    end component;
    
    -- Signals
    %SIGNALS_5%
    
    -- Dummies
    %DUMMIES_5% 

begin

    %BODY_5%

end layer_5;
