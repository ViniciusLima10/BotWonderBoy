from time import time
from enum import Enum

# Command mapping — designed for human testing and logic handling
class Command(Enum):
    B = 0        # Accelerate
    UNKNOWN = 1 
    SELECT = 2
    START = 3    # RESET
    UP = 4 
    DOWN = 5 
    LEFT = 6     # Steer Left
    RIGHT = 7    # Steer Right
    A = 8

# Key to Command mapping
KEY_MAP = {
    ord(' '): Command.B,
    ord('2'): Command.UNKNOWN,
    ord('1'): Command.SELECT,
    ord('0'): Command.START,
    ord('w'): Command.UP,
    ord('s'): Command.DOWN,
    ord('a'): Command.LEFT,
    ord('d'): Command.RIGHT,
    ord('c'): Command.A
}

# Input with timestamp for button press duration
class Input():
    def __init__(self, command):
        self.command = command
        self.time = time()
    
    def __repr__(self):
        return f"{self.command} ({self.time})"

# Main control handler
class Controls:
    def __init__(self):
        self.inputs = []
        self.buttons = [0] * 9  # 9 logical buttons from Command Enum
        self.quit = False
        self.save = False
        self.manual = False

    def clear_buttons(self):
        self.buttons = [0] * 9  
        self.inputs = []

    def input_commands(self, commands, hold=True):
        for command in commands:
            if hold:
                self.inputs = [inp for inp in self.inputs if inp.command.value != command.value]
            if hold or self.buttons[command.value] == 0:
                self.inputs.append(Input(command))
                self.buttons[command.value] = 1

    def update_inputs(self):        
        current_time = time()
        self.inputs = [inp for inp in self.inputs if current_time - inp.time <= 0.001]
        # Rebuild button state
        self.buttons = [0] * 9
        for inp in self.inputs:
            self.buttons[inp.command.value] = 1

    def process_key(self, key):
        if key != 255 and key in KEY_MAP:
            self.clear_buttons()
            self.input_commands([KEY_MAP[key]])

        if key == ord('q'):
            self.quit = True
        
        if key == ord('p'):
            self.save = True

        if key == ord('m'):
            self.manual = not self.manual

    def get_action_array(self):
        """
        Maps Command buttons to the actual Atari 2600 action array:
        [LEFT, RIGHT, UP, DOWN, BUTTON, RESET, SELECT]
        """
        return [
            self.buttons[Command.B.value],     # LEFT
            self.buttons[Command.UNKNOWN.value],    # RIGHT
            self.buttons[Command.SELECT.value],       # UP (accelerate)
            self.buttons[Command.START.value],     # DOWN (brake)
            self.buttons[Command.UP.value],
            self.buttons[Command.DOWN.value],        # BUTTON (unused in Enduro)
            self.buttons[Command.LEFT.value],    # RESET (Start game)
            self.buttons[Command.RIGHT.value],   # SELECT
            self.buttons[Command.A.value],
        ]