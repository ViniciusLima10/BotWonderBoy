import numpy as np
import random
import pickle
from collections import deque
import os
import math
import retro
import cv2
from controller import Controls, Command
from objects import PlayerCar, EnemyCar

def start_game(controller: Controls, game):
    controller.input_commands([Command.START])

    controller.update_inputs()
    action = controller.get_action_array()

    game.set_button_mask(action)

    for _ in range(5):
        game.step()

    controller.clear_buttons()
    controller.input_commands([Command.B])

    action = controller.get_action_array()
    game.set_button_mask(action)

    for _ in range(5):
        game.step()

    controller.clear_buttons()


def get_frame(game):
    frame = game.get_screen()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    return frame


def detect_cars(frame):
    result = frame.copy()

    # Resize to original game resolution for consistent processing
    small = cv2.resize(frame, (160, 210), interpolation=cv2.INTER_AREA)

    # Define Region of Interest
    y_start, y_end = 50, 155
    x_start, x_end = 10, small.shape[1]
    roi = small[y_start:y_end, x_start:x_end]

    # Create masks
    white_mask = cv2.inRange(roi, (190, 190, 190), (205, 205, 205))  # Player car (white)
    enemy_mask = cv2.inRange(roi, (15, 15, 15), (250, 190, 190))     # Enemy cars (darker)

    white_mask = cv2.dilate(white_mask, None, iterations=1)
    enemy_mask = cv2.dilate(enemy_mask, None, iterations=1)

    # === Detect Player ===
    player = None
    player_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in player_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 5 < w < 20 and 5 < h < 20:
            abs_x = (x + x_start) * 3
            abs_y = (y + y_start) * 3
            player = PlayerCar(abs_x, abs_y, w * 3, h * 3)
            break

    # === Detect Enemies ===
    enemies = []
    enemy_contours, _ = cv2.findContours(enemy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in enemy_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 5 < w < 20 and 5 < h < 20:
            abs_x = (x + x_start) * 3
            abs_y = (y + y_start) * 3
            enemies.append(EnemyCar(abs_x, abs_y, w * 3, h * 3))

    # === Debug Drawing ===
    if player:
        cv2.rectangle(result, (player.x, player.y), (player.x + player.w, player.y + player.h), (255, 255, 255), 2)
        cv2.putText(result, "Player", (player.x, player.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    for enemy in enemies:
        cv2.rectangle(result, (enemy.x, enemy.y), (enemy.x + enemy.w, enemy.y + enemy.h), (0, 0, 255), 2)
        cv2.putText(result, "Enemy", (enemy.x, enemy.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return player, enemies, result


def detect_track_limits(frame, debug=False):
    """
    Detects the left and right road limits using dilated light-tile contours.
    Returns (left_limit, right_limit) in full-res coordinates.
    """
    # Resize to original resolution
    small = cv2.resize(frame, (160, 210), interpolation=cv2.INTER_AREA)
    display = small.copy()

    # Define ROI (ignore sky/hood, focus on track)
    y_start, y_end = 100, 170
    x_start, x_end = 10, 150
    roi = small[y_start:y_end, x_start:x_end]

    # Color mask to detect white/gray track edge squares
    edge_mask = cv2.inRange(roi, (165, 165, 165), (255, 255, 255))

    # Dilation to make edge tiles more continuous
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=2)

    # Contour detection
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the leftmost and rightmost positions
    left_limit = None
    right_limit = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 2:
            abs_x = x + x_start
            if left_limit is None or abs_x < left_limit:
                left_limit = abs_x
            if right_limit is None or abs_x + w > right_limit:
                right_limit = abs_x + w

    # Fallback if no edges found
    if left_limit is None or right_limit is None:
        left_limit, right_limit = 0, 160

    # Debug Overlay
    if debug:
        debug_display = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
        debug_display = cv2.resize(debug_display, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

        # Scale limits for debug view
        cv2.line(debug_display, ((left_limit or 0)*3, 0), ((left_limit or 0)*3, debug_display.shape[0]), (0, 255, 255), 2)
        cv2.line(debug_display, ((right_limit or 159)*3, 0), ((right_limit or 159)*3, debug_display.shape[0]), (0, 255, 255), 2)

        cv2.imshow("Track Debug", debug_display)

    return (left_limit * 3), (right_limit * 3)   # Convert to full-res
    #return (left_limit * 3)  + 35, (right_limit * 3) + 35   # Convert to full-res


def draw_track_limits_on_frame(frame, left_limit, right_limit, color=(0, 255, 255)):
    """
    Draws vertical lines for the detected track limits directly on the full-resolution frame.
    """
    h = frame.shape[0]
    cv2.line(frame, (left_limit, 0), (left_limit, h), color, 2)
    cv2.line(frame, (right_limit, 0), (right_limit, h), color, 2)

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.model_file = "enduro_q_model.pkl"
        
    def _build_model(self):
        # Initialize Q-table with zeros
        # For simplicity, we'll use a dictionary-based Q-table
        # In a more advanced implementation, you could use a neural network
        return {}
    
    def get_state_key(self, state):
        # Convert state to a hashable key (tuple) for dictionary storage
        return tuple(state)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def get_q_values(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.model:
            # Initialize with small random values if state not seen before
            self.model[state_key] = np.random.randn(self.action_size) * 0.01
        return self.model[state_key]
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random action for exploration
            return random.randrange(self.action_size)
        
        # Get Q-values for current state
        q_values = self.get_q_values(state)
        # Return best action
        return np.argmax(q_values)
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            
            # Current Q-value
            current_q = self.get_q_values(state)[action]
            
            if done:
                target = reward
            else:
                # Get max Q-value for next state
                next_q_values = self.get_q_values(next_state)
                target = reward + self.gamma * np.amax(next_q_values)
            
            # Update Q-value
            self.model[state_key][action] += self.learning_rate * (target - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.model_file}")
            self.epsilon = self.epsilon_min  # Set to minimal exploration after loading
            return True
        return False

def get_state_representation(player, enemies, left_limit, right_limit):
    """
    Convert game state to a simplified representation for Q-learning
    Returns a normalized state vector
    """
    if player is None:
        return np.zeros(8)  # Fallback if player not detected
    
    px, py = player.center()
    
    # Find closest enemy in front
    closest_enemy = None
    min_dist = float('inf')
    
    for enemy in enemies:
        ex, ey = enemy.center()
        if ey < py:  # Enemy is in front of player
            dist = math.sqrt((ex - px)**2 + (ey - py)**2)
            if dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
    
    # Normalized state representation
    state = [
        px / 480,  # Normalized x position (0-1)
        (py - 50) / 300,  # Normalized y position
        (px - left_limit) / 480,  # Distance to left track limit
        (right_limit - px) / 480,  # Distance to right track limit
        0,  # Placeholder for closest enemy x
        0,  # Placeholder for closest enemy y
        0,  # Placeholder for closest enemy distance
        1 if min_dist < 100 else 0  # Danger flag
    ]
    
    if closest_enemy:
        ex, ey = closest_enemy.center()
        state[4] = ex / 480
        state[5] = ey / 300
        state[6] = min_dist / 300
    
    return np.array(state)

def train_model(episodes=1000, batch_size=32):
    """
    Train the Q-learning model
    """
    # Initialize game and controller
    game = retro.RetroEmulator("game-folder/Enduro-Atari2600.a26")
    controller = Controls()
    
    # Initialize Q-learning agent
    state_size = 8  # Size of our state representation
    action_size = 4  # Possible actions: LEFT, RIGHT, ACCELERATE, NOOP
    agent = QLearningAgent(state_size, action_size)
    
    # Try to load existing model
    agent.load_model()
    
    for e in range(episodes):
        # Reset game
        start_game(controller, game)
        state = np.zeros(state_size)
        total_reward = 0
        done = False
        frame_count = 0
        cars_passed = 0  # Track cars passed in current episode
        
        while not done and frame_count < 3000:
            frame = get_frame(game)

            cv2.imshow("Enduro", frame)
            
            # Detect game state
            player, enemies, _ = detect_cars(frame)
            left_limit, right_limit = detect_track_limits(frame, debug=False)
            
            # Get state representation
            next_state = get_state_representation(player, enemies, left_limit, right_limit)
            
            # Get action from agent
            action = agent.act(next_state)
            
            # Map action to game commands
            commands = []
            if action == 0:  # LEFT
                commands.append(Command.LEFT)
            elif action == 1:  # RIGHT
                commands.append(Command.RIGHT)
            elif action == 2:  # ACCELERATE
                commands.append(Command.B)
            
            # Execute action
            controller.clear_buttons()
            if commands:
                controller.input_commands(commands)
            action_array = controller.get_action_array()
            game.set_button_mask(action_array)
            game.step()
            
            # Calculate reward
            reward = 0.1  # Small reward for surviving
            
            if player:
                # Reward for staying centered
                track_center = (left_limit + right_limit) / 2
                distance_from_center = abs(player.center()[0] - track_center)
                reward += 0.05 * (1 - distance_from_center / (right_limit - left_limit))
                
                # Reward for passing cars (count enemies that moved behind player)
                current_enemies_behind = sum(1 for enemy in enemies if enemy.center()[1] > player.center()[1])
                if hasattr(player, 'last_enemy_count'):
                    if current_enemies_behind > player.last_enemy_count:
                        cars_passed += (current_enemies_behind - player.last_enemy_count)
                        reward += 0.2 * (current_enemies_behind - player.last_enemy_count)
                player.last_enemy_count = current_enemies_behind
                
                # Penalty for collisions (detected by sudden position changes)
                current_pos = player.center()
                if hasattr(player, 'last_pos'):
                    pos_change = abs(current_pos[0] - player.last_pos[0]) + abs(current_pos[1] - player.last_pos[1])
                    if pos_change > 20:  # Sudden large movement indicates collision
                        reward -= 1.0
                player.last_pos = current_pos
            
            # Check if game over (player not detected)
            done = player is None
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            frame_count += 1
            
            # Train on batch
            if frame_count % 10 == 0:
                agent.replay(batch_size)
            
            if frame_count % 100 == 0:
                print(f"Frame: {frame_count}, Cars passed: {cars_passed}, Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                done = True
                break
        
        print(f"Episode: {e + 1}/{episodes}, Cars passed: {cars_passed}, Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Save model periodically
        if e % 10 == 0:
            agent.save_model()
    
    # Save final model
    agent.save_model()
    cv2.destroyAllWindows()

def run_trained_model():
    """
    Run the game using the trained Q-learning model
    """
    # Initialize game and controller
    game = retro.RetroEmulator("game-folder/Enduro-Atari2600.a26")
    controller = Controls()
    
    # Initialize Q-learning agent
    state_size = 8
    action_size = 4
    agent = QLearningAgent(state_size, action_size)
    
    # Load trained model
    if not agent.load_model():
        print("No trained model found. Please train first.")
        return
    
    # Start game
    start_game(controller, game)
    
    while True:
        frame = get_frame(game)
        
        # Detect game state
        player, enemies, debug_frame = detect_cars(frame)
        left_limit, right_limit = detect_track_limits(frame, debug=False)
        
        # Get state representation
        state = get_state_representation(player, enemies, left_limit, right_limit)
        
        # Get action from agent (no exploration)
        current_epsilon = agent.epsilon
        agent.epsilon = 0  # No exploration
        action = agent.act(state)
        agent.epsilon = current_epsilon
        
        # Map action to game commands
        commands = []
        if action == 0:  # LEFT
            commands.append(Command.LEFT)
        elif action == 1:  # RIGHT
            commands.append(Command.RIGHT)
        elif action == 2:  # ACCELERATE
            commands.append(Command.B)
        # else NOOP
        
        # Execute action
        controller.clear_buttons()
        if commands:
            controller.input_commands(commands)
        action_array = controller.get_action_array()
        game.set_button_mask(action_array)
        game.step()
        
        # Display
        draw_track_limits_on_frame(debug_frame, left_limit, right_limit)
        cv2.imshow("Enduro Q-Learning", debug_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cv2.destroyAllWindows()

# To train the model:
train_model(episodes=100)

# To run with trained model:
#run_trained_model()