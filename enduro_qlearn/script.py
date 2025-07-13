import numpy as np
import random
import pickle
from collections import deque
import os
import math
import retro
import cv2
import time

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
    small = cv2.resize(frame, (160, 210), interpolation=cv2.INTER_AREA)

    y_start, y_end = 50, 155
    x_start, x_end = 10, small.shape[1]
    roi = small[y_start:y_end, x_start:x_end]

    white_mask = cv2.inRange(roi, (190, 190, 190), (205, 205, 205))
    enemy_mask = cv2.inRange(roi, (15, 15, 15), (250, 190, 190))

    white_mask = cv2.dilate(white_mask, None, iterations=1)
    enemy_mask = cv2.dilate(enemy_mask, None, iterations=1)

    player = None
    player_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in player_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 5 < w < 20 and 5 < h < 20:
            abs_x = (x + x_start) * 3
            abs_y = (y + y_start) * 3
            player = PlayerCar(abs_x, abs_y, w * 3, h * 3)
            break

    enemies = []
    enemy_contours, _ = cv2.findContours(enemy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in enemy_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 5 < w < 20 and 5 < h < 20:
            abs_x = (x + x_start) * 3
            abs_y = (y + y_start) * 3
            enemies.append(EnemyCar(abs_x, abs_y, w * 3, h * 3))

    if player:
        cv2.rectangle(result, (player.x, player.y), (player.x + player.w, player.y + player.h), (255, 255, 255), 2)
        cv2.putText(result, "Player", (player.x, player.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    for enemy in enemies:
        cv2.rectangle(result, (enemy.x, enemy.y), (enemy.x + enemy.w, enemy.y + enemy.h), (0, 0, 255), 2)
        cv2.putText(result, "Enemy", (enemy.x, enemy.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return player, enemies, result


def detect_track_limits(frame, debug=False):
    small = cv2.resize(frame, (160, 210), interpolation=cv2.INTER_AREA)
    display = small.copy()

    y_start, y_end = 100, 170
    x_start, x_end = 10, 150
    roi = small[y_start:y_end, x_start:x_end]

    edge_mask = cv2.inRange(roi, (165, 165, 165), (255, 255, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    if left_limit is None or right_limit is None:
        left_limit, right_limit = 0, 160

    if debug:
        debug_display = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
        debug_display = cv2.resize(debug_display, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.line(debug_display, ((left_limit or 0)*3, 0), ((left_limit or 0)*3, debug_display.shape[0]), (0, 255, 255), 2)
        cv2.line(debug_display, ((right_limit or 159)*3, 0), ((right_limit or 159)*3, debug_display.shape[0]), (0, 255, 255), 2)
        cv2.imshow("Track Debug", debug_display)

    return (left_limit * 3), (right_limit * 3)


def draw_track_limits_on_frame(frame, left_limit, right_limit, color=(0, 255, 255)):
    h = frame.shape[0]
    cv2.line(frame, (left_limit, 0), (left_limit, h), color, 2)
    cv2.line(frame, (right_limit, 0), (right_limit, h), color, 2)


class KalmanTrackedObject:
    def __init__(self, x, y):
        self.kalman = cv2.KalmanFilter(4, 2)  # Estado 4D, Medida 2D

        # Matriz de transição do estado
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x_new = x + vx
            [0, 1, 0, 1],  # y_new = y + vy
            [0, 0, 1, 0],  # vx_new = vx
            [0, 0, 0, 1]   # vy_new = vy
        ], dtype=np.float32)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # Medimos x diretamente
            [0, 1, 0, 0]   # Medimos y diretamente
        ], dtype=np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

    def update(self, x, y):
        """Atualiza o filtro com a nova posição observada."""
        self.kalman.correct(np.array([[x], [y]], dtype=np.float32))

    def predict(self):
        """Prediz a próxima posição."""
        predicted = self.kalman.predict()
        return (predicted[0][0], predicted[1][0])  # x, y

    def current(self):
        return (self.kalman.statePost[0][0], self.kalman.statePost[1][0])


class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.model_file = "enduro_q_model.pkl"
        
        # Kalman trackers para jogador e inimigos
        self.player_tracker = None
        self.enemy_trackers = []

    def _build_model(self):
        return {}
    
    def get_state_key(self, state):
        return tuple(state)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def get_q_values(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.model:
            self.model[state_key] = np.random.randn(self.action_size) * 0.01
        return self.model[state_key]
    
    def act(self, state):
        danger_flag = state[7]
        px = state[0] * 480
        enemy_x = state[4] * 480

        # Ações reativas simples para perigo e posição
        if danger_flag:
            if enemy_x < px:
                return 1
            else:
                return 0

        center = 240
        if abs(px - center) > 50:
            if px < center:
                return 1
            else:
                return 0

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.get_q_values(state)
        return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            current_q = self.get_q_values(state)[action]
            if done:
                target = reward
            else:
                next_q_values = self.get_q_values(next_state)
                target = reward + self.gamma * np.amax(next_q_values)
            self.model[state_key][action] += self.learning_rate * (target - current_q)
        
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
            self.epsilon = self.epsilon_min
            return True
        return False

    def update_kalman_trackers(self, player, enemies):
        # Atualiza ou cria tracker para o player
        if player:
            cx, cy = player.center()
            if self.player_tracker is None:
                self.player_tracker = KalmanTrackedObject(cx, cy)
            else:
                self.player_tracker.update(cx, cy)
        else:
            self.player_tracker = None
        
        # Atualiza trackers para inimigos
        # Para simplificar, vamos associar os inimigos detectados aos trackers existentes pela proximidade
        updated_trackers = []
        enemies_centers = [enemy.center() for enemy in enemies]

        # Se não há trackers, crie um para cada inimigo
        if not self.enemy_trackers:
            for cx, cy in enemies_centers:
                updated_trackers.append(KalmanTrackedObject(cx, cy))
        else:
            # Para cada inimigo detectado, encontre o tracker mais próximo
            for cx, cy in enemies_centers:
                min_dist = float('inf')
                closest_tracker = None
                for tracker in self.enemy_trackers:
                    tx, ty = tracker.current()
                    dist = math.sqrt((tx - cx)**2 + (ty - cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_tracker = tracker
                
                # Se estiver perto o suficiente, atualiza tracker, senão cria novo
                if min_dist < 50 and closest_tracker is not None:
                    closest_tracker.update(cx, cy)
                    updated_trackers.append(closest_tracker)
                else:
                    updated_trackers.append(KalmanTrackedObject(cx, cy))
        
        self.enemy_trackers = updated_trackers

    def get_filtered_positions(self):
        # Retorna a posição filtrada do player e do inimigo mais próximo à frente (menor y)
        if self.player_tracker is None:
            player_pos = None
        else:
            player_pos = self.player_tracker.current()

        if not self.enemy_trackers:
            enemy_pos = None
        else:
            # Escolhe inimigo mais próximo acima do player (menor y)
            if player_pos is None:
                enemy_pos = None
            else:
                px, py = player_pos
                enemy_ahead = None
                min_dist = float('inf')
                for tracker in self.enemy_trackers:
                    ex, ey = tracker.current()
                    if ey < py:
                        dist = math.sqrt((ex - px)**2 + (ey - py)**2)
                        if dist < min_dist:
                            min_dist = dist
                            enemy_ahead = (ex, ey)
                enemy_pos = enemy_ahead

        return player_pos, enemy_pos


def get_state_representation_filtered(player_pos, enemy_pos, left_limit, right_limit):
    if player_pos is None:
        return np.zeros(8)

    px, py = player_pos
    if enemy_pos is None:
        ex, ey = 0, 0
        dist = 1  # Normalizado
    else:
        ex, ey = enemy_pos
        dist = math.sqrt((ex - px)**2 + (ey - py)**2) / 300

    state = [
        px / 480,
        (py - 50) / 300,
        (px - left_limit) / 480,
        (right_limit - px) / 480,
        ex / 480,
        ey / 300,
        dist,
        1 if dist < 100/300 else 0
    ]

    return np.array(state)


def check_collision(player, enemies):
    if not player:
        return False
    for enemy in enemies:
        px1, py1 = player.x, player.y
        px2, py2 = player.x + player.w, player.y + player.h
        ex1, ey1 = enemy.x, enemy.y
        ex2, ey2 = enemy.x + enemy.w, enemy.y + enemy.h
        if px1 < ex2 and px2 > ex1 and py1 < ey2 and py2 > ey1:
            return True
    return False


def train_model(episodes=1000, batch_size=32, duration_minutes=60):
    game = retro.RetroEmulator("game-folder/Enduro-Atari2600.a26")
    controller = Controls()
    state_size = 8
    action_size = 4
    agent = QLearningAgent(state_size, action_size)
    agent.load_model()

    start_time = time.time()
    max_duration = duration_minutes * 60

    for e in range(episodes):
        if time.time() - start_time > max_duration:
            print("⏱️ Tempo de treino atingido. Encerrando.")
            break

        start_game(controller, game)
        state = np.zeros(state_size)
        total_reward = 0
        done = False
        frame_count = 0
        cars_passed = 0

        # Reset trackers no início de cada episódio
        agent.player_tracker = None
        agent.enemy_trackers = []

        while not done and frame_count < 3000:
            frame = get_frame(game)
            cv2.imshow("Enduro", frame)

            player, enemies, _ = detect_cars(frame)
            left_limit, right_limit = detect_track_limits(frame, debug=False)

            # Atualiza Kalman trackers
            agent.update_kalman_trackers(player, enemies)
            player_pos, enemy_pos = agent.get_filtered_positions()

            next_state = get_state_representation_filtered(player_pos, enemy_pos, left_limit, right_limit)
            action = agent.act(next_state)

            commands = []
            if action == 0:
                commands.append(Command.LEFT)
            elif action == 1:
                commands.append(Command.RIGHT)
            elif action == 2:
                commands.append(Command.B)

            controller.clear_buttons()
            if commands:
                controller.input_commands(commands)
            action_array = controller.get_action_array()
            game.set_button_mask(action_array)
            game.step()

            reward = 0.1
            if player:
                track_center = (left_limit + right_limit) / 2
                distance_from_center = abs(player_pos[0] - track_center) if player_pos else 0
                reward += 0.1 * (1 - distance_from_center / (right_limit - left_limit))

                current_enemies_behind = sum(1 for enemy in enemies if enemy.center()[1] > player.center()[1])
                if hasattr(player, 'last_enemy_count'):
                    if current_enemies_behind > player.last_enemy_count:
                        cars_passed += (current_enemies_behind - player.last_enemy_count)
                        reward += 0.2 * (current_enemies_behind - player.last_enemy_count)
                player.last_enemy_count = current_enemies_behind

                if check_collision(player, enemies):
                    reward -= 2.0
                    done = True

            if player is None:
                done = True

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            frame_count += 1

            if frame_count % 10 == 0:
                agent.replay(batch_size)

            if frame_count % 100 == 0:
                print(f"Frame: {frame_count}, Cars passed: {cars_passed}, Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                done = True
                break

        print(f"Episode: {e + 1}/{episodes}, Cars passed: {cars_passed}, Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        if e % 10 == 0:
            agent.save_model()

    agent.save_model()
    cv2.destroyAllWindows()


def run_trained_model():
    game = retro.RetroEmulator("game-folder/Enduro-Atari2600.a26")
    controller = Controls()
    state_size = 8
    action_size = 4
    agent = QLearningAgent(state_size, action_size)

    if not agent.load_model():
        print("No trained model found. Please train first.")
        return

    start_game(controller, game)

    # Reset trackers
    agent.player_tracker = None
    agent.enemy_trackers = []

    while True:
        frame = get_frame(game)
        player, enemies, debug_frame = detect_cars(frame)
        left_limit, right_limit = detect_track_limits(frame, debug=False)

        agent.update_kalman_trackers(player, enemies)
        player_pos, enemy_pos = agent.get_filtered_positions()

        state = get_state_representation_filtered(player_pos, enemy_pos, left_limit, right_limit)

        current_epsilon = agent.epsilon
        agent.epsilon = 0
        action = agent.act(state)
        agent.epsilon = current_epsilon

        commands = []
        if action == 0:
            commands.append(Command.LEFT)
        elif action == 1:
            commands.append(Command.RIGHT)
        elif action == 2:
            commands.append(Command.B)

        controller.clear_buttons()
        if commands:
            controller.input_commands(commands)
        action_array = controller.get_action_array()
        game.set_button_mask(action_array)
        game.step()

        draw_track_limits_on_frame(debug_frame, left_limit, right_limit)
        cv2.imshow("Enduro Q-Learning", debug_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# Para treinar o modelo:
train_model(episodes=10000, duration_minutes=300)

# Para executar o modelo treinado:
# run_trained_model()