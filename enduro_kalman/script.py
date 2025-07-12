import retro
import cv2
import numpy as np
from controller import Controls, Command
from objects import PlayerCar, EnemyCar
from kalman import KalmanTrackedObject

def decide_movement(player_pos, enemies, left_track_limit, right_track_limit):
    repulsion_strength = 8000
    wall_strength = 12000
    brake_threshold = 100
    danger_zone_width = 65
    lane_margin = 20

    force_x = 0
    should_accelerate = True
    forced_evasion = None

    px, py = player_pos

    for enemy in enemies:
        ex, ey = enemy.center()
        dx = ex - px
        dy = py - ey

        if dy > 0:
            dist_sq = dx ** 2 + dy ** 2
            if dist_sq == 0:
                dist_sq = 0.01

            fx = repulsion_strength * dx / dist_sq
            force_x += fx

            if abs(dx) < danger_zone_width and dy < brake_threshold:
                should_accelerate = False

                dist_left = max(px - left_track_limit, 1)
                dist_right = max(right_track_limit - px, 1)

                if dist_left > dist_right:
                    forced_evasion = Command.LEFT
                else:
                    forced_evasion = Command.RIGHT

    dist_left = max(px - left_track_limit, 1)
    dist_right = max(right_track_limit - px, 1)

    if dist_left < lane_margin:
        force_x += wall_strength / (dist_left ** 2)
    if dist_right < lane_margin:
        force_x -= wall_strength / (dist_right ** 2)

    move = None
    if forced_evasion:
        move = forced_evasion
    elif force_x < -10:
        move = Command.LEFT
    elif force_x > 10:
        move = Command.RIGHT

    commands = []
    if should_accelerate:
        commands.append(Command.B)
    if move:
        commands.append(move)

    return commands

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

    return player, enemies, result

def detect_track_limits(frame, debug=False):
    small = cv2.resize(frame, (160, 210), interpolation=cv2.INTER_AREA)
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

    return (left_limit * 3), (right_limit * 3)

def draw_track_limits_on_frame(frame, left_limit, right_limit, color=(0, 255, 255)):
    h = frame.shape[0]
    cv2.line(frame, (left_limit, 0), (left_limit, h), color, 2)
    cv2.line(frame, (right_limit, 0), (right_limit, h), color, 2)

# ==== Main Loop ====
game = retro.RetroEmulator("game-folder/Enduro-Atari2600.a26")
controller = Controls()
start_game(controller, game)

player_tracker = None
enemy_trackers = []

for _ in range(3000):
    frame = get_frame(game)
    controller.clear_buttons()

    player, enemies, debug_frame = detect_cars(frame)
    left_limit, right_limit = detect_track_limits(frame, debug=False)
    draw_track_limits_on_frame(debug_frame, left_limit, right_limit)

    # Kalman for player
    if player:
        if not player_tracker:
            player_tracker = KalmanTrackedObject(*player.center())
        else:
            player_tracker.update(*player.center())
        px, py = player_tracker.predict()
    else:
        px, py = player.center() if player else (80 * 3, 160 * 3)

    # Kalman for enemies
    new_trackers = []
    filtered_enemies = []
    for enemy in enemies:
        ex, ey = enemy.center()
        matched = None
        for tracker in enemy_trackers:
            tx, ty = tracker.current()
            if abs(tx - ex) < 50 and abs(ty - ey) < 50:
                matched = tracker
                break
        if matched:
            matched.update(ex, ey)
            new_trackers.append(matched)
        else:
            new_trackers.append(KalmanTrackedObject(ex, ey))
    enemy_trackers = new_trackers

    for et in enemy_trackers:
        ex, ey = et.predict()
        filtered_enemies.append(EnemyCar(ex, ey, 15, 15))

    cv2.imshow("Enduro Detection", debug_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    commands = decide_movement((px, py), filtered_enemies, left_limit, right_limit)
    controller.clear_buttons()
    controller.input_commands(commands)

    action = controller.get_action_array()
    game.set_button_mask(action)
    game.step()

cv2.destroyAllWindows()
