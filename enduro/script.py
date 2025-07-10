import retro
import cv2
import numpy as np
from controller import Controls, Command
from objects import PlayerCar, EnemyCar

import math

def decide_movement(player, enemies, left_track_limit, right_track_limit):
    """
    Uses potential fields and intelligent overtaking:
    - Avoids enemies
    - Steers around close obstacles using track width
    - Avoids borders
    - Accelerates when safe
    """

    repulsion_strength = 8000
    wall_strength = 12000
    brake_threshold = 100
    danger_zone_width = 65
    lane_margin = 60

    force_x = 0
    should_accelerate = True
    forced_evasion = None

    px, py = player.center()

    # === Detect Enemy Danger & Evasion Direction ===
    for enemy in enemies:
        ex, ey = enemy.center()
        dx = ex - px
        dy = py - ey

        if dy > 0:  # enemy is ahead
            dist_sq = dx ** 2 + dy ** 2
            if dist_sq == 0:
                dist_sq = 0.01

            fx = repulsion_strength * dx / dist_sq
            force_x += fx

            # Danger zone: consider overtaking
            if abs(dx) < danger_zone_width and dy < brake_threshold:
                should_accelerate = False

                # Check track space to left and right
                dist_left = max(px - left_track_limit, 1)
                dist_right = max(right_track_limit - px, 1)

                if dist_left > dist_right:
                    forced_evasion = Command.LEFT
                else:
                    forced_evasion = Command.RIGHT

    # === Wall Repulsion ===
    dist_left = max(px - left_track_limit, 1)
    dist_right = max(right_track_limit - px, 1)

    if dist_left < lane_margin:
        force_x += wall_strength / (dist_left ** 2)
    if dist_right < lane_margin:
        force_x -= wall_strength / (dist_right ** 2)

    # === Direction Decision ===
    move = None
    if forced_evasion:
        move = forced_evasion  # Prioritize safety
    elif force_x < -10:
        move = Command.LEFT
    elif force_x > 10:
        move = Command.RIGHT

    # === Command Output ===
    commands = []
    if should_accelerate:
        commands.append(Command.B)
    if move:
        commands.append(move)

    return commands





def start_game(controller: Controls):
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
    enemy_mask = cv2.inRange(roi, (15, 15, 15), (190, 190, 190))     # Enemy cars (darker)

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

    return (left_limit * 3) + 50, (right_limit * 3) - 50  # Convert to full-res


def draw_track_limits_on_frame(frame, left_limit, right_limit, color=(0, 255, 255)):
    """
    Draws vertical lines for the detected track limits directly on the full-resolution frame.
    """
    h = frame.shape[0]
    cv2.line(frame, (left_limit, 0), (left_limit, h), color, 2)
    cv2.line(frame, (right_limit, 0), (right_limit, h), color, 2)



# Main loop
game = retro.RetroEmulator("game-folder/Enduro-Atari2600.a26")

controller = Controls()

start_game(controller)

for _ in range(5000):
    frame = get_frame(game)

    controller.clear_buttons()


    # Detect player and opponents
    #frame_with_detections = detect_cars(frame)

    player, enemies, debug_frame = detect_cars(frame)
    left_limit, right_limit = detect_track_limits(frame, debug=False)

    draw_track_limits_on_frame(debug_frame, left_limit, right_limit)

    # Display
    cv2.imshow("Enduro Detection", debug_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


    if player:
        commands = decide_movement(player, enemies, left_limit, right_limit)
        controller.clear_buttons()
        controller.input_commands(commands)
    else:
        controller.clear_buttons()
        controller.input_commands([Command.B])

    action = controller.get_action_array()

    game.set_button_mask(action)

    game.step()

cv2.destroyAllWindows()
