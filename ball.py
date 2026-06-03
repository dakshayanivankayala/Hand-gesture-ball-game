import cv2
import mediapipe as mp
import random
import numpy as np

# =========================
# CAMERA SETUP
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

GAME_W = 480
GAME_H = 360

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# GAME VARIABLES
# =========================
player_x = GAME_W / 2
player_y = GAME_H / 2
score = 0
game_over = False
speed = 10
start_time = cv2.getTickCount()

# =========================
# HAND TRACKING VARIABLES
# =========================
last_hand_x = 0.5
last_hand_y = 0.5
DEAD_ZONE = 0.005


# =========================
# BALL SYSTEM
# =========================
def create_ball():
    t = random.randint(1, 3)
    return [random.randint(40, GAME_W - 40), random.randint(-150, 0), t]


balls = [create_ball(), create_ball(), create_ball()]


def reset_game():
    global player_x, player_y, score, game_over, balls, speed
    global last_hand_x, last_hand_y, start_time
    player_x = GAME_W / 2
    player_y = GAME_H / 2
    score = 0
    game_over = False
    speed = 10
    last_hand_x = 0.5
    last_hand_y = 0.5
    start_time = cv2.getTickCount()
    balls = [create_ball(), create_ball(), create_ball()]


# =========================
# MAIN LOOP
# =========================
while True:
    ret, raw_frame = cap.read()
    if not ret:
        break

    raw_frame = cv2.flip(raw_frame, 1)
    raw_frame = cv2.resize(raw_frame, (GAME_W, GAME_H))
    h, w, _ = raw_frame.shape

    # Webcam panel (right side)
    cam_panel = raw_frame.copy()

    # Game canvas (left side) — black background
    game_frame = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)

    # =========================
    # GAME OVER SCREEN
    # =========================
    if game_over:
        game_frame[:] = (0, 0, 0)
        cv2.putText(game_frame, "GAME OVER", (120, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        cv2.putText(game_frame, f"Score : { score}", (160, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(game_frame, "Restart", (170, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        combined = np.hstack([game_frame, cam_panel])
        cv2.imshow("Ball Dodge", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            reset_game()
        elif key == ord('q'):
            break
        continue

    # =========================
    # HAND TRACKING
    # =========================
    rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_detected = False

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        raw_x = hand.landmark[8].x
        raw_y = hand.landmark[8].y

        if abs(raw_x - last_hand_x) > DEAD_ZONE:
            last_hand_x = raw_x
        if abs(raw_y - last_hand_y) > DEAD_ZONE:
            last_hand_y = raw_y

        hand_detected = True

        # Draw hand skeleton on cam panel
        mp.solutions.drawing_utils.draw_landmarks(
            cam_panel,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 200, 100), thickness=1, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1)
        )

        # Highlight index fingertip
        dot_x = int(last_hand_x * w)
        dot_y = int(last_hand_y * h)
        cv2.circle(cam_panel, (dot_x, dot_y), 12, (0, 255, 255), -1)
        cv2.circle(cam_panel, (dot_x, dot_y), 12, (255, 255, 255), 2)

    # Move player toward finger
    target_x = max(25, min(int(last_hand_x * w), w - 25))
    target_y = max(25, min(int(last_hand_y * h), h - 25))

    smooth = 0.55 if hand_detected else 0.2
    player_x += (target_x - player_x) * smooth
    player_y += (target_y - player_y) * smooth
    player_x = max(25, min(player_x, w - 25))
    player_y = max(25, min(player_y, h - 25))

    # =========================
    # DRAW PLAYER
    # =========================
    px, py = int(player_x), int(player_y)
    cv2.rectangle(game_frame, (px - 27, py - 12), (px + 27, py + 12), (255, 255, 255), 2)
    cv2.rectangle(game_frame, (px - 25, py - 10), (px + 25, py + 10), (0, 255, 120), -1)
    cv2.circle(game_frame, (px, py), 4, (0, 160, 80), -1)

    # =========================
    # BALLS
    # =========================
    new_balls = []
    for bx, by, btype in balls:
        by += speed

        if btype == 1:
            color = (0, 200, 255)
            value = 1
        elif btype == 2:
            color = (255, 215, 0)
            value = 5
        else:
            color = (0, 0, 255)
            value = 0

        cv2.circle(game_frame, (bx, by), 15, color, -1)
        cv2.circle(game_frame, (bx, by), 15, (255, 255, 255), 1)

        label = "+1" if btype == 1 else "+5" if btype == 2 else "X"
        cv2.putText(game_frame, label, (bx - 9, by + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Collision
        if abs(bx - player_x) < 38 and abs(by - player_y) < 28:
            if btype == 3:
                game_over = True
            else:
                score += value
            continue

        if by > GAME_H:
            new_balls.append(create_ball())
        else:
            new_balls.append([bx, by, btype])

    balls = new_balls
    while len(balls) < 3:
        balls.append(create_ball())

    # =========================
    # SPEED SCALING
    # =========================
    speed = min(10 + (score // 10), 25)

    # =========================
    # ELAPSED TIME
    # =========================
    elapsed = int((cv2.getTickCount() - start_time) / cv2.getTickFrequency())

    # =========================
    # HUD — Game Panel
    # =========================
    cv2.putText(game_frame, "BALL DODGE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(game_frame, f"Score: {score}", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(game_frame, f"Speed: {speed}", (10, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(game_frame, f"Time:  {elapsed}s", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(game_frame, "+1  +5  RED=DEAD", (10, 348),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    # =========================
    # HUD — Cam Panel
    # =========================
    s_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    s_text = "HAND: ON" if hand_detected else "HAND: OFF"
    cv2.putText(cam_panel, "WEBCAM", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 160, 255), 2)
    cv2.putText(cam_panel, s_text, (w - 135, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 2)

    # =========================
    # STITCH & SHOW
    # =========================
    combined = np.hstack([game_frame, cam_panel])
    cv2.imshow("Ball Dodge", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reset_game()

cap.release()
cv2.destroyAllWindows()