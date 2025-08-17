import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Finger indices
finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
thumb_tip = 4

def get_hand_gesture(hand_landmarks):
    """Determine Rock, Paper, Scissors from landmarks."""
    fingers = []

    # Thumb
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
        fingers.append(1)  # Thumb up
    else:
        fingers.append(0)

    # Other fingers
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)  # Finger up
        else:
            fingers.append(0)

    # Rock
    if fingers[1:] == [0, 0, 0, 0]:
        return "Rock"
    # Paper
    elif fingers == [1, 1, 1, 1, 1] or fingers == [0, 1, 1, 1, 1]:
        return "Paper"
    # Scissors
    elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
        return "Scissors"
    else:
        return "Unknown"


# Game state
moves = ["Rock", "Paper", "Scissors"]
comp_move = None
winner = None
round_active = False
round_start_time = None

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    user_move = "Waiting..."
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_hand_gesture(hand_landmarks)

            # Start round if not active
            if not round_active and gesture in moves:
                round_active = True
                round_start_time = time.time()
                comp_move = random.choice(moves)  # computer picks once
                user_move = gesture

            # During round (freeze move)
            if round_active:
                user_move = gesture

    # Show choices only if round active
    if round_active:
        elapsed = time.time() - round_start_time
        cv2.putText(frame, f"You: {user_move}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Computer: {comp_move}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if user_move in moves:
            if user_move == comp_move:
                winner = "Draw"
            elif (user_move == "Rock" and comp_move == "Scissors") or \
                 (user_move == "Paper" and comp_move == "Rock") or \
                 (user_move == "Scissors" and comp_move == "Paper"):
                winner = "You Win!"
            else:
                winner = "Computer Wins!"

            cv2.putText(frame, winner, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        # Reset round after 3 seconds
        if elapsed > 3:
            round_active = False
            winner = None
            comp_move = None

    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
