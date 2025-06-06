import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="./tflite_model/model_signlang.tflite")
# interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract input shape
input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]

# Label list
# labels = ['A', 'B', 
#           'C', 'D', 
#           'E', 'F', 
#           'G', 'H', 
#           'I', 'J', 
#           'K', 'L', 
#           'M', 'N', 
#           'O', 'P',
#           'Q', 'R',
#           'S', 'T',
#           'U', 'V',
#           'W', 'X',
#           'Y', 'Z',
#           'delete', 'nothing', 'space']

labels = ['A', 'B', 'blank', 
          'C', 'D', 
          'E', 'F', 
          'G', 'H', 
          'I', 'J', 
          'K', 'L', 
          'M', 'N', 
          'O', 'P',
          'Q', 'R',
          'S', 'T',
          'U', 'V',
          'W', 'X',
          'Y', 'Z',]
# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Open camera
cap = cv2.VideoCapture(0)
pred_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    h, w, _ = frame.shape

    # Convert frame to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = "No hand detected"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            # Get bounding box for the hand
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Add some margin around the hand
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # Crop the hand image
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Skip if the cropped hand image is empty
            if hand_img.size == 0 or hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
                continue

            # Resize the cropped hand image to the model input size
            hand_img = cv2.resize(hand_img, (width, height))
            img_input = hand_img.astype(np.float32) / 255.0  # Normalize image
            img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension

            # Run inference with TFLite model
            interpreter.set_tensor(input_details[0]['index'], img_input)
            interpreter.invoke()

            # Get the output and prediction
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.argmax(output_data)
            confidence = np.max(output_data)

            if confidence > 0.7:
                pred_buffer.append(prediction)
                if len(pred_buffer) > 5:
                    pred_buffer.pop(0)

                final_pred = max(set(pred_buffer), key=pred_buffer.count)
                label = f"{labels[final_pred]} ({confidence:.2f})"
            else:
                label = "Unknown"

            # Draw the bounding box around the hand and the landmarks
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the result
    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
