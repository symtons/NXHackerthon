import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# -------------------------------
# 1. Build the Emotion Detection Model
# -------------------------------
def build_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# 2. Preprocess a Face Image for Emotion Detection
# -------------------------------
def preprocess_face_for_emotion(face_img):
    # Convert to grayscale for the emotion model
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # Resize to 48x48
    resized_face = cv2.resize(gray_face, (48, 48))
    # Normalize to [0,1]
    normalized_face = resized_face.astype("float32") / 255.0
    # Add batch and channel dimensions -> (1, 48, 48, 1)
    processed_face = np.expand_dims(normalized_face, axis=-1)
    processed_face = np.expand_dims(processed_face, axis=0)
    return processed_face

# -------------------------------
# 3. Live Detection (Face + Emotion + Gender + Age)
# -------------------------------
def run_live_detection():
    # --- 3A. Initialize MTCNN for Face Detection ---
    detector = MTCNN()
    
    # --- 3B. Emotion Model ---
    emotion_model = build_emotion_model()
    # Load your trained weights
    emotion_model.load_weights('my_emotion_model_weights.h5')
    emotion_labels = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    # --- 3C. Gender Detection Model (OpenCV DNN) ---
    genderProto = "deploy_gender.prototxt"
    genderModel = "gender_net.caffemodel"
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    genderList = ['Male', 'Female']
    
    # --- 3D. Age Detection Model (OpenCV DNN) ---
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    # Typical age buckets (from the public age models)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    
    # Mean values used by the gender/age models
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    # --- 3E. Start Video Capture ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Convert to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect faces
        faces = detector.detect_faces(frame_rgb)
        num_faces = len(faces)
        
        for face in faces:
            x, y, w, h = face['box']
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Safely handle zero-size ROIs
            if w <= 0 or h <= 0:
                continue
            
            # ---------------------------
            # (1) Emotion Detection
            # ---------------------------
            # Use the BGR frame for grayscale conversion, or convert separately
            face_roi_bgr = frame[y:y+h, x:x+w]
            if face_roi_bgr.size == 0:
                continue
            
            # Preprocess for emotion
            processed_face = preprocess_face_for_emotion(face_roi_bgr)
            emotion_prediction = emotion_model.predict(processed_face)
            predicted_emotion = emotion_labels.get(np.argmax(emotion_prediction), "Unknown")
            
            # Put emotion label on the frame
            cv2.putText(frame, predicted_emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # ---------------------------
            # (2) Gender Detection
            # ---------------------------
            # The gender model expects a 227x227 BGR image with mean subtraction
            blob_gender = cv2.dnn.blobFromImage(face_roi_bgr, 1.0, (227, 227),
                                                MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob_gender)
            gender_preds = genderNet.forward()
            gender = genderList[gender_preds[0].argmax()]
            
            # Put gender label
            cv2.putText(frame, "Gender: " + gender, (x, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # ---------------------------
            # (3) Age Detection
            # ---------------------------
            blob_age = cv2.dnn.blobFromImage(face_roi_bgr, 1.0, (227, 227),
                                             MODEL_MEAN_VALUES, swapRB=False)
            ageNet.setInput(blob_age)
            age_preds = ageNet.forward()
            age = ageList[age_preds[0].argmax()]
            
            # Put age label
            cv2.putText(frame, "Age: " + age, (x, y - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # People count
        cv2.putText(frame, f'People Count: {num_faces}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('Live Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_detection()
