import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# -------------------------------
# Step 1: Build the Emotion Detection Model
# -------------------------------
def build_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Constructs and compiles a CNN model for emotion detection.
    
    Args:
        input_shape (tuple): Dimensions of the input image (height, width, channels).
        num_classes (int): Number of emotion categories.
    
    Returns:
        model: A compiled Keras model.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flattening Layer
        Flatten(),
        
        # Fully Connected Layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------
# Step 2: Preprocess a Face Image for the Emotion Model
# -------------------------------
def preprocess_face(face_img):
    """
    Preprocesses the face image for the emotion detection model.
    
    Args:
        face_img (numpy array): Cropped face image in BGR or RGB format.
    
    Returns:
        processed_face (numpy array): Preprocessed image with shape (1, 48, 48, 1).
    """
    # Convert to grayscale (model expects a single channel)
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to 48x48 pixels (as expected by our model)
    resized_face = cv2.resize(gray_face, (48, 48))
    
    # Normalize pixel values to the [0, 1] range
    normalized_face = resized_face.astype("float32") / 255.0
    
    # Reshape to add channel and batch dimensions
    processed_face = np.expand_dims(normalized_face, axis=-1)  # Shape: (48, 48, 1)
    processed_face = np.expand_dims(processed_face, axis=0)      # Shape: (1, 48, 48, 1)
    
    return processed_face

# -------------------------------
# Step 3: Integrate Facial Detection with Emotion Prediction
# -------------------------------
def main():
    # Initialize the MTCNN face detector
    detector = MTCNN()
    
    # Build the emotion detection model
    emotion_model = build_emotion_model()
    emotion_model.load_weights('my_emotion_model_weights.h5')
    # Optionally, load pre-trained weights if available.
    # Uncomment and update the path below if you have weights saved.
    # emotion_model.load_weights('path_to_your_emotion_model_weights.h5')
    
    # Dictionary to map model output indices to human-readable emotion labels
    emotion_labels = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    # Load the input image (replace with your image file path)
    image_path = 'pic2.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image from", image_path)
        return
    
    # Convert image to RGB for proper processing and display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image using MTCNN
    results = detector.detect_faces(image_rgb)
    print(f"Detected {len(results)} face(s).")
    
    # Process each detected face
    for result in results:
        x, y, width, height = result['box']
        
        # Draw a bounding box around the face
        cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Crop the face region of interest (ROI)
        face_roi = image_rgb[y:y + height, x:x + width]
        
        # Preprocess the face ROI for the emotion model
        processed_face = preprocess_face(face_roi)
        
        # Predict the emotion
        prediction = emotion_model.predict(processed_face)
        predicted_class = np.argmax(prediction)
        emotion_text = emotion_labels.get(predicted_class, "Unknown")
        
        # Put the predicted emotion text above the face bounding box
        cv2.putText(image_rgb, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the final image with bounding boxes and predicted emotions
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Facial Detection and Emotion Prediction")
    plt.show()

# -------------------------------
# Run the Script
# -------------------------------
if __name__ == "__main__":
    main()
