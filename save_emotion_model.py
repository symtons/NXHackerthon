import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define your model architecture (same as before)
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

# Build and load your model
emotion_model = build_emotion_model()
emotion_model.load_weights('my_emotion_model_weights.h5')

# Save the model as a TensorFlow SavedModel
tf.saved_model.save(emotion_model, "saved_emotion_model")
print("Saved model to 'saved_emotion_model' directory")
