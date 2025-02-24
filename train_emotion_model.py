import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# 1. Define model-building function
def build_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten
        Flatten(),
        
        # Dense Layers
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # 2. Paths to your train and validation directories
    train_dir = 'data/train'
    val_dir = 'data/test'
    
    # 3. Image Data Generators
    # Rescale pixel values from [0, 255] to [0, 1]
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=10,      # random rotation
        width_shift_range=0.1,  # random horizontal shift
        height_shift_range=0.1, # random vertical shift
        horizontal_flip=True    # random flip
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # 4. Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        color_mode='grayscale', # FER2013 is grayscale
        class_mode='categorical',
        batch_size=64,
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=64,
        shuffle=False
    )
    
    # 5. Build the emotion model
    model = build_emotion_model()
    model.summary()
    
    # 6. Train the model
    # Adjust epochs as needed (e.g., 30, 50, or more, depending on your hardware).
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator
    )
    
    # 7. Save model weights
    # This file can then be loaded in your face detection + emotion script.
    model.save_weights('my_emotion_model_weights.h5')
    print("Model weights saved to 'my_emotion_model_weights.h5'")

if __name__ == '__main__':
    main()
