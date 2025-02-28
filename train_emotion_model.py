import os
import tensorflow as tf
import tf2onnx
import onnx
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
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator
    )
    
    # 7. Save the model
    model.save('emotion_model.h5')
    print("Model saved as 'emotion_model.h5'")

    # 8. Convert to ONNX format
    onnx_model_path = "emotion_model.onnx"
    input_signature = [tf.TensorSpec([None, 48, 48, 1], tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
    onnx.save_model(onnx_model, onnx_model_path)
    
    print(f"Model converted to ONNX format and saved as '{onnx_model_path}'")

if __name__ == '__main__':
    main()
