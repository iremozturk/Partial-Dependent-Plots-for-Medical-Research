import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
def create_cnn_model(images_dir):
    # Define the image dimensions
    image_shape = (64, 64, 1)

    # Load and preprocess the images
    image_data = []
    labels = []

    # Iterate over subdirectories (assuming each subdirectory represents a class)
    for class_name in os.listdir(images_dir):
        class_dir = os.path.join(images_dir, class_name)
        if os.path.isdir(class_dir):
            # Get all image files in the class directory
            class_images = [f for f in os.listdir(class_dir) if f.endswith('.jpg') or f.endswith('.png')]
            # Load and preprocess each image
            for img_name in class_images:
                img_path = os.path.join(class_dir, img_name)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(image_shape[0], image_shape[1]))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.image.rgb_to_grayscale(img_array)
                img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
                image_data.append(img_array)
                labels.append(class_name)

    # Convert lists to numpy arrays
    X_train = np.array(image_data)
    y_train = np.array(labels)

    # Define your CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32)  # Adjust epochs and batch_size as needed

    return model
