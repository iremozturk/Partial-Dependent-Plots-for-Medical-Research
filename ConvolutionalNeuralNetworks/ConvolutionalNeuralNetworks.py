from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
from pdpbox import pdp
import pandas as pd
from PIL import Image


def pdp_to_img_array(pdp_isol, feature_name):
    fig, axes = pdp.pdp_plot(pdp_isol, feature_name)
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    img = Image.fromarray(img_array)
    img = img.resize((224, 224))
    img_array = np.array(img)

    return img_array


def generate_pdp_images_and_labels(models, X_df, features, num_classes):
    images = []
    labels = []
    aggregated_values = None
    for model_idx, (model_name, model) in enumerate(models.items()):
        for feature_name in features:
            pdp_isol = pdp.pdp_isolate(model=model, dataset=X_df, model_features=X_df.columns.tolist(),
                                       feature=feature_name)
            img_array = pdp_to_img_array(pdp_isol, feature_name)
            if img_array is not None:
                images.append(img_array)
                label = np.zeros(num_classes)
                label[model_idx] = 1
                labels.append(label)

                values = pdp_isol.pdp
                if aggregated_values is None:
                    aggregated_values = np.zeros_like(values)
                aggregated_values += values

    if images:
        image_batch = np.stack(images, axis=0)
        label_batch = np.stack(labels, axis=0)
        return image_batch, label_batch, aggregated_values
    return None, None, None


def train_cnn_to_evaluate_models(models, X_df, features, num_classes):
    pdp_images, labels, aggregated_values = generate_pdp_images_and_labels(models, X_df, features, num_classes)
    if pdp_images is None or len(pdp_images) == 0:
        return None

    pdp_images = pdp_images.astype('float32') / 255.0

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(pdp_images, labels, batch_size=4, epochs=10, validation_split=0.2)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    elif 'acc' in history.history:
        plt.plot(history.history['acc'], label='Training Accuracy')
        plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return model, aggregated_values


def plot_heatmap(aggregated_values):
    if aggregated_values.ndim == 1:
        aggregated_values = np.expand_dims(aggregated_values, axis=0)
    plt.imshow(aggregated_values, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Aggregated PDP Values Heatmap')
    plt.show()


def evaluate_models_with_cnn(models, X_df, features, num_classes):
    for model_name, model in models.items():
        for feature_name in features:
            pdp_isol = pdp.pdp_isolate(model=model, dataset=X_df, model_features=X_df.columns.tolist(),
                                       feature=feature_name)
            pdp.pdp_plot(pdp_isol, feature_name)
            plt.title(f'PDP for {model_name} - {feature_name}')
            plt.show()

    cnn_model, aggregated_values = train_cnn_to_evaluate_models(models, X_df, features, num_classes)
    if cnn_model is not None and aggregated_values is not None:
        plot_heatmap(aggregated_values)


def preprocess_for_cnn(images):
    images = images.astype('float32')
    images /= 255.0
    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)
    return images