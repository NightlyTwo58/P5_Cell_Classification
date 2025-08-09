import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from keras.callbacks import ModelCheckpoint


# this for test data, needs to be manually downloaded and transferred into folder data/test
# https://www.kaggle.com/datasets/aleemaparakatta/cats-and-dogs-mini-dataset?resource=download

def download_and_extract_data(url, download_dir='data'):
    """
    Downloads and extracts a zip file from a URL into a specified directory.

    Args:
        url (str): The URL of the zip file.
        download_dir (str): The directory to download and extract the data to.

    Returns:
        str: The path to the extracted directory.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        print(f"Created directory: {download_dir}")

    path_to_zip = tf.keras.utils.get_file(
        origin=url,
        fname='cats_and_dogs.zip',
        extract=True,
        cache_dir=download_dir
    )

    extracted_dir = os.path.join(download_dir, 'datasets', 'cats_and_dogs_filtered')

    print(f"Data downloaded and extracted to: {extracted_dir}")
    return extracted_dir


def create_datasets(data_dir, image_size, batch_size, seed):
    """
    Creates training, validation, and test datasets from a directory.

    Args:
        data_dir (str): The base directory containing 'train' and 'validation' subdirectories.
        image_size (tuple): The target size of the images (height, width).
        batch_size (int): The number of samples per batch.
        seed (int): A seed for shuffling and transformations.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')

    # The validation set will be split into a new validation and test set
    cvd_val_test = tf.keras.utils.image_dataset_from_directory(
        directory=validation_dir,
        labels='inferred',
        label_mode='binary',
        color_mode='rgb',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_size = int(0.8 * len(cvd_val_test))
    cvd_val = cvd_val_test.take(val_size)
    cvd_test = cvd_val_test.skip(val_size)

    cvd_train = tf.keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels='inferred',
        label_mode='binary',
        color_mode='rgb',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    return cvd_train, cvd_val, cvd_test


def build_model(input_shape):
    """
    Builds and returns the CNN model with Batch Normalization.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
    ])

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=input_shape),
        data_augmentation,

        tf.keras.layers.Conv2D(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(3),

        # Second convolutional block with Batch Normalization
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(3),

        tf.keras.layers.Conv2D(128, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(256, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(2048),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(2048),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def predict_image(model, image_path, image_size):
    """
    Loads an image, preprocesses it, and makes a prediction.

    Args:
        model (tf.keras.Model): The trained Keras model.
        image_path (str): The path to the image file.
        image_size (tuple): The size the image should be resized to.
    """
    try:
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title("Input Image")
        plt.show()

        img = img.resize(image_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        prediction = model.predict(img_array)
        is_cat = prediction[0][0] < 0.5
        confidence = 1 - prediction[0][0] if is_cat else prediction[0][0]

        label = "cat" if is_cat else "dog"
        print(f"The model thinks the image is a {label} with {confidence * 100:.2f}% confidence.")

    except FileNotFoundError:
        print(f"Error: The image file at {image_path} was not found.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")


def plot_hist(history):
    print(history.history.keys())
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def main():
    """
    Main function to run the entire cats vs dogs classification workflow.
    """
    # URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    # print(tf.config.list_physical_devices('GPU'))
    # extracted_data_dir = download_and_extract_data(URL)
    # train_dir = os.path.join(extracted_data_dir, 'train')
    # validation_dir = os.path.join(extracted_data_dir, 'validation')

    IMAGE_SIZE = (250, 250)
    BATCH_SIZE = 4
    SEED = 44775

    train = 'data/train'
    validation = 'data/validation'

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=train,
        labels='inferred',
        label_mode='binary',
        color_mode='rgb',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory=validation,
        labels='inferred',
        label_mode='binary',
        color_mode='rgb',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    model = build_model(IMAGE_SIZE + (3,))
    model.summary()

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,  # Reduce the learning rate by a factor of 0.2
        patience=4,  # Wait 4 epochs with no improvement before reducing LR
        min_lr=0.000001,
        verbose=1
    )

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, "best_model.keras")
    checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )

    history = model.fit(
        train_ds,
        epochs=9,
        validation_data=validation_ds,
        callbacks=[checkpoint_callback, reduce_lr]
    )

    plot_hist(history)

    img_dir = "data/test/cats_set/cat.4001.jpg"
    predict_image(model, img_dir, IMAGE_SIZE)

def main_checkpoint(image_path):
    IMAGE_SIZE = (250, 250)
    # Load the best saved model
    best_model = tf.keras.models.load_model('checkpoints/best_model.keras')
    # Use best_model for evaluation or predictions
    predict_image(best_model, image_path, IMAGE_SIZE)

if __name__ == "__main__":
    main()
    # main_checkpoint("data/test/cats_set/cat.4004.jpg")
    # main_checkpoint("data/test/dogs_set/dog.4005.jpg")