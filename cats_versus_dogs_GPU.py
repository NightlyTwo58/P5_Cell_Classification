import tensorflow as tf
from tensorflow.python.client import device_lib
import cats_versus_dogs as central
import os

print(device_lib.list_local_devices())

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
    BATCH_SIZE = 3
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
    with tf.device('/GPU:0'):
        model = central.build_model(IMAGE_SIZE + (3,))
        model.summary()

        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        filepath = os.path.join(checkpoint_dir, "best_model.keras")

        checkpoint_callback = central.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=4,
            min_lr=0.000001,
            verbose=1
        )

        history = model.fit(
            train_ds,
            epochs=10,
            validation_data=validation_ds,
            callbacks=[checkpoint_callback, reduce_lr]
        )

    central.plot_hist(history)

    img_dir = "data/test/cats_set/cat.4001.jpg"
    central.predict_image(model, img_dir, IMAGE_SIZE)

def main_checkpoint():
    IMAGE_SIZE = (250, 250)
    # Load the best saved model
    best_model = tf.keras.models.load_model('checkpoints/best_model.keras')
    # Use best_model for evaluation or predictions
    central.predict_image(best_model, "data/test/cats_set/cat.4001.jpg", IMAGE_SIZE)

if __name__ == "__main__":
    main()