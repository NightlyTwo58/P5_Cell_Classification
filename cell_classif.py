import math

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
import xml.etree.ElementTree as ET
from skimage.draw import polygon
from PIL import Image

def xml_to_label_mask(xml_path, image_shape):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    label_mask = np.zeros(image_shape[:2], dtype=np.int32)

    regions = root.findall('.//Region')

    for i, region in enumerate(regions, start=1):
        vertices = region.find('Vertices')
        x_coords = []
        y_coords = []
        for vertex in vertices.findall('Vertex'):
            x_coords.append(float(vertex.get('X')))
            y_coords.append(float(vertex.get('Y')))

        rr, cc = polygon(y_coords, x_coords, shape=label_mask.shape)
        label_mask[rr, cc] = i

    return label_mask

def load_image_and_mask(image_path, xml_path, image_size):
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size

    img = img.resize(image_size)
    img_array = np.array(img) / 255.0

    label_mask = xml_to_label_mask(xml_path, (orig_size[1], orig_size[0]))
    mask_img = Image.fromarray(label_mask.astype(np.uint8))
    mask_img = mask_img.resize(image_size, resample=Image.NEAREST)
    mask_array = np.array(mask_img)
    binary_mask = (mask_array > 0).astype(np.float32)[..., np.newaxis]

    return img_array.astype(np.float32), binary_mask.astype(np.float32)

def create_dataset(images_dir, annotations_dir, image_size, batch_size, shuffle=True, augment=False):
    image_filenames = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])
    xml_filenames = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.xml')])
    print("Images:", image_filenames)
    print("XMLs:", xml_filenames)
    print("Number of images:", len(image_filenames))
    print("Number of XMLs:", len(xml_filenames))

    image_paths = [os.path.join(images_dir, f) for f in image_filenames]
    xml_paths = [os.path.join(annotations_dir, f) for f in xml_filenames]

    def generator():
        for img_path, xml_path in zip(image_paths, xml_paths):
            image, mask = load_image_and_mask(img_path, xml_path, image_size)
            yield image, mask

    output_signature = (
        tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(*image_size, 1), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=100).repeat()
    else:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if augment:
        # Add augmentation if you want, e.g. flipping, rotation
        def augment_fn(img, mask):
            img = tf.image.random_flip_left_right(img)
            mask = tf.image.random_flip_left_right(mask)
            return img, mask
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset, len(image_filenames)

def plot_training(history):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

def display_sample_prediction(model, dataset):
    for images, masks in dataset.take(1):
        preds = model.predict(images)
        plt.figure(figsize=(12,4))
        for i in range(min(3, images.shape[0])):
            plt.subplot(3,3,i*3+1)
            plt.imshow(images[i])
            plt.title("Image")
            plt.axis('off')
            plt.subplot(3,3,i*3+2)
            plt.imshow(masks[i,:,:,0], cmap='gray')
            plt.title("Mask")
            plt.axis('off')
            plt.subplot(3,3,i*3+3)
            plt.imshow(preds[i,:,:,0] > 0.5, cmap='gray')
            plt.title("Prediction")
            plt.axis('off')
        plt.show()

def build_unet(input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)

    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    bn = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    bn = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(bn)

    u3 = tf.keras.layers.UpSampling2D()(bn)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u3)
    c4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c4)

    u2 = tf.keras.layers.UpSampling2D()(c4)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c5)

    u1 = tf.keras.layers.UpSampling2D()(c5)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c6 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    c6 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c6)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c6)

    model = tf.keras.Model(inputs, outputs)

    def dice_coef(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    def dice_loss(y_true, y_pred):
        return 1.0 - dice_coef(y_true, y_pred)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=dice_loss,
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), dice_coef])
    return model

def visualize_dataset_sample(dataset, num_samples=3):
    for images, masks in dataset.take(1):
        for i in range(num_samples):
            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(images[i])
            plt.title("Image")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(masks[i, :, :, 0], cmap='gray')
            plt.title("Mask")
            plt.axis('off')

            plt.show(block=True)
            plt.close('all')

def main():
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 8
    EPOCHS = 30

    train_images = "data/train/images"
    train_masks = "data/train/annotations"
    val_images = "data/test/images"
    val_masks = "data/test/annotations"

    train_dataset, train_len = create_dataset(train_images, train_masks, IMAGE_SIZE, BATCH_SIZE, shuffle=True, augment=True)
    plt.close('all')
    val_dataset, val_len = create_dataset(val_images, val_masks, IMAGE_SIZE, BATCH_SIZE, shuffle=False, augment=False)
    plt.close('all')

    visualize_dataset_sample(train_dataset)
    visualize_dataset_sample(val_dataset)
    plt.close('all')

    model = build_unet(IMAGE_SIZE + (3,))
    model.summary()

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, "best_model.keras")
    checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=math.ceil(train_len / BATCH_SIZE),
        validation_steps=math.ceil(val_len / BATCH_SIZE),
        callbacks=[checkpoint_callback]
    )

    plot_training(history)
    display_sample_prediction(model, val_dataset)

if __name__ == "__main__":
    main()
