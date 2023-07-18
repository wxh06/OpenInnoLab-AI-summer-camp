import os.path

import tensorflow as tf
import tensorflow_datasets as tfds

HEIGHT = 224
WIDTH = 224
MODEL_PATH = "saved_model/resnet50v2"

DIR = os.path.dirname(__file__)

(ds_train, ds_test) = tfds.load(
    "CatsVsDogs",
    split=["train[:80%]", "train[80%:]"],
    shuffle_files=True,
    as_supervised=True,
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(tf.image.resize(image, (HEIGHT, WIDTH)), tf.float32) / 255.0, label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.batch(32)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(32)

base_model = tf.keras.applications.resnet_v2.ResNet50V2(
    include_top=False, weights="imagenet", input_shape=(HEIGHT, WIDTH, 3)
)
base_model.trainable = False
model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(ds_train, epochs=10, validation_data=ds_test)

model.save(os.path.join(DIR, MODEL_PATH))
