import os.path

import tensorflow as tf

HEIGHT = 224
WIDTH = 224
MODEL_PATH = "saved_model/DenseNet121"

DIR = os.path.dirname(__file__)

(ds_train, ds_test) = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DIR, "data"),
    label_mode="categorical",
    seed=0,
    validation_split=0.1,
    subset="both",
)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(tf.image.resize(image, (HEIGHT, WIDTH)), tf.float32) / 255.0, label


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.cache()

base_model = tf.keras.applications.DenseNet121(
    include_top=False, input_shape=(HEIGHT, WIDTH, 3)
)
base_model.trainable = False
model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation="softmax"),
    ]
)
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(ds_train, epochs=1, validation_data=ds_test)

model.save(os.path.join(DIR, MODEL_PATH))
