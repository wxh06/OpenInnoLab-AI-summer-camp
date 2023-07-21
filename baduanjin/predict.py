import os.path

import pandas as pd
import tensorflow as tf

HEIGHT = 224
WIDTH = 224
MODEL_PATH = "saved_model/DenseNet121"
NUM_TEST_IMAGES = 30

DIR = os.path.dirname(__file__)

model = tf.keras.models.load_model(os.path.join(DIR, MODEL_PATH))

files = [
    os.path.join(DIR, "test_data", f)
    for f in sorted(os.listdir(os.path.join(DIR, "test_data")))
]
images = tf.data.Dataset.from_tensor_slices(files).map(tf.io.read_file)
images = images.map(tf.image.decode_jpeg)
images = images.map(
    lambda image: tf.cast(tf.image.resize(image, (HEIGHT, WIDTH)), tf.float32) / 255.0
).batch(1)
result = tf.argmax(model.predict(images), axis=1)

df = pd.DataFrame(
    result,
    index=[os.path.split(pathname)[1] for pathname in files],
    columns=["prediction"],
)
df["pre_class"] = df.apply(
    lambda row: [f"pose{i + 1}" for i in range(8)][row["prediction"]], axis=1
)
df.to_csv(os.path.join(DIR, "predictions.csv"), index_label="filename")
