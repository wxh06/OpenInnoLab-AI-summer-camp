from pathlib import PurePath

import pandas as pd
import tensorflow as tf

HEIGHT = 224
WIDTH = 224
MODEL_PATH = "saved_model/DenseNet121"
NUM_TEST_IMAGES = 30

BASE_DIR = PurePath(__file__).parent

model = tf.keras.models.load_model(BASE_DIR / MODEL_PATH)

files = tf.data.Dataset.list_files(str(BASE_DIR / "test_data/*.jpg"), shuffle=False)
images = files.map(tf.io.read_file)
images = images.map(tf.image.decode_jpeg)
images = images.map(
    lambda image: tf.cast(tf.image.resize(image, (HEIGHT, WIDTH)), tf.float32) / 255.0
).batch(1)
result = tf.argmax(model.predict(images), axis=1)

df = pd.DataFrame(
    result,
    index=[PurePath(pathname.numpy().decode()).name for pathname in files],
    columns=["prediction"],
)
df["pre_class"] = df.apply(
    lambda row: [f"pose{i + 1}" for i in range(8)][row["prediction"]], axis=1
)
df.to_csv(BASE_DIR / "predictions.csv", index_label="filename")
