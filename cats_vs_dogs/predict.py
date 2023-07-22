from pathlib import PurePath

import pandas as pd
import tensorflow as tf

HEIGHT = 224
WIDTH = 224
MODEL_PATH = "saved_model/resnet50v2"
NUM_TEST_IMAGES = 30

BASE_DIR = PurePath(__file__).parent

model = tf.keras.models.load_model(BASE_DIR / MODEL_PATH)

files = [str(BASE_DIR / f"test_images/{i}.jpg") for i in range(NUM_TEST_IMAGES)]
images = tf.data.Dataset.from_tensor_slices(files).map(tf.io.read_file)
images = images.map(tf.image.decode_jpeg)
images = images.map(
    lambda image: tf.cast(tf.image.resize(image, (HEIGHT, WIDTH)), tf.float32) / 255.0
).batch(1)
result = model.predict(images)

df = pd.DataFrame(
    result.astype(int),
    index=[PurePath(pathname).name for pathname in files],
    columns=["prediction"],
)
df["pre_class"] = df.apply(lambda row: ["cat", "dog"][row["prediction"]], axis=1)
df.to_csv(BASE_DIR / "predictions.csv", index_label="filename")
