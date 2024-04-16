import pathlib
import keras
import tensorflow as tf
import os
import numpy as np
import re


IMAGES_PATH = "Flicker8k_Dataset"
IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 10000
SEQ_LENGTH = 25
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE


path = pathlib.Path(".")
keras.utils.get_file(
    origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
    cache_dir='.',
    cache_subdir=path,
    extract=True)
keras.utils.get_file(
    origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
    cache_dir='.',
    cache_subdir=path,
    extract=True)

dataset = pathlib.Path(path, "Flickr8k.token.txt").read_text(
    encoding='utf-8').splitlines()

dataset = [line.split('\t') for line in dataset]

dataset = [[os.path.join(IMAGES_PATH, fname.split(
    '#')[0].strip()), caption] for (fname, caption) in dataset]

caption_mapping = {}
text_data = []
X_en_data = []
X_de_data = []
Y_data = []

for img_name, caption in dataset:
    if img_name.endswith("jpg"):
        X_de_data.append("<start> " + caption.strip().replace(".", ""))
        Y_data.append(caption.strip().replace(".", "") + " <end>")
        text_data.append(
            "<start> " + caption.strip().replace(".", "") + " <end>")
        X_en_data.append(img_name)

        if img_name in caption_mapping:
            caption_mapping[img_name].append(caption)
        else:
            caption_mapping[img_name] = [caption]

train_size = 0.8
shuffle = True
np.random.seed(42)

zipped = list(zip(X_en_data, X_de_data, Y_data))
np.random.shuffle(zipped)
X_en_data, X_de_data, Y_data = zip(*zipped)

train_size = int(len(X_en_data)*train_size)
X_train_en = list(X_en_data[:train_size])
X_train_de = list(X_de_data[:train_size])
Y_train = list(Y_data[:train_size])
X_valid_en = list(X_en_data[train_size:])
X_valid_de = list(X_de_data[train_size:])
Y_valid = list(Y_data[train_size:])

strip_chars = "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~"


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f'{re.escape(strip_chars)}', '')


vectorization = keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)

vectorization.adapt(text_data)

vocab = np.array(vectorization.get_vocabulary())
np.save('./artifacts/vocabulary.npy', vocab)


def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(img_cap, y_captions):
    img_path, x_captions = img_cap
    return ((decode_and_resize(img_path), vectorization(x_captions)), vectorization(y_captions))


def make_dataset(images, x_captions, y_captions):
    dataset = tf.data.Dataset.from_tensor_slices(
        ((images, x_captions), y_captions))
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset


train_dataset = make_dataset(X_train_en, X_train_de, Y_train)

valid_dataset = make_dataset(X_valid_en, X_valid_de, Y_valid)
