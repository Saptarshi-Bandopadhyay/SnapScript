import keras
import numpy as np
import tensorflow as tf
from make_dataset import decode_and_resize, vectorization

SEQ_LENGTH = 25

loaded_model = keras.saving.load_model(
    "./artifacts/caption_model.keras", compile=False)
vocab = np.load("./artifacts/vocabulary.npy")

index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1


def generate_caption(img_path):
    # Select a random image from the validation dataset

    # Read the image from the disk
    sample_img = decode_and_resize(sample_img)

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = loaded_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = loaded_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = loaded_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    print("Predicted Caption: ", decoded_caption)


# Check predictions for a few samples
generate_caption("./image.jpg")
