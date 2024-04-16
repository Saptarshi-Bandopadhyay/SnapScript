import keras
from keras import layers
import tensorflow as tf

IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 10000
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512


image_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomContrast(0.3),
    ]
)


@keras.saving.register_keras_serializable()
def get_cnn_model():
    base_model = keras.applications.efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape(
        (-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model


@keras.saving.register_keras_serializable()
class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

    def get_config(self):
        base_config = super().get_config()
        config = {
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        }
        return {**base_config, **config}

    def call(self, inputs, training):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1


@keras.saving.register_keras_serializable()
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.add = layers.Add()

    def get_config(self):
        base_config = super().get_config()
        config = {
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        }
        return {**base_config, **config}

    def call(self, seq):
        seq = self.token_embeddings(seq)

        x = tf.range(tf.shape(seq)[1])
        x = x[tf.newaxis, :]
        x = self.position_embeddings(x)

        return self.add([seq, x])


@keras.saving.register_keras_serializable()
class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM,
            sequence_length=SEQ_LENGTH,
            vocab_size=VOCAB_SIZE,
        )
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")

        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def get_config(self):
        base_config = super().get_config()
        config = {
            "embed_dim": self.embed_dim,
            "ff_dim": self.ff_dim,
            "num_heads": self.num_heads,

        }
        return {**base_config, **config}

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training,
            use_causal_mask=True
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds


@keras.saving.register_keras_serializable()
class ImageCaptioningModel(keras.Model):
    def __init__(
        self,
        cnn_model,
        encoder,
        decoder,
        image_aug=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug

    def get_config(self):
        base_config = super().get_config()
        config = {
            "cnn_model": self.cnn_model,
            "encoder": self.encoder,
            "decoder": self.decoder,
            "image_aug": self.image_aug,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        # Note that you can also use [`keras.saving.deserialize_keras_object`](/api/models/model_saving_apis/serialization_utils#deserializekerasobject-function) here
        config["cnn_model"] = keras.saving.deserialize_keras_object(
            config["cnn_model"])
        config["encoder"] = keras.saving.deserialize_keras_object(
            config["encoder"])
        config["decoder"] = keras.saving.deserialize_keras_object(
            config["decoder"])
        config["image_aug"] = keras.saving.deserialize_keras_object(
            config["image_aug"])

    # Instantiate the ImageCaptioningModel with the remaining configuration
        return cls(**config)

    def call(self, inputs, training):
        img, caption = inputs
        if self.image_aug:
            img = self.image_aug(img)
        img_embed = self.cnn_model(img)
        encoder_out = self.encoder(img_embed, training=training)
        pred = self.decoder(caption, encoder_out, training=training)
        return pred
