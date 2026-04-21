#%%
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K

IMAGE_SHAPE = [94, 24]
CHARS = "ABCDEFGHIJKLMNPQRSTUVWXYZ0123456789"  # I included, only O excluded (avoid confusion with 0)
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}
NUM_CLASS = len(CHARS) + 1

@keras.saving.register_keras_serializable()
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    
    # Calculate TRUE label length by counting elements that are not padded (-1)
    mask = tf.cast(tf.not_equal(y_true, -1), dtype="int64")
    label_length = tf.reduce_sum(mask, axis=1) # 1D tensor of lengths

    # Get the time dimension (width) of the predictions
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len,), dtype="int64") # 1D tensor

    # tf.nn.ctc_loss expects raw logits. Since your model outputs Softmax probabilities, 
    # we take the log to convert them back into pseudo-logits.
    logits = tf.math.log(y_pred + 1e-7)

    # Use Native TensorFlow CTC Loss
    loss = tf.nn.ctc_loss(
        labels=tf.cast(y_true, tf.int32),
        logits=logits,
        label_length=tf.cast(label_length, tf.int32),
        logit_length=tf.cast(input_length, tf.int32),
        logits_time_major=False,
        blank_index=-1 # Tells TF the blank token is the last class index
    )
    
    # Expand dims to match the (batch_size, 1) shape Keras expects for loss tracking
    return tf.expand_dims(loss, 1)

@keras.saving.register_keras_serializable()
class small_basic_block(keras.layers.Layer):
    def __init__(self, out_channels, name=None, **kwargs):
        super().__init__(**kwargs)
        out_div4 = int(out_channels / 4)
        self.main_layers = [
            keras.layers.SeparableConv2D(
                out_div4,
                (1, 1),
                padding="same",
                depthwise_initializer=keras.initializers.glorot_uniform(),
                pointwise_initializer=keras.initializers.glorot_uniform(),
                bias_initializer=keras.initializers.constant(),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.SeparableConv2D(
                out_div4,
                (3, 1),
                padding="same",
                depthwise_initializer=keras.initializers.glorot_uniform(),
                pointwise_initializer=keras.initializers.glorot_uniform(),
                bias_initializer=keras.initializers.constant(),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.SeparableConv2D(
                out_div4,
                (1, 3),
                padding="same",
                depthwise_initializer=keras.initializers.glorot_uniform(),
                pointwise_initializer=keras.initializers.glorot_uniform(),
                bias_initializer=keras.initializers.constant(),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.SeparableConv2D(
                out_channels,
                (1, 1),
                padding="same",
                depthwise_initializer=keras.initializers.glorot_uniform(),
                pointwise_initializer=keras.initializers.glorot_uniform(),
                bias_initializer=keras.initializers.constant(),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ]

    def call(self, input):
        x = input
        for layer in self.main_layers:
            x = layer(x)
        return x

@keras.saving.register_keras_serializable()
class global_context(keras.layers.Layer):
    def __init__(self, kernel_size, stride, **kwargs):
        super().__init__(**kwargs)
        self.ksize = kernel_size
        self.stride = stride

    def call(self, input):
        x = input
        avg_pool = keras.layers.AveragePooling2D(
            pool_size=self.ksize, strides=self.stride, padding="same"
        )(x)
        sq = keras.layers.Lambda(lambda x: tf.math.square(x))(avg_pool)
        sqm = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x))(sq)
        out = keras.layers.Lambda(lambda x: tf.math.divide(x[0], x[1]))([avg_pool, sqm])
        # out = keras.layers.Lambda(lambda x: K.l2_normalize(x,axis=1))(avg_pool)
        return out

    def get_config(self):
        return {
            "kernel_size": self.ksize,
            "stride": self.stride,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.saving.register_keras_serializable()
class LPRnet(keras.Model):
    def __init__(self, input_shape=(24, 94, 3), **kwargs):
        super(LPRnet, self).__init__(**kwargs)
        self.input_layer = keras.layers.Input(input_shape)
        self.cnn_layers = [
            keras.layers.SeparableConv2D(
                64,
                kernel_size=(3, 3),
                strides=1,
                padding="same",
                name="main_conv1",
                depthwise_initializer=keras.initializers.glorot_uniform(),
                pointwise_initializer=keras.initializers.glorot_uniform(),
                bias_initializer=keras.initializers.constant(),
            ),
            keras.layers.BatchNormalization(name="BN1"),
            keras.layers.ReLU(name="RELU1"),
            keras.layers.MaxPool2D(
                pool_size=(3, 3), strides=(1, 1), name="maxpool2d_1", padding="same"
            ),
            small_basic_block(128),
            keras.layers.MaxPool2D(
                pool_size=(3, 3), strides=(1, 2), name="maxpool2d_2", padding="same"
            ),
            small_basic_block(256),
            small_basic_block(256),
            keras.layers.MaxPool2D(
                pool_size=(3, 3), strides=(1, 2), name="maxpool2d_3", padding="same"
            ),
            keras.layers.Dropout(0.5),
            keras.layers.SeparableConv2D(
                256,
                (4, 1),
                strides=1,
                padding="same",
                name="main_conv2",
                depthwise_initializer=keras.initializers.glorot_uniform(),
                pointwise_initializer=keras.initializers.glorot_uniform(),
                bias_initializer=keras.initializers.constant(),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(0.5),
            keras.layers.SeparableConv2D(
                NUM_CLASS,
                (1, 13),
                padding="same",
                name="main_conv3",
                depthwise_initializer=keras.initializers.glorot_uniform(),
                pointwise_initializer=keras.initializers.glorot_uniform(),
                bias_initializer=keras.initializers.constant(),
            ),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ]
        self.out_layers = [
            keras.layers.SeparableConv2D(
                NUM_CLASS,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="same",
                name="conv_out",
                depthwise_initializer=keras.initializers.glorot_uniform(),
                pointwise_initializer=keras.initializers.glorot_uniform(),
                bias_initializer=keras.initializers.constant(),
            ),
        ]
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=False):
        x = inputs
        layer_outputs = []
        for layer in self.cnn_layers:
            x = layer(x)
            layer_outputs.append(x)
        scale1 = global_context((1, 4), (1, 4))(layer_outputs[0])
        scale2 = global_context((1, 4), (1, 4))(layer_outputs[4])
        scale3 = global_context((1, 2), (1, 2))(layer_outputs[6])
        scale5 = global_context((1, 2), (1, 2))(layer_outputs[7])
        sq = keras.layers.Lambda(lambda x: tf.math.square(x))(x)
        sqm = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x))(sq)
        scale4 = keras.layers.Lambda(lambda x: tf.math.divide(x[0], x[1]))([x, sqm])
        gc_concat = keras.layers.Lambda(
            lambda x: tf.concat([x[0], x[1], x[2], x[3], x[4]], 3)
        )([scale1, scale2, scale3, scale5, scale4])
        for layer in self.out_layers:
            gc_concat = layer(gc_concat)
        logits = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x[0], axis=1))(
            [gc_concat]
        )
        logits = keras.layers.Softmax()(logits)
        return logits


model = LPRnet()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=CTCLoss)
model.build((1, 24, 94, 3))
model.summary()
# %%