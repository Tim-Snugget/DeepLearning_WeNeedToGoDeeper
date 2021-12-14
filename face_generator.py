import abc
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import Conv2D, LeakyReLU, Flatten, Dropout, Dense, Reshape, Conv2DTranspose
from keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import shutil


def load_from_directory(filename, resize, batch_size=32,):
    dataset = keras.preprocessing.image_dataset_from_directory(
        filename, label_mode=None, image_size=resize, batch_size=batch_size
    )
    dataset = dataset.map(lambda x: (x - 127.0) / 127.0)  # make 255 values between 0 and 1
    # dataset is of type MapDataset or map of tensors
    return dataset


def load_from_npz(filename, resize, data_parm='arr_0', batch_size=32, shuffle_buffer_size=100):
    with np.load(filename) as data:
        dataset = data[data_parm]

    dataset = [np.array(Image.fromarray(data, 'RGB').resize(size=resize)).astype('float32') for data in dataset]
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.map(lambda x: (x - 127.0) / 127.0)

    dataset = dataset.shuffle(shuffle_buffer_size).batch(
        batch_size)  # shuffle (5arbat 5arabit (｡･∀･)ﾉﾞ（＾∀＾●）ﾉｼ) and batch them
    return dataset

class CustomModel:
    def __init__(self):
        self.model = self.build_model()

    @abc.abstractmethod
    def build_model(self):
        pass


class Discriminator(CustomModel):
    def build_model(self):
        model = Sequential(
            [
                # 64x64 input
                Conv2D(64, input_shape=(64, 64, 3), kernel_size=4, strides=2, padding="same", ),
                LeakyReLU(alpha=0.2),
                # to 32x32
                Conv2D(128, kernel_size=4, strides=2, padding="same"),
                LeakyReLU(alpha=0.2),
                # to 16x16
                Conv2D(128, kernel_size=4, strides=2, padding="same"),
                LeakyReLU(alpha=0.2),

                Flatten(),  # from matrix to vector
                Dropout(0.2),  # 20% of the network is at rest, hush they are sleeping (∪.∪ )...zzz
                Dense(1, activation="sigmoid"),
            ],
            name="discriminator",
        )
        return model


class Generator(CustomModel):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        super().__init__()

    def build_model(self):
        model = keras.Sequential(
            [
                Input(shape=(self.latent_dim,)),
                Dense(8 * 8 * 128),
                Reshape((8, 8, 128)),
                Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
                LeakyReLU(alpha=0.2),
                Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
                LeakyReLU(alpha=0.2),
                Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
                LeakyReLU(alpha=0.2),
                Conv2D(3, kernel_size=5, padding="same", activation="tanh"),
            ],
            name="generator",
        )
        return model


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("./generated_images/generated_img_%03d_%d.png" % (epoch, i))

if __name__ == "__main__":
        latent_dim = 128

        gan = GAN(discriminator=Discriminator().model, generator=Generator(latent_dim).model, latent_dim=latent_dim)
        gan.compile(
            d_optimizer=Adam(learning_rate=0.0001),
            g_optimizer=Adam(learning_rate=0.0001),
            loss_fn=BinaryCrossentropy(),
        )

        dataset = load_from_directory('Female',(64,64))

        epochs = 1 # 10
        ### We first wanted to run 60 epochs to train the AI the best we could
        # but even with a GPU (GTX 1050 2GB) one epoch took 1 hour to be processed (in the best case, otherwise longer)
        # We so chose to reduce epochs to 10, then one to have a result during the delivery

        gan.fit(dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)])

        gan.generator.save('gan_temp.h5')

        random_latent_vectors = tf.random.normal(shape=(10, 128))
        generated_images = gan.generator.predict(random_latent_vectors)
        generated_images *= 255


        for i in range(10):
            img = keras.preprocessing.image.array_to_img(generated_images[i])

            #plt.figure(figsize=(3,3)) 
            plt.axis("off")
            plt.imshow(img)
            break
        
        gan.generator.save('stable_20_epochs_tanh_celeba.h5')
        shutil.make_archive('20images', 'zip', './')        