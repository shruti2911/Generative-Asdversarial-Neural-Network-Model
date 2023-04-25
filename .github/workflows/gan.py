import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Loading the MNIST dataset
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalizing the images to [-1, 1] range
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train = (x_train - 127.5) / 127.5

# Defining the generator model
generator = tf.keras.Sequential()
generator.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(layers.Reshape((7, 7, 256)))
generator.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU())

generator.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

# Defining the discriminator model
discriminator = tf.keras.Sequential()
discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.Dropout(0.3))

discriminator.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
discriminator.add(layers.LeakyReLU())
discriminator.add(layers.Dropout(0.3))

discriminator.add(layers.Flatten())
discriminator.add(layers.Dense(1))

# Defining the loss/error functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Defining the generator loss/error function
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Defining the discriminator loss/error function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Defining the optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Defining the batch size and number of epochs
batch_size = 128
epochs = 50

# Defining the training loop
@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Defining a function to generate and save the generated images
def generate_and_save_images(model, epoch, test_input):
    # Generating images from the model
    predictions = model(test_input, training=False)

    # Rescaling the images to [0, 1] range
    predictions = (predictions + 1) / 2.0

    # Creating a figure to plot the images
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

# Saving the figure
plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
plt.show()

# Defining the training function
def train(dataset, epochs):
    for epoch in range(epochs):
        # Looping through the batches in the dataset
        for batch in dataset:
            # Generating random noise for the generator input
            noise = tf.random.normal([batch_size, 100])

            # Recording operations for automatic differentiation
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # Generate images from the generator
                generated_images = generator(noise, training=True)

                # Getting the discriminator's output for the real and generated images
                real_output = discriminator(batch, training=True)
                fake_output = discriminator(generated_images, training=True)

                # Calculating the generator and discriminator losses
                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            # Calculating the gradients of the generator and discriminator with respect to their losses
            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            # Applying the gradients to the optimizer
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        # Generating and saving images after every 10 epochs
        if epoch % 10 == 0:
            generate_and_save_images(generator, epoch, test_input)

        # Printing the losses after every epoch
        print("Epoch {}: Generator loss: {}, Discriminator loss: {}".format(epoch+1, gen_loss, disc_loss))

# Generating a fixed noise vector for the test images
test_input = tf.random.normal([16, 100])

# Creating a dataset from the training images and shuffling it
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(len(x_train)).batch(batch_size)

# Training the GAN
train(train_dataset, epochs)
