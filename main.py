# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:37:48 2020

@author: Arham Jain
"""
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import model
import losses
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

RESHAPE = (256,256)

def is_an_image_file(filename):
    IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']
    for ext in IMAGE_EXTENSIONS:
        if ext in filename:
            return True
    return False


def list_image_files(directory):
    files = sorted(os.listdir(directory))
    return [os.path.join(directory, f) for f in files if is_an_image_file(f)]


def load_image(path):
    img = Image.open(path)
    return img


def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img


def deprocess_image(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')


def save_image(np_arr, path):
    img = np_arr * 127.5 + 127.5
    im = Image.fromarray(img)
    im.save(path)


def load_images(path, n_images):
    if n_images < 0:
        n_images = float("inf")
    A_paths, B_paths = os.path.join(path, 'A'), os.path.join(path, 'B')
    all_A_paths, all_B_paths = list_image_files(A_paths), list_image_files(B_paths)
    images_A, images_B = [], []
    images_A_paths, images_B_paths = [], []
    for path_A, path_B in zip(all_A_paths, all_B_paths):
        img_A, img_B = load_image(path_A), load_image(path_B)
        images_A.append(preprocess_image(img_A))
        images_B.append(preprocess_image(img_B))
        images_A_paths.append(path_A)
        images_B_paths.append(path_B)
        if len(images_A) > n_images - 1: break

    return {
        'A': np.array(images_A),
        'A_paths': np.array(images_A_paths),
        'B': np.array(images_B),
        'B_paths': np.array(images_B_paths)
    }

def write_log(callback, names, logs, batch_no):
    """
    Util to write callback for Keras training
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
        
data = load_images('./images/train', n_images)
y_train, x_train = data['B'], data['A']

# Initialize models
g = model.generator_model()
d = model.discriminator_model()
d_on_g = model.generator_containing_discriminator_multiple_outputs(g, d)

# Initialize optimizers
g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# Compile models
d.trainable = True
d.compile(optimizer=d_opt, loss=losses.wasserstein_loss)
d.trainable = False
loss = [losses.perceptual_loss,losses.l2_loss]
loss_weights = [100, 1]
d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
d.trainable = True
for epoch in range(epoch_num):
  print('epoch: {}/{}'.format(epoch, epoch_num))
  print('batches: {}'.format(x_train.shape[0] / batch_size))

  # Randomize images into batches
  permutated_indexes = np.random.permutation(x_train.shape[0])

  for index in range(int(x_train.shape[0] / batch_size)):
      batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
      image_blur_batch = x_train[batch_indexes]
      image_full_batch = y_train[batch_indexes]

for epoch in range(epoch_num):
  for index in range(batches):
    # [Batch Preparation]

    # Generate fake inputs
    generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)
    
    # Train multiple times discriminator on real and fake inputs
    for _ in range(critic_updates):
        d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
        d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

    d.trainable = False
    # Train generator only on discriminator's decision and generated images
    d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])

    d.trainable = True
    
