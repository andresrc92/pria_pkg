import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
# import tensorflow_datasets as tfds

import build.pria.pria.tensorflow_model as tensorflow_model # custom module for reading .yaml file into np array

# read the labels

gt_file_dir = './black_cube/gt.yaml'
labels, _, image_count = tensorflow_model.load_labels_from_yaml(gt_file_dir)

# read the images

data_dir = './black_cube/imgs/*'

list_ds = tf.data.Dataset.list_files(data_dir, shuffle=False)
# list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

# Print the length of the dataset
print("Data size: ", tf.data.experimental.cardinality(list_ds).numpy())

# Print 5 dataset objects
for f in list_ds.take(5):
  print(f.numpy())

# Train / Val split
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  file_name = parts[-1] # convert bytes to string
  number_string = tf.strings.regex_replace(file_name, r"\.png$", "")
  number = tf.strings.to_number(number_string, out_type=tf.int32)

  # index = file_name.split(".")[0]        # split and get the number
  # index = int(index)

  return number

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_png(img, channels=3)
  # Resize the image to the desired size
  # return tf.image.resize(img, [img_height, img_width])
  return img

def process_path(file_path):
  print(file_path)
  label = get_label(file_path)
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

for image, label in train_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())