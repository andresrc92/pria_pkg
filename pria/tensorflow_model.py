import os, time, json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import yaml

def load_dataset(dir):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(dir, labels=None)

    return dataset

def explore_dataset(ds):
    for image in ds.take(1):
        # Convert the image tensor to a NumPy array for display
        image_np = image.numpy()

        # Plot the image
        plt.figure(figsize=(4, 4))
        plt.imshow(image_np[433])
        # plt.title(f"Label: {label.numpy()}")
        plt.axis('off')
        plt.show()

def load_image(filepath, label):
    # print("AAAHHHHHHHHHHHHHHHHHHHHHHHHHH ", filepath[0])
    # filepath = os.path.join("./black_cube/imgs", filepath[0])
    # Read and decode the image
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [224, 224])  # Resize to desired size
    image = image / 255.0  # Normalize to [0, 1]
    return image, label

def display_image_cv(image):
    image_np = image.numpy()

    # Step 3: Display the image with OpenCV
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image - OpenCV", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_image_plt(image):
    image_np = image.numpy()
    plt.imshow(image_np)  # Matplotlib expects RGB format
    plt.axis("off")       # Hide axes for better visualization
    plt.title("Image - Matplotlib")
    plt.show()

def load_labels_from_yaml(dir):
    """
    load_labels_from_yaml(dir)

    parameters:
        - dir: path to .yaml file

    returns:
        - list of numpy arrays with [xt, yt, zt, q1, q2, q3, q0]
        - list of image files paths
    """

    with open(dir, "r") as file:
        gt = yaml.safe_load(file)

    return convert_labels_to_numpy_array(gt)
    # return gt

def convert_labels_to_numpy_array(gt):
    """
    load_labels_from_yaml(dir)

    parameters:
        - gt: dict object containing data from .yaml file

    returns:
        - list of numpy arrays with [xt, yt, zt, q1, q2, q3, q0]
        - list of image files paths
    """
    gt_np = []
    filepaths = []

    total_images = gt['initial_pose']['total_images']
    # print(total_images)

    for i in range(total_images):
        gt_t = np.array(gt[i]['translation'], dtype='f')
        gt_r = np.array(gt[i]['rotation'], dtype='f')
        gt_np.append(np.concatenate((gt_t, gt_r)))
        filepaths.append('./black_cube/imgs/{}.png'.format(i))

    return gt_np, filepaths, total_images
    



if __name__ == '__main__':
    gt, filepaths = load_labels_from_yaml("./black_cube/gt.yaml")

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, gt))
    dataset = dataset.batch(6, drop_remainder=True)
    dataset = dataset.map(lambda filepath, label: load_image(filepath, label))

    # dataset = load_dataset("./black_cube/imgs")
    # image, label = load_image("./black_cube/imgs/100.png",0)
    # display_image_plt(image)