from hyperparameters import *
import numpy as np
import cv2
import os

import torch
from metric.iou import IoU
from models import SegNet


# FUNCTIONS
def load_and_resize_images(dataset_dir, batch_size, x_dim, y_dim, greyscale, seed=1):
    original_images = []
    names = os.listdir(dataset_dir)
    np.random.seed(seed)
    random_img_numbers = np.random.randint(low=0, high=len(names), size=batch_size)
    for i in random_img_numbers:
        if greyscale:
            input_image = cv2.imread(os.path.join(dataset_dir, names[i]), 0)
        else:
            input_image = cv2.imread(os.path.join(dataset_dir, names[i]))
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        input_image = cv2.resize(input_image, (x_dim, y_dim), interpolation=cv2.INTER_NEAREST)
        original_images.append(input_image)
    return original_images


def get_batch_numpy_image(numpy_image):
    if len(numpy_image.shape) == 3:
        y_dim, x_dim, channels = np.shape(numpy_image)
        input_image = numpy_image.transpose((2, 0, 1))
        batch_input_original_img = np.zeros(shape=(1, channels, y_dim, x_dim))
        batch_input_original_img[0, :, :, :] = input_image
    else:
        raise AssertionError("wrong image dimensions!")
    return batch_input_original_img


def prepare_label_batch(label_numpy, num_classes):
    y_dim, x_dim = np.shape(label_numpy)
    batch_label_one_hot = np.zeros(shape=(1, num_classes, y_dim, x_dim))
    # assuming the label has numbers from 0 to num_classes -1
    for i in range(num_classes):
        batch_label_one_hot[0, i, :, :] = (label_numpy == i)
    return batch_label_one_hot


def create_noisy_input(numpy_image, shape, noise_type, noise_parameter):
    if noise_type == "pepper":
        p = noise_parameter
        pepper_mask = np.random.choice([0, 1], size=shape, p=[p, 1-p])
        batch_modified_images = pepper_mask * numpy_image
        return batch_modified_images

    elif noise_type == "gaussian":
        noise_mean = 0
        noise_std = noise_parameter
        noise_matrices = np.random.normal(loc=noise_mean, scale=noise_std, size=shape)
        batch_modified_images = noise_matrices + numpy_image
        return batch_modified_images

    else:
        raise AssertionError("Noise type not available")


def create_label_copies(numpy_label, batch_size):
    batch_label_copies = np.repeat(numpy_label, batch_size, axis=0)
    return batch_label_copies


def load_trained_model(model_name, weights_path, num_classes):
    if model_name == 'segnet':
        model = SegNet(num_classes)
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise AssertionError('model not available')

    return model


def compute_accuracy(outputs, labels, num_classes):
    metric = IoU(num_classes, ignore_index=None)
    metric.reset()
    metric.add(outputs.detach(), labels.detach())
    (iou, miou) = metric.value()
    return miou
