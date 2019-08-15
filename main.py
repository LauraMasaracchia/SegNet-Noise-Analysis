""" script that runs some analyses on the trained model(s).

The script can do some sensitivity analyses on the trained model testing accuracy of the network by varying amount of
pepper noise or gaussian noise on the input images. The predictions on the images are stored so that the analysis can
be done also pixel by pixel. Eventually, the aim is to identify regions of a picture where the network is more accurate,
and regions where we need it to be, customizing thresholds for acceptability.

Also, the script is used as a basis to analyse the response of the network to brightness, sharpness and other
characteristics of the pictures in the dataset. These are already computed features stored in our dataset.

It will save stuff in a .npy file.
content of the saved dict:
'original_images' : list of numpy arrays, original images used to do the analysis
'original_labels' : list of numpy arrays, original labels to be predicted
'original_predictions' : numpy array, predictions of the trained network on the unmodified image
'reference_accuracy' : float value, the average accuracy that the network has on these images
'p=<noise_value>' : dictionary. containing:
                    'image<nbr>' : dictionary. containing:
                                   'noise_std_pred' : numpy array, standard deviation (pixel by pixel) of the noisy predictions of the image
                                   'noise_mean_pred' : numpy array, mean value (pixel by pixel) of the noisy predictions of the image
                                   'accuracy' : float value, average accuracy on the noisy predictions of the image
"""
import numpy as np
import os
from utils import *
import torch
import matplotlib.pyplot as plt


if NOISE_TYPE == "pepper":
    percent_p_noise = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    noise_modul = percent_p_noise
    title = "Accuracy as a function of percentage of pepper noise"
    xlabel = 'p'
if NOISE_TYPE == "gaussian":
    noise_std = [0.05, 0.1, 0.2, 0.5, 0.8, 1]
    noise_modul = noise_std
    title = "Accuracy as a function of standard deviation of gaussian noise"
    xlabel = 'std'


#####################################################################################################################
# MAIN

# STEP 1 : load images
saving_dict = {}
original_images = load_and_resize_images(dataset_dir=IMAGES_FOLDER_PATH,
                                         batch_size=NBR_IMAGES_TO_COMPARE,
                                         greyscale=False,
                                         x_dim=INPUT_SHAPE[0], y_dim=INPUT_SHAPE[1])

original_labels = load_and_resize_images(dataset_dir=LABELS_FOLDER_PATH,
                                         batch_size=NBR_IMAGES_TO_COMPARE,
                                         greyscale=True,
                                         x_dim=INPUT_SHAPE[0], y_dim=INPUT_SHAPE[1])

saving_dict['original_images'] = original_images
saving_dict['original_labels'] = original_labels


# STEP 2 : load trained model
model = load_trained_model(model_name=MODEL_NAME, weights_path=WEIGHTS_PATH, num_classes=NBR_CLASSES)
model.eval()


# STEP 3 : get the reference accuracy and predictions
reference_accuracy = np.zeros(shape=NBR_IMAGES_TO_COMPARE)
reference_predictions = np.zeros(shape=(NBR_IMAGES_TO_COMPARE, NBR_CLASSES, INPUT_SHAPE[1], INPUT_SHAPE[0]))
for i in range(NBR_IMAGES_TO_COMPARE):
    batch_input_original_numpy = get_batch_numpy_image(original_images[i])
    batch_label_original_numpy = prepare_label_batch((original_labels[i] / 255), NBR_CLASSES)

    batch_input_original_tensor = torch.from_numpy(batch_input_original_numpy).float()
    batch_label_original_tensor = torch.from_numpy(batch_label_original_numpy).float()

    with torch.no_grad():
        preds = model(batch_input_original_tensor)

    # accuracy
    reference_accuracy[i] = compute_accuracy(preds, batch_label_original_tensor, NBR_CLASSES)
    reference_predictions[i] = preds.numpy()

accuracy = [np.mean(reference_accuracy)]
saving_dict['reference_accuracy'] = np.mean(reference_accuracy)
saving_dict['original_pred'] = reference_predictions

# STEP 4 : run the analysis: for every noise parameter for every image

for p in noise_modul:
    key = 'p='+str(p)
    saving_dict[key] = {}
    noise_accuracy = np.zeros(shape=NBR_IMAGES_TO_COMPARE)
    for i in range(NBR_IMAGES_TO_COMPARE):
        name = 'image'+str(i)
        saving_dict[key][name] = {}
        batch_input_original_numpy = get_batch_numpy_image(original_images[i])
        batch_label_original_numpy = prepare_label_batch((original_labels[i] / 255), NBR_CLASSES)

        batch_modified_images = create_noisy_input(numpy_image=batch_input_original_numpy,
                                                   shape=(NOISE_BATCH, INPUT_SHAPE[2], INPUT_SHAPE[1], INPUT_SHAPE[0]),
                                                   noise_type=NOISE_TYPE,
                                                   noise_parameter=p)
        batch_multiple_label = create_label_copies(numpy_label=batch_label_original_numpy,
                                                   batch_size=NOISE_BATCH)

        batch_modified_images_tensor = torch.from_numpy(batch_modified_images).float()
        batch_multiple_label_tensor = torch.from_numpy(batch_multiple_label).float()

        with torch.no_grad():
            noise_prediction = model(batch_modified_images_tensor)
        # accuracy
        noise_accuracy[i] = compute_accuracy(noise_prediction, batch_multiple_label_tensor, NBR_CLASSES)
        noise_prediction = noise_prediction.numpy()

        # analysis
        # save predictions, so we can run more analyses on them later.
        # for each image, average over a batch of noisy images with the same percentage of noise, pixel by pixel
        pred_noise_mean = np.mean(noise_prediction, axis=0)
        pred_noise_std = np.std(noise_prediction, axis=0)  # std on the predicted noisy images per pixel
        average_on_class_std_pred = np.mean(pred_noise_std, axis=0)

        saving_dict[key][name]['noise_std_pred'] = pred_noise_std
        saving_dict[key][name]['noise_mean_pred'] = pred_noise_mean
        saving_dict[key][name]['accuracy'] = noise_accuracy[i]

    accuracy.append(np.mean(noise_accuracy))

# save
saving_name = 'segnet_noise_' + NOISE_TYPE + '.npy'
np.save(os.path.join(SAVING_DIRECTORY, saving_name), saving_dict)


# plot
noise_x = [0] + noise_modul
ref_y = [np.mean(reference_accuracy)] * len(noise_x)
plt.plot(noise_x, accuracy, '-', noise_x, ref_y, '--')
plt.xlabel(xlabel)
plt.ylabel('miou')
plt.title(title)
plt.show()

