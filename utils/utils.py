from scipy.io import loadmat
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count


def load_wetland_data(time_folder, image_path, label_path, wetland_name, mode, value, save_colormap):
    # MATLAB variable names vary by dataset.
    image_mat = loadmat(image_path)
    if wetland_name in ["HHK", "ZY_YC"]:
        image = image_mat['Data']
    elif wetland_name == "Botswana":
        image = image_mat['Botswana']
    else:
        image_mark = wetland_name + "_image"
        image = image_mat[image_mark]

    label_mat = loadmat(label_path)
    if wetland_name in ["HHK", "ZY_YC"]:
        label = label_mat['DataClass']
    elif wetland_name == "Botswana":
        label = label_mat['Botswana_gt']
    else:
        label_mark = wetland_name + "_label"
        label = label_mat[label_mark]

    train_label = np.zeros(label.shape)
    test_label = np.copy(label)
    num_classes = int(np.max(label))
    if np.min(label) == 0:
        class_indices = range(0, num_classes + 1)
    else:
        class_indices = range(1, num_classes + 1)

    MIN_TRAIN_SAMPLES = 10
    for i in class_indices:
        if i == 0:
            continue

        num = np.sum(label == i)
        if num == 0:
            continue

        train_sample_num = 0
        if mode == "percentage":
            train_sample_num = math.floor(num * value)
        elif mode == 'fixed':
            train_sample_num = math.floor(value)

        if num > 1:
            final_train_num = max(train_sample_num, MIN_TRAIN_SAMPLES)
            final_train_num = min(final_train_num, num - 1)
        else:
            continue

        position = np.argwhere(label == i)
        sample_indices = random.sample(range(num), final_train_num)
        sample_position = position[sample_indices]

        for j in range(final_train_num):
            train_label[sample_position[j][0]][sample_position[j][1]] = label[sample_position[j][0]][
                sample_position[j][1]]
            test_label[sample_position[j][0]][sample_position[j][1]] = 0

    plt.imsave(time_folder + r"\\\all_label.png", label, cmap=save_colormap, dpi=300)
    plt.imsave(time_folder + r"\\\train_label.png", train_label, cmap=save_colormap, dpi=300)
    plt.imsave(time_folder + r"\\\test_label.png", test_label, cmap=save_colormap, dpi=300)

    print("Training set sample distribution:", flush=True)
    for i in range(1, num_classes + 1):
        count = np.sum(train_label == i)
        if count > 0:
            print(f"Class {i}: {count} samples", flush=True)

    print("\nTest set sample distribution:", flush=True)
    for i in range(1, num_classes + 1):
        count = np.sum(test_label == i)
        if count > 0:
            print(f"Class {i}: {count} samples", flush=True)

    return image, train_label, test_label
