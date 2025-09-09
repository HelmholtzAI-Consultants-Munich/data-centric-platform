import numpy as np
import random
from skimage import io, color, transform, util
import scipy.ndimage as ndi

# Set seed for reproducibility
seed_value = 2023
random.seed(seed_value)
np.random.seed(seed_value)

def assign_unique_colors(labels, colors):
    unique_labels = np.unique(labels)
    label_colors = {}
    for label in unique_labels:
        if label == 0:
            continue
        color_index = label % len(colors)
        label_colors[label] = colors[color_index]
    return label_colors

def custom_label2rgb(labels, colors=["red", "green", "blue"], bg_label=0, alpha=0.5):
    label_colors = assign_unique_colors(labels, colors)
    rgb_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=float)
    for label in np.unique(labels):
        mask = labels == label
        if label in label_colors:
            rgb = color.label2rgb(mask, colors=[label_colors[label]], bg_label=bg_label, alpha=alpha)
            rgb_image += rgb
    return rgb_image

def add_padding_for_rotation(image, angle):
    h, w = image.shape[:2]
    diagonal = int(np.ceil(np.sqrt(h**2 + w**2)))
    pad_h = (diagonal - h) // 2
    pad_w = (diagonal - w) // 2
    padded_image = util.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    rotated_image = transform.rotate(padded_image, angle, resize=False, order=1, mode='constant', cval=0)
    rotated_image = (rotated_image * 255).astype(np.uint8) if rotated_image.dtype != np.uint8 else rotated_image
    return rotated_image

def get_object_images(objects):
    object_images = []
    for obj in objects:
        img = io.imread(obj["path"], as_gray=True)
        img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        object_images.append(img)
    return object_images

def resize_image(image, size):
    resized = transform.resize(image, (size, size), order=1, preserve_range=True, anti_aliasing=True)
    return resized.astype(np.uint8)

def generate_dataset(num_samples, objects, canvas_size, max_object_counts=None, noise_intensity=None, max_rotation_angle=None):
    dataset_images = []
    dataset_masks = []

    object_images = get_object_images(objects)
    class_intensities = [(obj["intensity"][0], obj["intensity"][1]) for obj in objects]

    num_of_img_channels = 1
    if max_object_counts is None:
        max_object_counts = [10] * len(object_images)

    for _ in range(num_samples):
        canvas = np.zeros((canvas_size[0], canvas_size[1], num_of_img_channels), dtype=np.uint8)
        mask = np.zeros((canvas_size[0], canvas_size[1], len(object_images)), dtype=np.uint8)

        for object_index, object_img in enumerate(object_images):
            max_count = max_object_counts[object_index]
            object_count = random.randint(1, max_count)

            for _ in range(object_count):
                canvas_range = max(canvas_size)
                object_size = random.randint(canvas_range // 20, canvas_range // 5)
                object_img_resized = resize_image(object_img, object_size)

                intensity_mean = (class_intensities[object_index][1] - class_intensities[object_index][0]) / 2
                intensity_scale = (class_intensities[object_index][1] - intensity_mean) / 3
                class_intensity = np.clip(np.random.normal(loc=intensity_mean, scale=intensity_scale),
                                          class_intensities[object_index][0], class_intensities[object_index][1])
                object_img_resized = ((object_img_resized > 0).astype(np.uint8) * class_intensity * 255).astype(np.uint8)

                if max_rotation_angle is not None:
                    rotation_angle = random.uniform(-max_rotation_angle, max_rotation_angle)
                    object_img_transformed = add_padding_for_rotation(object_img_resized, rotation_angle)
                else:
                    object_img_transformed = object_img_resized

                object_size_x, object_size_y = object_img_transformed.shape
                object_mask = np.zeros((object_size_x, object_size_y), dtype=np.uint8)
                object_mask[object_img_transformed > 0] = object_index + 1
                object_img_transformed = np.expand_dims(object_img_transformed, axis=-1)

                x = random.randint(0, canvas_size[1] - object_size_x)
                y = random.randint(0, canvas_size[0] - object_size_y)

                intersecting_mask = mask[y:y + object_size_y, x:x + object_size_x].max(axis=-1)
                if (intersecting_mask > 0).any():
                    continue

                canvas[y:y + object_size_y, x:x + object_size_x] = object_img_transformed
                mask[y:y + object_size_y, x:x + object_size_x, object_index] = np.maximum(
                    mask[y:y + object_size_y, x:x + object_size_x, object_index], object_mask)

        if noise_intensity is not None:
            noise = np.random.normal(scale=noise_intensity, size=canvas.shape)
            noisy_canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)
            dataset_images.append(noisy_canvas.squeeze(2))
        else:
            dataset_images.append(canvas.squeeze(2))

        mask = mask.max(axis=-1)
        if len(mask.shape) == 2:
            mask = custom_label2rgb(mask, colors=["red", "green", "blue"])
            mask = ndi.label(mask)[0]
        else:
            for j in range(mask.shape[-1]):
                mask[..., j] = ndi.label(mask[..., j])[0]
            mask = mask.transpose(2, 0, 1)

        dataset_masks.append(mask)

    return dataset_images, dataset_masks

def get_synthetic_dataset(num_samples, canvas_size=(512, 512), max_object_counts=[15, 15, 15]):
    objects = [
        {"name": "triangle", "path": "test/shapes/triangle.png", "intensity": [0, 0.33]},
        {"name": "circle", "path": "test/shapes/circle.png", "intensity": [0.34, 0.66]},
        {"name": "square", "path": "test/shapes/square.png", "intensity": [0.67, 1.0]},
    ]

    images, masks = generate_dataset(num_samples, objects, canvas_size=canvas_size,
                                     max_object_counts=max_object_counts, noise_intensity=5,
                                     max_rotation_angle=30)
    return images, masks

'''
import numpy as np
import cv2
import random
import os
import sys

import skimage.color as color
import scipy.ndimage as ndi

# set seed for reproducibility
seed_value = 2023
random.seed(seed_value)
np.random.seed(seed_value)


def assign_unique_colors(labels, colors):
    """Assigns unique colors to each label in the given label array.

    :param labels: The array containing labels.
    :type labels: numpy.ndarray
    :param colors: The list of colors to assign to each label.
    :type colors: list
    :return: A dictionary containing the color assignment for each unique label.
    :rtype: dict
    """
    unique_labels = np.unique(labels)
    # Create a dictionary to store the color assignment for each label
    label_colors = {}

    # Iterate over the unique labels and assign colors
    for label in unique_labels:
        # Skip assigning colors if the label is 0 (background)
        if label == 0:
            continue

        # Check if the label is present in the labels
        if label in labels:
            # Assign the color to the label
            color_index = label % len(colors)
            label_colors[label] = colors[color_index]

    return label_colors


def custom_label2rgb(labels, colors=["red", "green", "blue"], bg_label=0, alpha=0.5):
    """
    Converts a label array to an RGB image using assigned colors for each label.

    :param labels: The array containing labels.
    :type labels: numpy.ndarray
    :param colors: The list of colors to assign to each label. Defaults to ['red', 'green', 'blue'].
    :type colors: list, optional
    :param bg_label: The label representing the background. Defaults to 0.
    :type bg_label: int, optional
    :param alpha: The transparency level of the colors. Defaults to 0.5.
    :type alpha: float, optional
    :return: The RGB image representing the labels with assigned colors.
    :rtype: numpy.ndarray
    """

    label_colors = assign_unique_colors(labels, colors)

    # Convert the labels to RGB using the assigned colors
    rgb_image = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=float)
    for label in np.unique(labels):
        mask = labels == label
        if label in label_colors:
            rgb = color.label2rgb(
                mask, colors=[label_colors[label]], bg_label=bg_label, alpha=alpha
            )
            rgb_image += rgb

    return rgb_image


def add_padding_for_rotation(image, angle):
    """
    Apply padding and rotation to an image.

    The purpose of this function is to ensure that the rotated image fits within its original dimensions by adding padding, preventing any parts of the image from being cropped.

    :param image: The input image.
    :type image: numpy.ndarray
    :param angle: The rotation angle in degrees.
    :type angle: float
    :return: The rotated and padded image.
    :rtype: numpy.ndarray
    """

    # Calculate rotated bounding box
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_theta = abs(rotation_matrix[0, 0])
    sin_theta = abs(rotation_matrix[0, 1])
    new_w = int((h * sin_theta) + (w * cos_theta))
    new_h = int((h * cos_theta) + (w * sin_theta))

    # Calculate padding amounts
    pad_w = (new_w - w) // 2
    pad_h = (new_h - h) // 2

    # Add padding to the image
    padded_image = cv2.copyMakeBorder(
        image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT
    )

    # Rotate the padded image
    center = (padded_image.shape[1] // 2, padded_image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        padded_image, rotation_matrix, (padded_image.shape[1], padded_image.shape[0])
    )

    return rotated_image


def get_object_images(objects):
    """
    Load object images from file paths.

    :param objects: A list of dictionaries containing information about the objects such as name, path, intensity
    :type objects: list[dict]
    :return: A list of object images loaded from the specified file paths.
    :rtype: list[numpy.ndarray]
    """

    object_images = []

    for obj in objects:
        img = cv2.imread(obj["path"])
        # img = cv2.resize(img, obj['size'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        object_images.append(img)

    return object_images


def generate_dataset(
    num_samples,
    objects,
    canvas_size,
    max_object_counts=None,
    noise_intensity=None,
    max_rotation_angle=None,
):
    """
    Generate a synthetic dataset with images and masks.

    :param num_samples: The number of samples to generate.
    :type num_samples: int
    :param objects: List of object descriptions.
    :type objects: list
    :param canvas_size: Size of the canvas to place objects on.
    :type canvas_size: tuple
    :param max_object_counts: Maximum object counts for each class. Default is None.
    :type max_object_counts: list, optional
    :param noise_intensity: Intensity of the additional noise to the image. Default is None.
    :type noise_intensity: float, optional
    :param max_rotation_angle: Maximum rotation angle in degrees. Default is None.
    :type max_rotation_angle: float, optional
    :return: A tuple containing the generated images and masks.
    :rtype: tuple
    """

    dataset_images = []
    dataset_masks = []

    object_images = get_object_images(objects)
    class_intensities = [(obj["intensity"][0], obj["intensity"][1]) for obj in objects]

    if len(object_images[0].shape) == 3:
        num_of_img_channels = object_images[0].shape[-1]
    else:
        num_of_img_channels = 1

    if max_object_counts is None:
        max_object_counts = [10] * len(object_images)

    for _ in range(num_samples):
        canvas = np.zeros(
            (canvas_size[0], canvas_size[1], num_of_img_channels), dtype=np.uint8
        )
        mask = np.zeros(
            (canvas_size[0], canvas_size[1], len(object_images)), dtype=np.uint8
        )

        for object_index, object_img in enumerate(object_images):

            max_count = max_object_counts[object_index]
            object_count = random.randint(1, max_count)

            for _ in range(object_count):

                canvas_range = max(canvas_size)
                object_size = random.randint(canvas_range // 20, canvas_range // 5)

                object_img_resized = cv2.resize(object_img, (object_size, object_size))
                # object_img_resized =  (object_img_resized>0).astype(np.uint8)*(255 - object_size)
                intensity_mean = (
                    class_intensities[object_index][1]
                    - class_intensities[object_index][0]
                ) / 2
                intensity_scale = (
                    class_intensities[object_index][1] - intensity_mean
                ) / 3
                class_intensity = np.random.normal(
                    loc=intensity_mean, scale=intensity_scale
                )
                class_intensity = np.clip(
                    class_intensity,
                    class_intensities[object_index][0],
                    class_intensities[object_index][1],
                )
                # class_intensity = random.randint(int(class_intensities[object_index][0]), int(class_intensities[object_index][1]))
                object_img_resized = (
                    (object_img_resized > 0).astype(np.uint8) * (class_intensity) * 255
                )

                if num_of_img_channels == 1:

                    if max_rotation_angle is not None:
                        # Randomly rotate the object image
                        rotation_angle = random.uniform(
                            -max_rotation_angle, max_rotation_angle
                        )
                        object_img_transformed = add_padding_for_rotation(
                            object_img_resized, rotation_angle
                        )
                    else:
                        object_img_transformed = object_img_resized

                    object_size_x, object_size_y = object_img_transformed.shape

                object_mask = np.zeros((object_size_x, object_size_y), dtype=np.uint8)

                if num_of_img_channels == 1:  # Grayscale image
                    object_mask[object_img_transformed > 0] = object_index + 1
                    # object_img_resized = np.expand_dims(object_img_resized, axis=-1)
                    object_img_transformed = np.expand_dims(
                        object_img_transformed, axis=-1
                    )
                else:  # Color image with alpha channel
                    object_mask[object_img_resized[:, :, -1] > 0] = object_index + 1

                x = random.randint(0, canvas_size[1] - object_size_x)
                y = random.randint(0, canvas_size[0] - object_size_y)

                intersecting_mask = mask[
                    y : y + object_size_y, x : x + object_size_x
                ].max(axis=-1)
                if (intersecting_mask > 0).any():
                    continue  # Skip if there is an intersection with objects from other classes

                assert (
                    mask[
                        y : y + object_size_y, x : x + object_size_x, object_index
                    ].shape
                    == object_mask.shape
                )

                canvas[y : y + object_size_y, x : x + object_size_x] = (
                    object_img_transformed
                )
                mask[y : y + object_size_y, x : x + object_size_x, object_index] = (
                    np.maximum(
                        mask[
                            y : y + object_size_y, x : x + object_size_x, object_index
                        ],
                        object_mask,
                    )
                )

        # Add noise to the canvas
        if noise_intensity is not None:

            if num_of_img_channels == 1:
                noise = np.random.normal(
                    scale=noise_intensity, size=(canvas_size[0], canvas_size[1], 1)
                )
                # noise = random_noise(canvas, mode='speckle', mean=noise_intensity)

            else:
                noise = np.random.normal(
                    scale=noise_intensity,
                    size=(canvas_size[0], canvas_size[1], num_of_img_channels),
                )
            noisy_canvas = canvas + noise.astype(np.uint8)

            dataset_images.append(noisy_canvas.squeeze(2))

        else:

            dataset_images.append(canvas.squeeze(2))

        mask = mask.max(axis=-1)
        if len(mask.shape) == 2:
            mask = custom_label2rgb(mask, colors=["red", "green", "blue"])
            mask = ndi.label(mask)[0]
        else:
            for j in range(mask.shape[-1]):
                mask[..., j] = ndi.label(mask[..., j])[0]
            mask = mask.transpose(2, 0, 1)

        dataset_masks.append(mask)

    return dataset_images, dataset_masks


def get_synthetic_dataset(
    num_samples, canvas_size=(512, 512), max_object_counts=[15, 15, 15]
):
    """Generates a synthetic dataset with images and masks.

    :param num_samples: The number of samples to generate.
    :type num_samples: int
    :param canvas_size: Size of the canvas to place objects on. Default is (512, 512).
    :type canvas_size: tuple, optional
    :param max_object_counts: Maximum object counts for each class. Default is [15, 15, 15].
    :type max_object_counts: list, optional
    :return: A tuple containing the generated images and masks.
    :rtype: tuple
    """
    objects = [
        {
            "name": "triangle",
            "path": "test/shapes/triangle.png",
            "intensity": [0, 0.33],
        },
        {"name": "circle", "path": "test/shapes/circle.png", "intensity": [0.34, 0.66]},
        {"name": "square", "path": "test/shapes/square.png", "intensity": [0.67, 1.0]},
    ]

    images, masks = generate_dataset(
        num_samples,
        objects,
        canvas_size=canvas_size,
        max_object_counts=max_object_counts,
        noise_intensity=5,
        max_rotation_angle=30,
    )
    return images, masks
'''