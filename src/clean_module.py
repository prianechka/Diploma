import numpy as np
import output as ot
import cv2
import tensorflow as tf
import metrics as ms
from PyQt5.QtWidgets import QMessageBox

WHITE = np.array([255, 255, 255])
BLACK = np.array([0, 0, 0])
COLOR = WHITE
EPS = 220

WHITE_MODEL = 'models/white.h5'
BLACK_MODEL = 'models/black.h5'
MEDIAN_WHITE_MODEL = 'models/median_white.h5'
MEDIAN_BLACK_MODEL = 'models/black_median.h5'

RIDNET_WHITE_MODEL = 'models/ridnet_white.h5'
RIDNET_BLACK_MODEL = 'models/ridnet_black.h5'


def clean_image(imagefile, flag):
    original_image = cv2.imread(imagefile)
    noised_image = cv2.imread(imagefile)
    src_patches = np.array(get_all_patches(noised_image)).astype('float32') / 255
    src_patches = tf.clip_by_value(src_patches, clip_value_min=0., clip_value_max=1.)
    final_noised_image = create_image_from_patches(src_patches, noised_image.shape)

    image = median_filter(original_image, flag)

    patches = np.array(image).astype('float32') / 255
    patches = tf.clip_by_value(patches, clip_value_min=0., clip_value_max=1.)

    ridnet_new = tf.keras.models.load_model(WHITE_MODEL)
    if flag == 2:
        ridnet_new = tf.keras.models.load_model(BLACK_MODEL)

    denoised_image = predict_fun(ridnet_new, patches, original_image)

    ot.plot_clean_image(final_noised_image, denoised_image)

    return denoised_image, True


def check_metrics(imagefile, original_image, flag):
    noised_image = cv2.imread(imagefile)
    src_patches = np.array(get_all_patches(noised_image)).astype('float32') / 255
    src_patches = tf.clip_by_value(src_patches, clip_value_min=0., clip_value_max=1.)
    final_noised_image = create_image_from_patches(src_patches, noised_image.shape)

    original_image = cv2.imread(original_image)
    image = median_filter(noised_image, flag)

    patches = np.array(image).astype('float32') / 255
    patches = tf.clip_by_value(patches, clip_value_min=0., clip_value_max=1.)

    ridnet_new = tf.keras.models.load_model(WHITE_MODEL)
    if flag == 2:
        ridnet_new = tf.keras.models.load_model(BLACK_MODEL)

    denoised_image = predict_fun(ridnet_new, patches, noised_image)

    src_patches = np.array(get_all_patches(original_image)).astype('float32') / 255
    src_patches = tf.clip_by_value(src_patches, clip_value_min=0., clip_value_max=1.)
    original_image = create_image_from_patches(src_patches, original_image.shape)

    PNSR = ms.PSNR(denoised_image, original_image)
    msgBox = QMessageBox()
    msgBox.setIcon(QMessageBox.Information)
    msgBox.setWindowTitle("Метрика PSNR")
    msgBox.setText(str(round(PNSR, 3)))
    msgBox.exec()

    ot.plot_original_image(final_noised_image, denoised_image, original_image)

    return denoised_image, True


def predict_fun(model, patches_noisy, gt):
    denoised_patches = model.predict(patches_noisy)
    denoised_patches = tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

    denoised_image = create_image_from_patches(denoised_patches, gt.shape)

    return denoised_image


def median_filter(image, flag=1):
    patches = get_all_patches(image)
    i = 0
    for patch in patches:
        patches[i] = apply_median_filter(patches[i], i, flag)
        i += 1
    return patches


def apply_median_filter(patch, i, flag):
    if count_noise(patch, i):
        after_median_patch = np.array([cv2.medianBlur(patch, 3)])
        tf_type_patch = after_median_patch.astype('float32') / 255
        tf_type_patch = tf.clip_by_value(tf_type_patch, clip_value_min=0., clip_value_max=1.)

        ridnet_median = tf.keras.models.load_model(MEDIAN_WHITE_MODEL)
        if flag == 2:
            ridnet_median = tf.keras.models.load_model(MEDIAN_BLACK_MODEL)

        corrected_patch = ridnet_median.predict(tf_type_patch)
        corrected_patch = (corrected_patch * 255).astype('int')
        return create_image_from_patches(corrected_patch, patch.shape)

    return patch


def count_noise(patch, i):
    counter = 0
    for row in patch:
        for col in row:
            if np.array_equal(COLOR, WHITE) and col[0] > 230 and col[1] > 230 and col[2] > 230:
                counter += 1
            elif np.array_equal(COLOR, BLACK) and col[0] < 20 and col[1] < 20 and col[2] < 20:
                counter += 1
    return counter > EPS


def create_image_from_patches(patches, image_shape):
    image = np.zeros(image_shape)
    patch_size = patches.shape[1]
    p = 0
    for i in range(0, image.shape[0] - patch_size + 1, int(patch_size / 1)):
        for j in range(0, image.shape[1] - patch_size + 1, int(patch_size / 1)):
            image[i:i + patch_size, j:j + patch_size] = patches[p]  # Assigning values of pixels from patches to image
            p += 1
    return np.array(image)


def get_all_patches(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    crop_sizes = [1]
    patch_size = 40
    patches = []
    for crop_size in crop_sizes:  # We will crop the image to different sizes
        crop_h, crop_w = int(height * crop_size), int(width * crop_size)
        image_scaled = cv2.resize(image, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
        for i in range(0, crop_h - patch_size + 1, int(patch_size / 1)):
            for j in range(0, crop_w - patch_size + 1, int(patch_size / 1)):
                x = image_scaled[i:i + patch_size,
                    j:j + patch_size]  # This gets the patch from the original image with size patch_size x patch_size
                patches.append(x)
    return patches

def add_impulse_noise(image, noise_level=0.1):
    new_image = np.copy(image)
    if image is not None:
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                prob = np.random.uniform(0, 1)
                if prob <= noise_level:
                    new_image[row][col] = np.array([255, 255, 255])
    return new_image


def compare(image_path="camel.jpg", flag=1):
    noise_image = cv2.imread(image_path)

    src_patches = np.array(get_all_patches(noise_image)).astype('float32') / 255
    src_patches = tf.clip_by_value(src_patches, clip_value_min=0., clip_value_max=1.)
    src_image = create_image_from_patches(src_patches, noise_image.shape)

    median_image = cv2.medianBlur(noise_image, 3)
    median_patches = np.array(get_all_patches(median_image)).astype('float32') / 255
    median_patches = tf.clip_by_value(median_patches, clip_value_min=0., clip_value_max=1.)
    median_result_image = create_image_from_patches(median_patches, noise_image.shape)

    my_noise_image = median_filter(noise_image, flag)

    patches = np.array(my_noise_image).astype('float32') / 255
    patches = tf.clip_by_value(patches, clip_value_min=0., clip_value_max=1.)

    ridnet_new = tf.keras.models.load_model(WHITE_MODEL)
    if flag == 2:
        ridnet_new = tf.keras.models.load_model(BLACK_MODEL)

    denoised_image = predict_fun(ridnet_new, patches, noise_image)

    new_patches = np.array(get_all_patches(noise_image)).astype('float32') / 255
    new_patches = tf.clip_by_value(new_patches, clip_value_min=0., clip_value_max=1.)

    ridnet_old = tf.keras.models.load_model(RIDNET_WHITE_MODEL)
    if flag == 2:
        ridnet_old = tf.keras.models.load_model(RIDNET_BLACK_MODEL)

    denoised_old_image = predict_fun(ridnet_old, new_patches, noise_image)

    ot.compare(src_image, median_result_image, denoised_old_image, denoised_image)