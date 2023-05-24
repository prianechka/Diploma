import numpy as np
import output as ot
import cv2
import tensorflow as tf
import metrics as ms

WHITE = np.array([255, 255, 255])
EPS = 350


def clean_image(imagefile):
    original_image = cv2.imread(imagefile)
    image = median_filter(original_image)

    patches = np.array(image).astype('float32') / 255
    patches = tf.clip_by_value(patches, clip_value_min=0., clip_value_max=1.)

    ridnet_new = tf.keras.models.load_model('models/ridnet_white_30.h5')

    denoised_image = predict_fun(ridnet_new, patches, original_image)

    src_patches = np.array(get_all_patches(original_image)).astype('float32') / 255
    src_patches = tf.clip_by_value(src_patches, clip_value_min=0., clip_value_max=1.)

    ot.plot_clean_image(original_image, denoised_image)

    return denoised_image, True


def predict_fun(model, patches_noisy, gt):
    denoised_patches = model.predict(patches_noisy)
    denoised_patches = tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

    denoised_image = create_image_from_patches(denoised_patches, gt.shape)

    return denoised_image


def median_filter(image):
    patches = get_all_patches(image)
    i = 0
    for patch in patches:
        patches[i] = apply_median_filter(patches[i], i)
        i += 1
    return patches


def apply_median_filter(patch, i):
    if count_noise(patch, i):
        after_median_patch = np.array([cv2.medianBlur(patch, 3)])
        tf_type_patch = after_median_patch.astype('float32') / 255
        tf_type_patch = tf.clip_by_value(tf_type_patch)

        ridnet_median = tf.keras.models.load_model('model')
        corrected_patch = ridnet_median.predict(tf_type_patch)
        corrected_patch = (corrected_patch * 255).astype('int')
        return create_image_from_patches(corrected_patch, patch.shape)

    return patch


def count_noise(patch, i):
    counter = 0
    for row in patch:
        for col in row:
            if col[0] > COLOR[0] and col[1] > COLOR[1] and col[2] > COLOR[2]:
                counter += 1
            elif col[0] < COLOR[0] and col[1] < COLOR[1] and col[2] < COLOR[2]:
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