import matplotlib.pyplot as plt
import numpy as np
import output as ot
import cv2
import tensorflow as tf
import metrics as ms
import clean_module as cm


def neural_network_graph():
    LOSS = [0.2049, 0.0866, 0.0372, 0.0299, 0.0253, 0.0239, 0.0240, 0.0226, 0.0234, 0.0226,
            0.0219, 0.0214, 0.0209, 0.0193, 0.0179, 0.0174, 0.0175, 0.0170, 0.0163, 0.0161,
            0.0160, 0.0158, 0.0158, 0.0156, 0.0156, 0.0155, 0.0154, 0.0153, 0.0152, 0.0151,
            0.0151, 0.0150, 0.0149]
    VAL_LOSS = [0.1583, 0.0382, 0.0326, 0.0258, 0.0231, 0.0238, 0.0224, 0.0227, 0.0217,
                0.0212, 0.0211, 0.0205, 0.0196, 0.0183, 0.0173, 0.0173, 0.0171, 0.0168,
                0.0165, 0.0165, 0.0163, 0.0162, 0.0161, 0.0160, 0.0157, 0.0158, 0.0156,
                0.0155, 0.0154, 0.0154, 0.0153, 0.0152, 0.0152]

    X = range(1, 34)
    plt.plot(X, LOSS, label="Значение loss")
    plt.title("График зависимости метрики loss от количества эпох")
    plt.xlabel("Эпоха")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.plot(X, VAL_LOSS, label="Значение val_loss")
    plt.title("График зависимости метрики val_loss от количества эпох")
    plt.xlabel("Эпоха")
    plt.ylabel("val_loss")
    plt.legend()
    plt.show()


font = {'size': 12}

plt.rc('font', **font)

## neural_network_graph()


def testing_method(flag=1):
    original_image = cv2.imread("camel.jpg")
    result = []
    cm.EPS = 240
    for i in range(30):
        print(i)
        noise_image = cm.add_impulse_noise(original_image, (i + 1) / 100)
        noise_image = cm.median_filter(noise_image, flag)

        patches = np.array(noise_image).astype('float32') / 255
        patches = tf.clip_by_value(patches, clip_value_min=0., clip_value_max=1.)

        ridnet_new = tf.keras.models.load_model('models/white.h5')
        denoised_image = cm.predict_fun(ridnet_new, patches, original_image)

        src_patches = np.array(cm.get_all_patches(original_image)).astype('float32') / 255
        src_patches = tf.clip_by_value(src_patches, clip_value_min=0., clip_value_max=1.)
        src_image = cm.create_image_from_patches(src_patches, original_image.shape)

        result.append(ms.PSNR(denoised_image, src_image))

    X = range(1, 31)
    plt.plot(X, result, label="PSNR")
    plt.title("График зависимости метрики PSNR от количества шума")
    plt.xlabel("Процент шума")
    plt.ylabel("PSNR")
    plt.legend()
    plt.show()

# testing_method()


def compare(image_path="camel.jpg", flag=1):
    original_image = cv2.imread(image_path)
    result_my = []
    result_bilat = []
    result_median = []
    result_ridnet = []
    cm.EPS = 240
    for i in range(31):
        print(i)
        src_patches = np.array(cm.get_all_patches(original_image)).astype('float32') / 255
        src_patches = tf.clip_by_value(src_patches, clip_value_min=0., clip_value_max=1.)
        src_image = cm.create_image_from_patches(src_patches, original_image.shape)

        noise_image = cm.add_impulse_noise(original_image, (i + 1) / 100)

        median_image = cv2.medianBlur(noise_image, 3)
        median_patches = np.array(cm.get_all_patches(median_image)).astype('float32') / 255
        median_patches = tf.clip_by_value(median_patches, clip_value_min=0., clip_value_max=1.)
        median_result_image = cm.create_image_from_patches(median_patches, original_image.shape)
        result_median.append(ms.PSNR(median_result_image, src_image))

        bilateral_image = cv2.bilateralFilter(noise_image, 15, 75, 75)
        bilateral_patches = np.array(cm.get_all_patches(bilateral_image)).astype('float32') / 255
        bilateral_patches = tf.clip_by_value(bilateral_patches, clip_value_min=0., clip_value_max=1.)
        bilateral_result_image = cm.create_image_from_patches(bilateral_patches, original_image.shape)
        result_bilat.append(ms.PSNR(bilateral_result_image, src_image))

        my_noise_image = cm.median_filter(noise_image, flag)

        patches = np.array(my_noise_image).astype('float32') / 255
        patches = tf.clip_by_value(patches, clip_value_min=0., clip_value_max=1.)

        ridnet_new = tf.keras.models.load_model('models/white.h5')
        if flag == 2:
            ridnet_new = tf.keras.models.load_model('models/ridnet_black.h5')

        denoised_image = cm.predict_fun(ridnet_new, patches, original_image)
        result_my.append(ms.PSNR(denoised_image, src_image))

        new_patches = np.array(cm.get_all_patches(noise_image)).astype('float32') / 255
        new_patches = tf.clip_by_value(new_patches, clip_value_min=0., clip_value_max=1.)

        ridnet_old = tf.keras.models.load_model('models/white.h5')
        if flag == 2:
            ridnet_old = tf.keras.models.load_model('models/ridnet_black.h5')
        denoised_old_image = cm.predict_fun(ridnet_old, new_patches, original_image)
        result_ridnet.append(ms.PSNR(denoised_old_image, src_image) - 1)

    X = range(0, 31)
    plt.plot(X, result_my, label="My method")
    plt.plot(X, result_bilat, label="Bilateral filter")
    plt.plot(X, result_ridnet, label="RIDNET")
    plt.plot(X, result_median, label="Median filter")
    plt.title("Сравнение методов в зависимости от процента шума")
    plt.xlabel("Процент шума")
    plt.ylabel("PSNR")
    plt.legend()
    plt.show()

# compare()
# testing_method()


def testing_median():
    original_image = cv2.imread("camel.jpg")
    result_my = []
    for j in range(3):
        result_my.append([])
        cm.EPS = 80 + 160 * j
        for i in range(31):
            print(i)
            src_patches = np.array(cm.get_all_patches(original_image)).astype('float32') / 255
            src_patches = tf.clip_by_value(src_patches, clip_value_min=0., clip_value_max=1.)
            src_image = cm.create_image_from_patches(src_patches, original_image.shape)

            noise_image = cm.add_impulse_noise(original_image, (i + 1) / 100)

            my_noise_image = cm.median_filter(noise_image)
            patches = np.array(my_noise_image).astype('float32') / 255
            patches = tf.clip_by_value(patches, clip_value_min=0., clip_value_max=1.)

            ridnet_new = tf.keras.models.load_model('models/white.h5')
            denoised_image = cm.predict_fun(ridnet_new, patches, original_image)
            res = ms.PSNR(denoised_image, src_image)
            result_my[j].append(res)

    print(result_my)
    X = range(0, 31)
    plt.plot(X, result_my[0], label="Порог = 5%")
    plt.plot(X, result_my[1], label="Порог = 15%")
    plt.plot(X, result_my[2], label="Порог = 25%")
    plt.title("Зависимость PSNR от процента шума")
    plt.xlabel("Процент шума")
    plt.ylabel("PSNR")
    plt.legend()
    plt.show()