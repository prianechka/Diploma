import matplotlib.pyplot as plt


def plot_clean_image(noisy_image, denoised_image):
    font = {'size': 14}
    plt.rc('font', **font)
    plt.axis('off')
    fig, axs = plt.subplots(1, 2, figsize=(18,18))
    fig.suptitle('Результаты очистки изображения', fontsize=16)
    axs[0].imshow(noisy_image)
    axs[0].title.set_text('Исходное изображение')
    axs[0].axis('off')
    axs[1].imshow(denoised_image)
    axs[1].title.set_text('Очищенное изображение')
    axs[1].axis('off')
    fig.tight_layout()
    plt.show()

def plot_original_image(noisy_image, denoised_image, original_image):
    font = {'size': 14}
    plt.rc('font', **font)
    plt.axis('off')
    fig, axs = plt.subplots(1, 3, figsize=(18,18))
    fig.suptitle('Результаты очистки изображения', fontsize=16)
    axs[0].imshow(noisy_image)
    axs[0].title.set_text('Загрязненное изображение')
    axs[0].axis('off')
    axs[1].imshow(denoised_image)
    axs[1].title.set_text('Очищенное изображение')
    axs[1].axis('off')
    axs[2].imshow(original_image)
    axs[2].title.set_text('Оригинальное изображение')
    axs[2].axis('off')
    fig.tight_layout()
    plt.show()

def compare(noise_image, median_image, ridnet_image, denoised_image):
    font = {'size': 14}
    plt.rc('font', **font)
    plt.axis('off')
    fig, axs = plt.subplots(2, 2, figsize=(12,12))
    fig.suptitle('Результаты сравнения методов', fontsize=16)
    axs[0][0].imshow(noise_image)
    axs[0][0].title.set_text('Загрязненное изображение')
    axs[0][0].axis('off')

    axs[0][1].imshow(median_image)
    axs[0][1].title.set_text('Медианный фильтр')
    axs[0][1].axis('off')

    axs[1][0].imshow(ridnet_image)
    axs[1][0].title.set_text('RIDNET')
    axs[1][0].axis('off')

    axs[1][1].imshow(denoised_image)
    axs[1][1].title.set_text('My method')
    axs[1][1].axis('off')

    fig.tight_layout()
    plt.show()