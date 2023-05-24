#Importing the packages
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input,Conv2D,Activation,BatchNormalization,Add,Multiply,Concatenate,GlobalAveragePooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from google.colab.patches import cv2_imshow
import datetime
import pandas as pd
import time

from google.colab import drive
drive.mount('/content/drive')

new_model = tf.keras.models.load_model('ridnet_white_30.h5')

new_model.get_metrics_result()

'''
Структура папок:
train -- исходные изображения без шумов
train_data -- патчи изображения 40-40
train_shum -- исходные изображения + шумz

test -- тестовые изображения без шумов
test_data -- патчи изображения 40-40
test_shum -- тестовые изображения + шум
'''


def get_patches(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    crop_sizes = [1]
    patch_size = 40
    patches = []
    for crop_size in crop_sizes:
        crop_h, crop_w = int(height*crop_size), int(width*crop_size)
        image_scaled = cv2.resize(image, (crop_w, crop_h))
        for i in range(0, crop_h-patch_size+1, int(patch_size/1)):
            for j in range(0, crop_w-patch_size+1, int(patch_size/1)):
              x = image_scaled[i:i+patch_size, j:j+patch_size]
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

'''
Функция, которая считывает папку с изображениями, и записывает результат в другую папку 
'''


def create_dataset(src_dir, noise_level=0.1):
    all_file_path = [filename for filename in os.listdir(src_dir)]
    for file in all_file_path:
        image = cv2.imread(src_dir+file)
        patches = get_patches(image)
        i = 0
        for patch in patches:
            processed_image = add_impulse_noise(patch, noise_level)
            cv2.imwrite(src_dir[:len(src_dir)-1]+"_data_1/"+file+"_"+str(i), patch)
            cv2.imwrite(src_dir[:len(src_dir)-1]+"_shum_1/"+file+"_"+str(i), processed_image)
            i += 1


train_files=[filename for filename in os.listdir('drive/MyDrive/ridnet_data/train/')]
test_files =[filename for filename in os.listdir('drive/MyDrive/ridnet_data/test/')]

"""# Нейронная сеть"""

def _parse_function(filename):
  image_string = tf.io.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image = tf.cast(image_decoded, tf.float32)/255.

  return image, image

def _parse_function_1(filenames):
  image_string = tf.io.read_file(filenames[0])
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image = tf.cast(image_decoded, tf.float32)/255.

  noisy_image_string = tf.io.read_file(filenames[1])
  noisy_image_decoded = tf.image.decode_jpeg(noisy_image_string, channels=3)
  noisy_image = tf.cast(noisy_image_decoded, tf.float32)/255.

  return noisy_image, image

BATCH_SIZE=64
NOISE_LEVELS=[15,25,50] 

train_files=[('drive/MyDrive/ridnet_data/train_data_1/'+ filename, 'drive/MyDrive/ridnet_data/train_shum_1/'+ filename)
              for filename in sorted(os.listdir('drive/MyDrive/ridnet_data/train_data_1/'))]

test_files=[('drive/MyDrive/ridnet_data/test_data_1/'+ filename, 'drive/MyDrive/ridnet_data/test_shum_1/'+ filename)
              for filename in sorted(os.listdir('drive/MyDrive/ridnet_data/test_data_1/'))]

train_dataset = tf.data.Dataset.from_tensor_slices(np.array(train_files))
train_dataset = train_dataset.map(_parse_function_1)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices(np.array(test_files))
test_dataset = test_dataset.map(_parse_function_1)
test_dataset = test_dataset.batch(BATCH_SIZE)

'''
train_dataset_zip = zip(train_shum_dataset, train_dataset)

iterator = list(train_dataset_zip)
iterator = iter(iterator)
a, b = iterator.__next__()
print('Shape of single batch of x : ',a.shape)
print('Shape of single batch of y : ',b.shape)
'''
iterator = iter(train_dataset)
a, b = iterator.__next__()
print('Shape of single batch of x : ',a.shape)
print('Shape of single batch of y : ',b.shape)

fig, axs = plt.subplots(1,10,figsize=(20,4))
for i in range(10):
  axs[i].imshow(a[i])
  axs[i].get_xaxis().set_visible(False)
  axs[i].get_yaxis().set_visible(False)
fig.suptitle('Noisy Images',fontsize=20)
plt.show()
fig, axs = plt.subplots(1,10,figsize=(20,4))
for i in range(10):
  axs[i].imshow(b[i])
  axs[i].get_xaxis().set_visible(False)
  axs[i].get_yaxis().set_visible(False)
fig.suptitle('Ground Truth Images',fontsize=20)
plt.show()

def get_patches(file_name,patch_size,crop_sizes):
    '''This functions creates and return patches of given image with a specified patch_size'''
    image = cv2.imread(file_name) 
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    height, width , channels= image.shape
    patches = []
    for crop_size in crop_sizes: #We will crop the image to different sizes
        crop_h, crop_w = int(height*crop_size),int(width*crop_size)
        image_scaled = cv2.resize(image, (crop_w,crop_h), interpolation=cv2.INTER_CUBIC)
        for i in range(0, crop_h-patch_size+1, patch_size):
            for j in range(0, crop_w-patch_size+1, patch_size):
              x = image_scaled[i:i+patch_size, j:j+patch_size] # This gets the patch from the original image with size patch_size x patch_size
              patches.append(x)
    return patches

def create_image_from_patches(patches,image_shape):
  '''This function takes the patches of images and reconstructs the image'''
  image=np.zeros(image_shape) # Create a image with all zeros with desired image shape
  patch_size=patches.shape[1]
  p=0
  for i in range(0,image.shape[0]-patch_size+1,patch_size):
    for j in range(0,image.shape[1]-patch_size+1,patch_size):
      image[i:i+patch_size,j:j+patch_size]=patches[p] # Assigning values of pixels from patches to image
      p+=1
  return np.array(image)

def predict_fun(model,image_path,noise_level=30):
  #Creating patches for test image
  patches=get_patches(image_path,40,[1])
  test_image=cv2.imread(image_path)

  patches=np.array(patches)
  ground_truth=create_image_from_patches(patches,test_image.shape)

  #predicting the output on the patches of test image
  patches = patches.astype('float32') /255.
  patches_noisy = patches+ tf.random.normal(shape=patches.shape,mean=0,stddev=noise_level/255) 
  patches_noisy = tf.clip_by_value(patches_noisy, clip_value_min=0., clip_value_max=1.)
  noisy_image=create_image_from_patches(patches_noisy,test_image.shape)

  denoised_patches=model.predict(patches_noisy)
  denoised_patches=tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

  #Creating entire denoised image from denoised patches
  denoised_image=create_image_from_patches(denoised_patches,test_image.shape)

  return patches_noisy,denoised_patches,ground_truth/255.,noisy_image,denoised_image


def plot_patches(patches_noisy,denoised_patches):
  fig, axs = plt.subplots(2,10,figsize=(20,4))
  for i in range(10):

    axs[0,i].imshow(patches_noisy[i])
    axs[0,i].title.set_text(' Noisy')
    axs[0,i].get_xaxis().set_visible(False)
    axs[0,i].get_yaxis().set_visible(False)

    axs[1,i].imshow(denoised_patches[i])
    axs[1,i].title.set_text('Denoised')
    axs[1,i].get_xaxis().set_visible(False)
    axs[1,i].get_yaxis().set_visible(False)
  plt.show()

def plot_predictions(ground_truth,noisy_image,denoised_image):
  fig, axs = plt.subplots(1,3,figsize=(15,15))
  axs[0].imshow(ground_truth)
  axs[0].title.set_text('Ground Truth')
  axs[1].imshow(noisy_image)
  axs[1].title.set_text('Noisy Image')
  axs[2].imshow(denoised_image)
  axs[2].title.set_text('Denoised Image')
  plt.show()


#https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
def PSNR(gt, image, max_value=1):
    """"Function to calculate peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((gt - image) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def EAM(input):
    x=Conv2D(64, (3,3), dilation_rate=1,padding='same',activation='relu')(input)
    x=Conv2D(64, (3,3), dilation_rate=2,padding='same',activation='relu')(x)

    y=Conv2D(64, (3,3), dilation_rate=3,padding='same',activation='relu')(input)
    y=Conv2D(64, (3,3), dilation_rate=4,padding='same',activation='relu')(y)

    z=Concatenate(axis=-1)([x,y])
    z=Conv2D(64, (3,3),padding='same',activation='relu')(z)
    add_1=Add()([z, input])

    z=Conv2D(64, (3,3),padding='same',activation='relu')(add_1)
    z=Conv2D(64, (3,3),padding='same')(z)
    add_2=Add()([z,add_1])
    add_2 = Activation('relu')(add_2)

    z=Conv2D(64, (3,3),padding='same',activation='relu')(add_2)
    z=Conv2D(64, (3,3),padding='same',activation='relu')(z)
    z=Conv2D(64, (1,1),padding='same')(z)

    add_3=Add()([z,add_2])
    add_3 = Activation('relu')(add_3)

    z = GlobalAveragePooling2D()(add_3)
    z = tf.expand_dims(z,1)
    z = tf.expand_dims(z,1)

    z=Conv2D(4, (3,3),padding='same',activation='relu')(z)
    z=Conv2D(64, (3,3),padding='same',activation='sigmoid')(z)

    mul=Multiply()([z, add_3])

    return mul

def Model():
    input = Input((40, 40, 3),name='input')
    feat_extraction =Conv2D(64, (3,3),padding='same')(input)
    eam_1=EAM(feat_extraction)
    eam_2=EAM(eam_1)
    eam_3=EAM(eam_2)
    eam_4=EAM(eam_3)
    x=Conv2D(3, (3,3),padding='same')(eam_4)
    add_2=Add()([x, input])

    return Model(input,add_2)

tf.keras.backend.clear_session()
tf.random.set_seed(6908)
ridnet = RIDNET()

ridnet.compile(optimizer=tf.keras.optimizers.Adam(1e-03), loss=tf.keras.losses.MeanAbsoluteError(), run_eagerly=True)

def scheduler(epoch, lr):
  return lr*0.9

checkpoint_path = "model"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
lrScheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
callbacks = [cp_callback,tensorboard_callback,lrScheduler]
ridnet.fit(train_dataset, shuffle=True, epochs=2, validation_data=test_dataset, callbacks=callbacks)

# Функция временной печати изображений
def plot_predictions(ground_truth, noisy_image, denoised_image, last_image):
  '''This function is used for plotting predictions'''
  fig, axs = plt.subplots(2, 2,figsize=(10,10))
  axs[0][0].imshow(ground_truth)
  axs[0][0].title.set_text('Ground Truth')
  axs[0][1].imshow(noisy_image)
  axs[0][1].title.set_text('Noisy Image')
  axs[1][0].imshow(denoised_image)
  axs[1][0].title.set_text('Denoised Image using original model')
  axs[1][1].imshow(last_image)
  axs[1][1].title.set_text('Denoised Image using old model')
  fig.tight_layout()
  plt.show()

def predict_fun(image_path, noise_image_path, noise_level=30):
  #Creating patches for test image
  patches=get_patches(image_path, 40, [1])
  test_image=cv2.imread(image_path)
  patches=np.array(patches)
  ground_truth=create_image_from_patches(patches,test_image.shape)

  noise_patches=get_patches(noise_image_path, 40, [1])
  noise_test_image=cv2.imread(noise_image_path)
  noise_patches=np.array(noise_patches)
  noise_image=create_image_from_patches(patches,test_image.shape)

  #predicting the output on the patches of test image
  '''
  noise_patches = noise_patches.astype('float32') /255.
  patches_noisy = tf.clip_by_value(noise_patches, clip_value_min=0., clip_value_max=1.)
  noisy_image=create_image_from_patches(patches_noisy,test_image.shape)
  '''

  ridnet_new=tf.keras.models.load_model('drive/MyDrive/ridnet.h5')
  ridnet_old=tf.keras.models.load_model('drive/MyDrive/src/ridnet.h5')

  denoised_patches=ridnet_new.predict(noise_patches)
  denoised_patches_old = ridnet_old.predict(noise_patches)

  denoised_patches=tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)
  denoised_patches_old=tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

  #Creating entire denoised image from denoised patches
  denoised_image=create_image_from_patches(denoised_patches, test_image.shape)
  denoised_image_old=create_image_from_patches(denoised_patches_old, test_image.shape)

  return ground_truth, noise_image, denoised_image, denoised_image_old

image = cv2.imread("drive/MyDrive/src/images/camel.jpg")
noised_image = add_impulse_noise(image, 0.1)
cv2.imwrite("shum.jpg", noised_image)

ground_truth, noise_image, denoised_image, denoised_image_old = \
  predict_fun("drive/MyDrive/src/images/camel.jpg", "shum.jpg", noise_level=10)

denoised_image_old

ridnet_new=tf.keras.models.load_model('drive/MyDrive/ridnet.h5')
ridnet_old=tf.keras.models.load_model('drive/MyDrive/src/ridnet.h5')

def predict_fun(model, patches_noisy, gt):
  '''This  function takes noisy patches and original model as input and returns denoised image'''
  height, width, channels= gt.shape
  # gt=cv2.resize(gt, (width//40*40,height//40*40), interpolation=cv2.INTER_CUBIC)
  denoised_patches=model.predict(patches_noisy)
  denoised_patches=tf.clip_by_value(denoised_patches, clip_value_min=0., clip_value_max=1.)

  #Creating entire denoised image from denoised patches
  denoised_image = create_image_from_patches(denoised_patches, gt.shape)

  return denoised_image

def get_patches(image):
    '''This functions creates and return patches of given image with a specified patch_size'''
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    height, width , channels= image.shape
    crop_sizes=[1]
    patch_size=40
    patches = []
    for crop_size in crop_sizes: #We will crop the image to different sizes
        crop_h, crop_w = int(height*crop_size),int(width*crop_size)
        image_scaled = cv2.resize(image, (crop_w,crop_h), interpolation=cv2.INTER_CUBIC)
        for i in range(0, crop_h-patch_size+1, int(patch_size/1)):
            for j in range(0, crop_w-patch_size+1, int(patch_size/1)):
              x = image_scaled[i:i+patch_size, j:j+patch_size] # This gets the patch from the original image with size patch_size x patch_size
              patches.append(x)
    return patches



def create_image_from_patches(patches,image_shape):
  '''This function takes the patches of images and reconstructs the image'''
  image=np.zeros(image_shape) # Create a image with all zeros with desired image shape
  patch_size=patches.shape[1]
  p=0
  for i in range(0,image.shape[0]-patch_size+1,int(patch_size/1)):
    for j in range(0,image.shape[1]-patch_size+1,int(patch_size/1)):
      image[i:i+patch_size,j:j+patch_size]=patches[p] # Assigning values of pixels from patches to image
      p+=1
  return np.array(image)

def get_image(gt, noise_level, with_filter = True):
  '''This function takes a image and adds noise with specified level and return  noisy image and its patches'''
  original_patches = get_patches(gt)
  noisy_patches = get_patches(add_impulse_noise(gt, noise_level))

  height, width, channels = gt.shape
  
  patches = np.array(original_patches)
  noisy_patches = np.array(noisy_patches)
  ground_truth = create_image_from_patches(patches, gt.shape)
  

  #predicting the output on the patches of test image
  patches = patches.astype('float32') /255
  noisy_patches = noisy_patches.astype('float32') / 255
  patches_noisy = tf.clip_by_value(noisy_patches, clip_value_min=0., clip_value_max=1.)
  
  noisy_image = create_image_from_patches(patches_noisy, gt.shape)
  
  return ground_truth/255., noisy_image, patches_noisy

def final(imagefile, noise_level, with_filter = True):
  '''This function takes image path and noise level and does all the steps from input to getting predictions'''
  original_image = cv2.imread(imagefile)
  noisy_image = cv2.imread("shum.jpg")

  noisy_patches = np.array(get_patches(noisy_image)).astype('float32') / 255
  noisy_patches = tf.clip_by_value(noisy_patches, clip_value_min=0., clip_value_max=1.)
  noise_image = create_image_from_patches(noisy_patches, original_image.shape)

  noisy_patches = np.array(get_patches(noisy_image)).astype('float32') / 255
  noisy_patches = tf.clip_by_value(noisy_patches, clip_value_min=0., clip_value_max=1.)

  original_patches = np.array(get_patches(original_image)).astype('float32') / 255
  original_patches = tf.clip_by_value(original_patches, clip_value_min=0., clip_value_max=1.)

  original_image = create_image_from_patches(original_patches, original_image.shape)
  noisy_image = create_image_from_patches(noisy_patches, original_image.shape)

  ridnet_new=tf.keras.models.load_model('drive/MyDrive/ridnet.h5')
  ridnet_old=tf.keras.models.load_model('drive/MyDrive/src/ridnet.h5')

  start=time.time()
  denoised_image = predict_fun(ridnet_new, noisy_patches, original_image)
  end=time.time()

  denoised_image_old = predict_fun(ridnet_old, noisy_patches, original_image)
  
  time_taken=end-start

  print('PSNR of noisy image :', PSNR(noisy_image, original_image))
  print('PSNR of denoised image using original model: %.3f db , Time taken : %.5f seconds'%(PSNR(original_image, denoised_image),time_taken))

  print('PSNR of denoised image using original model: %.3f db , Time taken : %.5f seconds'%(PSNR(original_image, denoised_image_old),time_taken))
  plot_predictions(original_image, noisy_image, denoised_image, denoised_image_old)

  return denoised_image, True

denoised_image, result = final("drive/MyDrive/src/images/camel.jpg", 10)

