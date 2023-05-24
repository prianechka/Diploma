def get_patches(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    crop_sizes = [1]
    patch_size = 40
    patches = []
    for crop_size in crop_sizes:
        crop_h, crop_w = int(height * crop_size), int(width * crop_size)
        image_scaled = cv2.resize(image, (crop_w, crop_h))
        for i in range(0, crop_h - patch_size + 1, int(patch_size / 1)):
            for j in range(0, crop_w - patch_size + 1, int(patch_size / 1)):
                x = image_scaled[i:i + patch_size, j:j + patch_size]
                patches.append(x)
    return patches


def create_dataset(src_dir, noise_level = 0.1):
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