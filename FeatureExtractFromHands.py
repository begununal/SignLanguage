from dataLoader import readImageDatasets
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Deconvolution2D, Flatten, Activation, Concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras import backend as K
import cv2
import matplotlib
import shutil
from matplotlib import pyplot as plt
import uuid
from classifyHandFeatures import SVM

import numpy as np
import pickle
import os
import sys

id = uuid.uuid4()
def runFeatureExtraction(numEpochs,convLayerCount,convFilterSize,FilterCount,InputImageSize,params,GPU=0):
    if not os.path.isfile("Data.pickle"):
        ImageList, LabelList, UserList, DatasetNameList = readImageDatasets()
        pickle.dump([ImageList, LabelList, UserList, DatasetNameList], open("Data.pickle", "wb"))
    else:
        [ImageList, LabelList, UserList, DatasetNameList ] = pickle.load(open("Data.pickle", "rb"))
    UserList = np.array(UserList)
    ImageList = np.array(ImageList)
    LabelList = np.array(LabelList)

    ValidationSet = UserList == 'A'

    TrainingSet = ~ValidationSet


    x_train = ImageList[TrainingSet]
    x_val = ImageList[ValidationSet]
    y_train = LabelList[TrainingSet]
    y_val = LabelList[ValidationSet]
    #x_train = x_train[np.linspace(0, x_train.shape[0]-1, num=5000).astype(dtype=np.uint32)]
    #y_train = y_train[np.linspace(0, y_train.shape[0]-1, num=5000).astype(dtype=np.uint32)]
    #x_val = x_val[np.linspace(0, x_val.shape[0]-1, num=500).astype(dtype=np.uint32)]
    #y_val = y_val[np.linspace(0, y_val.shape[0]-1, num=500).astype(dtype=np.uint32)]
    input_img = Input(shape=(InputImageSize, InputImageSize, 3))
    encoded1 = input_img
    encoded2 = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same')(encoded1)
    print(encoded2.shape)
    encoded3 = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same')(encoded2)
    print(encoded3.shape)
    encoded4 = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', strides=2, padding='same')(encoded3)
    print(encoded4.shape)
    FilterCount *= 2
    encoded5 = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same')(encoded4)
    print(encoded5.shape)
    encoded6 = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same')(encoded5)
    print(encoded6.shape)
    encoded7 = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', strides=2, padding='same')(encoded6)
    print(encoded7.shape)
    encoded8 = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same')(encoded7)
    print(encoded8.shape)
    encoded9 = Conv2D(FilterCount, (1, 1), activation='relu', padding='same')(encoded8)
    print(encoded9.shape)

    """
    conv1 = Conv2D(32, (convFilterSize, convFilterSize), activation='relu', padding='same')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, (convFilterSize, convFilterSize), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(8, (convFilterSize, convFilterSize), activation='relu', padding='same',name='encoded')(pool2)
    """
    decoded = Conv2D(FilterCount, (1, 1), activation='relu', padding='same')(encoded9)
    print(decoded.shape)
    decoded = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same')(decoded)
    print(decoded.shape)
    decoded = Concatenate()([encoded8, decoded])
    decoded = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same')(decoded)
    print(decoded.shape)
    decoded = Concatenate()([encoded7, decoded])
    templayer = Conv2DTranspose(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same', strides=(2, 2), use_bias=False, data_format="channels_last")
    decoded = templayer(decoded)
    print(decoded.shape)
    decoded = Concatenate()([encoded6, decoded])
    decoded = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same')(decoded)
    print(decoded.shape)
    FilterCount = int(FilterCount / 2)
    decoded = Concatenate()([encoded5, decoded])
    decoded = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same')(decoded)
    print(decoded.shape)
    decoded = Concatenate()([encoded4, decoded])
    decoded = Conv2DTranspose(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same', strides=(2, 2), use_bias=False, data_format="channels_last")(decoded)
    print(decoded.shape)
    decoded = Concatenate()([encoded3, decoded])
    decoded = Conv2D(FilterCount, (convFilterSize, convFilterSize), activation='relu', padding='same')(decoded)
    print(decoded.shape)
    decoded = Concatenate()([encoded2, decoded])
    decoded = Conv2D(3, (convFilterSize, convFilterSize), activation='relu', padding='same')(decoded)
    print(decoded.shape)
    """
    conv4 = Conv2D(8, (convFilterSize, convFilterSize), activation='relu', padding='same')(conv3)
    up1 = UpSampling2D((2, 2))(conv4)
    conv5 = Conv2D(32, (convFilterSize, convFilterSize), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(conv5)
    decoded = Conv2D(3, (convFilterSize, convFilterSize), activation='sigmoid', padding='same')(up2)
    """

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.summary()
    x_train = x_train.astype('float32') / 255.
    x_val = x_val.astype('float32') / 255.


    autoencoder.fit(x_train, x_train,
                    epochs=int(numEpochs),
                    batch_size=32,
                    shuffle=True,
                    validation_data=(x_val, x_val),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    autoencoder.save('./my_autoencoder_model_' + str(id) + '.h5')

    get_3rd_layer_output = K.function([autoencoder.layers[0].input], [autoencoder.get_layer("encoded").output])

    train_features = get_3rd_layer_output([x_train])
    val_features = get_3rd_layer_output([x_val])

    train_imgs = autoencoder.predict(x_train)
    val_imgs = autoencoder.predict(x_val)

    #shutil.rmtree('results',ignore_errors=True)
    if not os.path.exists(os.path.join(os.getcwd(), 'results/')):
        print(os.path.join(os.getcwd(), 'results'))
        os.makedirs('results/')
    if not os.path.exists(os.path.join(os.getcwd(), 'results/images/')):
        print(os.path.join(os.getcwd(), 'results/images'))
        os.chdir('./results/')
        os.makedirs('images/')
        os.chdir('./../')
    print(os.getcwd(), "1")
    key = id
    for i in range(0,20):
        img1 = (cv2.resize(cv2.cvtColor(val_imgs[i], cv2.COLOR_BGR2RGB), (128, 128)) *255. ).astype(dtype=np.uint8)
        img2 = (cv2.resize(cv2.cvtColor(x_train[i], cv2.COLOR_BGR2RGB), (128, 128)) * 255.).astype(dtype=np.uint8)

        cv2.imwrite('results/images/'+format(i)+'.png',img1)
        cv2.imwrite('results/images/orig_' + format(i) + '.png', img2)
    os.chdir('./results/')
    print(os.getcwd(), "2")
    pickle.dump([train_features, val_features, train_imgs, val_imgs, y_train, y_val], open('Results_' + str(key) + '.pickle', 'wb'))
    os.chdir('./../')
    print(os.getcwd(), "3")
    return SVM(key, params)
if __name__ == "__main__":
    main(sys.argv[1])