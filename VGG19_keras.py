import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout


input_img = (224, 224, 3)

num_of_classes = 1000

def VGG19():
    inputs = keras.Input(shape=(224, 224, 3))


    # conv3-64
    conv1 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding = 'same')(inputs)
    # All hidden layers are equipped with the rectification (ReLU).
    conv1 = Activation('relu')(conv1)
    # LRN does not improve the performance on the ILSVRC dataset.
    # (224, 224, 64)


    # conv3-64
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding = 'same')(conv1)
    conv2 = Activation('relu')(conv2)
    # Max-pooling
    # Spatial pooling is carried out by five max-pooling layers, which follow some of the conv. layers.
    # Max-pooling is performed over a 2 × 2 pixel window, with stride 2.
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
    # (112, 112, 64)


    # conv3-128
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding = 'same')(pool1)
    conv3 = Activation('relu')(conv3)
    # (112, 112, 64)


    # conv3-128
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding = 'same')(conv3)
    conv4 = Activation('relu')(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)
    # (56, 56, 128)


    # conv3-256
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding = 'same')(pool2)
    conv5 = Activation('relu')(conv5)
    # (56, 56, 256)


    # conv3-256
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding = 'same')(conv5)
    conv6 = Activation('relu')(conv6)
    # (56, 56, 256)


    # conv3-256
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding = 'same')(conv6)
    conv7 = Activation('relu')(conv7)
    # (56, 56, 256)


    # conv3-256
    conv8 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding = 'same')(conv7)
    conv8 = Activation('relu')(conv8)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv8)
    # (28, 28, 256)


    # conv3-512
    conv9 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(pool3)
    conv9 = Activation('relu')(conv9)
    # (28, 28, 512)


    # conv3-512
    conv10 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(conv9)
    conv10 = Activation('relu')(conv10)
    # (28, 28, 512)


    # conv3-512
    conv11 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(conv10)
    conv11 = Activation('relu')(conv11)
    # (28, 28, 512)


    # conv3-512
    conv12 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(conv11)
    conv12 = Activation('relu')(conv12)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv12)
    # (14, 14, 512)


    # conv3-512
    conv13 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(pool4)
    conv13 = Activation('relu')(conv13)
    # (14, 14, 512)


    # conv3-512
    conv14 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(conv13)
    conv14 = Activation('relu')(conv14)
    # (14, 14, 512)


    # conv3-512
    conv15 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(conv14)
    conv15 = Activation('relu')(conv15)
    # (14, 14, 512)


    # conv3-512
    conv16 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(conv15)
    conv16 = Activation('relu')(conv16)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv16)
    # (7, 7, 512)


    # FC-4096
    fc1 = Flatten()(pool5)
    # The first two have 4096 channels each, the third performs 1000-way ILSVRC classification
    # and thus contains 1000 channels (one for each class).
    fc1 = Dense(4096)(fc1)
    fc1 = Activation('relu')(fc1)


    # FC-4096
    fc2 = Dense(4096)(fc1)
    fc2 = Activation('relu')(fc2)


    # FC-1000
    # The final layer is the soft-max layer.
    fc3 = Dense(1000)(fc1)
    outputs = Activation('softmax')(fc3)


    return keras.Model(inputs=inputs, outputs=outputs)

model = VGG19()
model.summary()