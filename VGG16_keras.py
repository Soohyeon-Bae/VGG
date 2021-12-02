import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout


input_img = (224, 224, 3)

num_of_classes = 1000

def VGG16():
    inputs = keras.Input(shape=(224, 224, 3))
    # 1번째 레이어 : conv3-64
    # same padding : 입력 이미지의 크기와 출력 이미지의 크기가 동일하도록 만드는 padding 값을 사용하는 것
    conv1 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding = 'same')(inputs)
    # All hidden layers are equipped with the rectification (ReLU).
    conv1 = Activation('relu')(conv1)
    # LRN does not improve the performance on the ILSVRC dataset.
    # 1번째 레이어 output = (224, 224, 64)


    # 2번째 레이어 : conv3-64
    # input_size = (224, 224, 64)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding = 'same')(conv1)
    conv2 = Activation('relu')(conv2)
    # Max-pooling
    # Spatial pooling is carried out by five max-pooling layers, which follow some of the conv. layers.
    # Max-pooling is performed over a 2 × 2 pixel window, with stride 2.
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
    # output_size = (((n-ps)/s)+1, ((n-ps)/s)+1)
    # 2번째 레이어 output = (112, 112, 64)


    # 3번째 레이어 : conv3-128
    # input_size = (112, 112, 64)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding = 'same')(pool1)
    conv3 = Activation('relu')(conv3)
    # 3번째 레이어 output = (112, 112, 64)


    # 4번째 레이어 : conv3-128
    # input_size = (112, 112, 64)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding = 'same')(conv3)
    conv4 = Activation('relu')(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv4)
    # 4번째 레이어 output = (56, 56, 128)


    # 5번째 레이어 : conv3-256
    # input_size = (56, 56, 128)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding = 'same')(pool2)
    conv5 = Activation('relu')(conv5)
    # 5번째 레이어 output = (56, 56, 256)


    # 6번째 레이어 : conv3-256
    # input_size = (56, 56, 256)
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding = 'same')(conv5)
    conv6 = Activation('relu')(conv6)
    # 6번째 레이어 output = (56, 56, 256)


    # 7번째 레이어 : conv3-256
    # input_size = (56, 56, 256)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding = 'same')(conv6)
    conv7 = Activation('relu')(conv7)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv7)
    # 7번째 레이어 output = (28, 28, 256)


    # 8번째 레이어 : conv3-512
    # input_size = (28, 28, 256)
    conv8 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(pool3)
    conv8 = Activation('relu')(conv8)
    # 8번째 레이어 output = (28, 28, 512)


    # 9번째 레이어 : conv3-512
    # input_size = (28, 28, 512)
    conv9 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(conv8)
    conv9 = Activation('relu')(conv9)
    # 9번째 레이어 output = (28, 28, 512)


    # 10번째 레이어 : conv3-512
    # input_size = (28, 28, 512)
    conv10 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(conv9)
    conv10 = Activation('relu')(conv10)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv10)
    # 10번째 레이어 output = (14, 14, 512)


    # 11번째 레이어 : conv3-512
    # input_size = (14, 14, 512)
    conv11 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(pool4)
    conv11 = Activation('relu')(conv11)
    # 11번째 레이어 output = (14, 14, 512)


    # 12번째 레이어 : conv3-512
    # input_size = (14, 14, 512)
    conv12 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(conv11)
    conv12 = Activation('relu')(conv12)
    # 12번째 레이어 output = (14, 14, 512)


    # 13번째 레이어 : conv3-512
    # input_size = (14, 14, 512)
    conv13 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding = 'same')(conv12)
    conv13 = Activation('relu')(conv13)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv13)
    # 10번째 레이어 output = (7, 7, 512)


    # 14번째 레이어 : FC-4096
    # FC layer는 layer의 뉴런 수와 동일한 길이의 벡터 출력
    fc1 = Flatten()(pool5)
    # The first two have 4096 channels each, the third performs 1000-way ILSVRC classification
    # and thus contains 1000 channels (one for each class).
    fc1 = Dense(4096)(fc1)
    fc1 = Activation('relu')(fc1)


    # 15번째 레이어 : FC-4096
    fc2 = Dense(4096)(fc1)
    fc2 = Activation('relu')(fc2)


    # 16번째 레이어 : FC-1000
    # The final layer is the soft-max layer.
    fc3 = Dense(1000)(fc1)
    outputs = Activation('softmax')(fc3)


    return keras.Model(inputs=inputs, outputs=outputs)

model = VGG16()
model.summary()
