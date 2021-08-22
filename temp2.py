
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, MaxPooling2D, BatchNormalization, Activation, \
    Conv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

def tfl_model():
    input_shape = (81, 81, 3)

    model = Sequential()

    def conv_bn_relu(filters, **conv_kw):
        # numbers of filters that convolutional layers will learn from.
        model.add(Conv2D(filters, use_bias=False, kernel_initializer='he_normal', **conv_kw))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def dense_bn_relu(units):
        # units = dimensionality of the output space.
        # kernel_initializer: Initializer for the `kernel` weights matrix.
        model.add(Dense(units, use_bias=False, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def spatial_layer(count, filters):
        for i in range(count):
            conv_bn_relu(filters, kernel_size=(3, 3))
        conv_bn_relu(filters, kernel_size=(3, 3), strides=(2, 2))

    conv_bn_relu(32, kernel_size=(3, 3), input_shape=input_shape)
    spatial_layer(1, 32)
    spatial_layer(2, 64)
    spatial_layer(2, 96)

    model.add(Flatten())
    dense_bn_relu(96)
    model.add(Dense(2, activation='softmax'))
    return model


m = tfl_model()
m.summary()

# data_dir = '/tmp/scaleup/'
# datasets = {
#     'val':load_tfl_data(join(data_dir,'val')),
#     'train': load_tfl_data(join(data_dir,'train')),
#     }
# #prepare our model
# m = tfl_model()
# m.compile(optimizer=Adam(),loss =sparse_categorical_crossentropy,metrics=['accuracy'])
#
# train,val = datasets['train'],datasets['val']
# #train it, the model uses the 'train' dataset for learning. We evaluate the "goodness" of the model, by predicting the label of the images in the val dataset.
# history=m.fit(train['images'],train['labels'],validation_data=(val['images'],val['labels']),epochs = 2)