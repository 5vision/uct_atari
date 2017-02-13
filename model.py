from keras.layers import Conv2D, Dense, Input, Flatten, Dropout, BatchNormalization
from keras.models import Model


def build_model(state_size, n_actions,
                conv_filters=(32, 32, 32, 32),
                conv_sizes=(3, 3, 3, 3),
                conv_strides=(2, 2, 2, 2),
                pads=['same']*4,
                conv_droputs=[0.0]*4,
                fc_sizes=(512, 256),
                fc_dropouts=[0.]*2,
                batch_norm=True,
                activation='relu'):

    inputs = Input(shape=state_size)

    conv = inputs
    for (f_num, f_size, f_stride, pad, droput) in zip(conv_filters, conv_sizes, conv_strides, pads, conv_droputs):
        conv = Conv2D(f_num, f_size, f_size,
                      border_mode=pad,
                      subsample=(f_stride, f_stride),
                      activation=activation,
                      dim_ordering='th')(conv)
        if batch_norm:
            conv = BatchNormalization(mode=0, axis=1)(conv)
        if droput > 0:
            conv = Dropout(droput)(conv)


    fc = Flatten()(conv)
    for fc_size, dropout in zip(fc_sizes, fc_dropouts):
        fc = Dense(fc_size, activation=activation)(fc)
        if batch_norm:
            fc = BatchNormalization(mode=1)(fc)
        if droput > 0:
            fc = Dropout(droput)(fc)

    actor = Dense(n_actions, activation='softmax')(fc)

    model = Model(input=inputs, output=actor)

    return model
