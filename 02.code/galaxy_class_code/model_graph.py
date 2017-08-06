from keras import backend as K
from keras.models import Sequential, Input, load_model, Model
from keras.layers import Input, Dropout, Activation
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers import Lambda, dot, add, concatenate




def model_graphical(img_rows,img_cols,img_channels)

    main_input = Input(shape=(img_rows,img_cols,img_channels), name='main_input')

    x = Lambda(lambda x : x*1.0/255)(main_input)

    x = BatchNormalization()(x)
    x = Conv2D(filters=32, padding='same', kernel_size=(4,4),
               data_format='channels_last', name='Conv-Input-a' )(x)
    x = Conv2D(filters=32, padding='same', kernel_size=(4,4),
               name='Conv-Input-b' )(x)
    x = Conv2D(filters=32, padding='same', kernel_size=(4,4),
               name='Conv-Input-c' )(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)



    x = BatchNormalization()(x)
    x = Conv2D(filters=64, padding='same', kernel_size=(2,2),
               name='Conv-1-a' )(x)
    x = Conv2D(filters=64, padding='same', kernel_size=(2,2),
               name='Conv-1-b' )(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)



    x = BatchNormalization()(x)
    x = Conv2D(filters=128, padding='same', kernel_size=(2,2),
               name='Conv-2-a' )(x)
    x = Conv2D(filters=128, padding='same', kernel_size=(2,2),
               name='Conv-2-b' )(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)



    x = BatchNormalization()(x)
    x = Conv2D(filters=256, padding='same', kernel_size=(2,2),
               name='Conv-3-a' )(x)
    x = Conv2D(filters=256, padding='same', kernel_size=(2,2),
               name='Conv-3-b' )(x)
    x = Conv2D(filters=256, padding='same', kernel_size=(2,2),
               name='Conv-3-c' )(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)



    x = BatchNormalization()(x)
    x = Conv2D(filters=512, padding='same', kernel_size=(2,2),
               name='Conv-4-a' )(x)
    x = Conv2D(filters=512, padding='same', kernel_size=(2,2),
               name='Conv-4-b' )(x)
    x = Conv2D(filters=512, padding='same', kernel_size=(2,2),
               name='Conv-4-c' )(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)



    x = BatchNormalization()(x)
    x = Conv2D(filters=768, padding='same', kernel_size=(2,2),
               name='Conv-5-a' )(x)
    x = Conv2D(filters=768, padding='same', kernel_size=(2,2),
               name='Conv-5-b' )(x)
    x = Conv2D(filters=768, padding='same', kernel_size=(2,2),
               name='Conv-5-c' )(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)



    x = BatchNormalization()(x)
    x = Conv2D(filters=1024, padding='same', kernel_size=(2,2),
               name='Conv-6-a' )(x)
    x = Conv2D(filters=1024, padding='same', kernel_size=(2,2),
               name='Conv-6-b' )(x)
    x = Conv2D(filters=1024, padding='same', kernel_size=(2,2),
               name='Conv-6-c' )(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.2)(x)



    x = BatchNormalization()(x)
    x = Dense(500,name='features')(x)
    features = Activation('relu')(x)

    model = Model(main_input,features)


    # In[36]:

    model_out_shape = features.shape.as_list()[1:]

    class1_input = Dense(100,activation='relu')(features)
    class1 = Dense(3,activation='softmax',name='Dense-class-01')(class1_input)


    class2_input = Dense(100,activation='relu')(features)
    class2_prob = Dense(2,activation='softmax')(class2_input)
    class2 = Lambda(lambda x : K.expand_dims(
                                    dot(inputs=[x,class1],axes=0)[:,:,:,1],
                                axis=1),
                    name='score-class-02')(class2_prob)


    class3_input = Dense(100,activation='relu')(features)
    class3_prob = Dense(2,activation='softmax')(class3_input)
    class3 = Lambda(lambda x : K.expand_dims(
                                    dot(inputs=[x,class2_prob],axes=0)[:,:,:,1],
                                axis=1),
                    name='score-class-03')(class3_prob)



    class4_input = Dense(100,activation='relu')(features)
    class4_prob = Dense(2,activation='softmax')(class4_input)
    class4 = Lambda(lambda x : K.expand_dims(
                                            K.sum(
                                                  dot(inputs=[x,class3_prob],axes=0),
                                                  axis=-1
                                                 ),
                                            axis=1),name='score-class-04')(class4_prob)



    class11_input = Dense(100,activation='relu')(features)
    class11_prob = Dense(6,activation='softmax')(class11_input)
    class11 = Lambda(lambda x : x , name='score-class-11')(class11_prob)




    class5_input = Dense(100,activation='relu')(features)
    class5_prob = Dense(4,activation='softmax')(class5_input)
    class5 = Lambda(lambda x : K.expand_dims(
                                    add(
                                        [
                                            dot(inputs=[x,class4_prob],axes=0)[:,:,:,1],
                                            K.sum(
                                                dot(inputs=[x,class11_prob],axes=0),
                                                axis=-1),
                                        ]),
                                axis=1),
                    name='score-class-05')(class5_prob)




    class6_input = Dense(100,activation='relu')(features)
    class6_prob = Dense(2,activation='softmax')(class6_input)
    class6 = Lambda(lambda x : x, name='Dense-class-06')(class6_prob)



    class7_input = Dense(100,activation='relu')(features)
    class7_prob = Dense(3,activation='softmax')(class7_input)
    class7 = Lambda(lambda x : K.expand_dims(
                                        dot(inputs=[x,class1],axes=0)[:,:,:,0],
                              axis=1),
                    name='score-class-07')(class7_prob)



    class8_input = Dense(100,activation='relu')(features)
    class8_prob = Dense(7,activation='softmax')(class8_input)
    class8 = Lambda(lambda x : K.expand_dims(
                                dot(inputs=[x,class6],axes=0)[:,:,:,0],
                                            axis=1),
                    name='score-class-08')(class8_prob)




    class9_input = Dense(100,activation='relu')(features)
    class9_prob = Dense(3,activation='softmax')(class9_input)
    class9 = Lambda(lambda x : K.expand_dims(
                                    dot(inputs=[x,class2_prob],axes=0)[:,:,:,0],
                                axis=1),
                    name='score-class-09')(class9_prob)



    class10_input = Dense(100,activation='relu')(features)
    class10_prob = Dense(3,activation='softmax')(class10_input)
    class10 = Lambda(lambda x : K.expand_dims(
                                    dot(inputs=[x,class4_prob],axes=0)[:,:,:,0],
                                axis=1),
                    name='score-class-10')(class10_prob)




    main_output = concatenate([class1, class2, class3,  class4,
                               class5, class6, class7,  class8,
                               class9, class10, class11 ],name='Main-output')

    model = Model(main_input,main_output,name='full-model')
    
    return(model)