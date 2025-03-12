"""Original version by Granik et al is accessible at:  https://github.com/AnomDiffDB/DB
Updated version of this code has different number of layers with different filter sizes 
and different dilations (2^n) inspired by p-variation statistical method; Refer to the 
manuscript for more information"""
import sys
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,BatchNormalization,Conv1D,Dropout
from tensorflow.keras.layers import Input,GlobalAveragePooling1D,concatenate
from tensorflow.keras.optimizers import Adam, SGD
from utils2 import generate
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,CSVLogger
import datetime

def create_model_ctrw(steps,filename=None):
    
    if filename==None:
        if not os.path.exists('./models'):
            os.makedirs('../models')
        filename='../models/classification_model_{}_tmp.h5'.format(steps)
        
    batchsize = 32
    T = np.arange(19,21,0.1) # this provides another layer of stochasticity to make the network more robust
    steps = steps # number of steps to generate
    initializer = 'he_normal'
    f = 32 #number of filters
    sigma = 0 #noise variance
    
    tf.random.set_seed(0)
    
    inputs = Input((steps-1,1))
    
    x1 = Conv1D(f,4,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x1)
    x1 = BatchNormalization()(x1)
    x1 = GlobalAveragePooling1D()(x1)
    
    
    x2 = Conv1D(f,5,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x2)
    x2 = BatchNormalization()(x2)
    x2 = GlobalAveragePooling1D()(x2)
    
    
    x3 = Conv1D(f,3,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x3)
    x3 = BatchNormalization()(x3)
    x3 = GlobalAveragePooling1D()(x3)
    
    x4 = Conv1D(f,8,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = Conv1D(f,3,dilation_rate=1,padding='same',activation='relu',kernel_initializer=initializer)(x4)
    x4 = BatchNormalization()(x4)
    x4 = GlobalAveragePooling1D()(x4)
    
    x5 = Conv1D(f,6,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x5 = BatchNormalization()(x5)
    x5 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x5)
    x5 = BatchNormalization()(x5)
    x5 = Conv1D(f,3,dilation_rate=1,padding='same',activation='relu',kernel_initializer=initializer)(x5)
    x5 = BatchNormalization()(x5)
    x5 = GlobalAveragePooling1D()(x5)
    
    x6 = Conv1D(f,3,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x6 = BatchNormalization()(x6)
    x6 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x6)
    x6 = BatchNormalization()(x6)
    x6 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x6)
    x6 = BatchNormalization()(x6)
    x6 = GlobalAveragePooling1D()(x6)
    
    x7 = Conv1D(f,5,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x7 = BatchNormalization()(x7)
    x7 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x7)
    x7 = BatchNormalization()(x7)
    x7 = Conv1D(f,3,dilation_rate=2,padding='same',activation='relu',kernel_initializer=initializer)(x7)
    x7 = BatchNormalization()(x7)
    x7 = GlobalAveragePooling1D()(x7)
    
    x8 = Conv1D(f,7,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x8 = BatchNormalization()(x8)
    x8 = Conv1D(f,3,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x8 = BatchNormalization()(x8)
    x8 = Conv1D(f,3,padding='same',activation='relu',kernel_initializer=initializer)(inputs)
    x8 = BatchNormalization()(x8)
    x8 = GlobalAveragePooling1D()(x8)
    
    con = concatenate([x1,x2,x3,x4,x5,x6,x8])
    # con = Dropout(0.5)(con)
    dense = Dense(2048,activation='relu')(con)
    # dense = Dropout(0.2)(dense)
    dense = Dense(512,activation='relu')(dense)
    dense2 = Dense(3,activation='softmax')(dense)
    model = Model(inputs=inputs, outputs=dense2)
    
    optimizer = Adam(lr=1e-5)
    # optimizer = SGD(learning_rate=1e-4, momentum=0.01, nesterov=True, name='SGD')
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['acc'])
    model.summary()
    
    
    callbacks = [
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.5,
                               patience=4,
                               verbose=1,
                               min_lr=1e-9),
             ModelCheckpoint(filepath=filename,
                             monitor='val_acc',
                             save_best_only=False,
                             mode='max',
                             save_weights_only=False)]
    
    
    gen = generate(batchsize=batchsize,steps=steps,T=T,sigma=sigma)
    model.fit_generator(generator=gen,
            steps_per_epoch=50,
            epochs=100,
            verbose=1,
            callbacks=callbacks,
            validation_data=generate(batchsize=batchsize,steps=steps,T=T,sigma=sigma),
            validation_steps=10)
    print(filename)
