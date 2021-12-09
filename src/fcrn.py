from keras.preprocessing import image
from keras import backend as K
import numpy as np
from keras.layers import Conv2DTranspose, Conv2D, BatchNormalization, Activation, Concatenate, Input
from keras.layers import Flatten, Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras import optimizers
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator
import csv


# importing from csv file

#########################################################################################################
for j in range(18):
    # csv file name
    filename = "/home/sumeet/Desktop/FCRN/OIRDS_v1_0/DataSet_"+ str(j+1) +  "/dataset_" + str(j+1) + ".csv"



    # filename = "/media/sf_Comp_Vision_Dop/FCRN/OIRDS_v1_0/DataSet_1/dataset_1.csv"

    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row

        # For python2 fields = csvreader.next()
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

        # get total number of rows
        print("Total no. of rows: %d"%(csvreader.line_num))

    # printing the field names
    # print('Field names are:' + ', '.join(field for field in fields))

    #  printing first 5 rows
    # print('\nFirst 5 rows are:\n')
    # for row in rows[:5]:
    #     # parsing each column of a row
    #     for col in row:
    #         print("%10s"%col),
    #     print('\n')




#------------------------------------------------------------------------------------------------------------------

    # Load images from data
    ref = 't'
    for i in range(len(rows)):
        curr = rows[i][2]
        if ref != curr:
            ref = curr
            img_path = '/home/sumeet/Desktop/FCRN/OIRDS_v1_0/DataSet_' + str(j+1) + '/' + rows[i][2]
            lab_path = '/home/sumeet/Desktop/FCRN/OIRDS_v1_0/DataSet_' + str(j+1) + '/gr/' + rows[i][2]

            # img_path = '/media/sf_Comp_Vision_Dop/FCRN/OIRDS_v1_0/DataSet_1/' + rows[i][2]
            # lab_path = '/media/sf_Comp_Vision_Dop/FCRN/OIRDS_v1_0/DataSet_1/gr/' + rows[i][2]
            img = image.load_img(img_path,target_size = (256,256))
            lab = image.load_img(lab_path,target_size = (256,256))
            x_img = image.img_to_array(img)
            x_img = np.expand_dims(x_img,axis=0)
            if i == 0 and j==0:
                print(i,j)
                x = x_img
            else:
                x = np.append(x,x_img,axis=0)
            y_img = image.img_to_array(lab)
            y_img = y_img[:,:,0].reshape(256,256,1)
            y_img = np.expand_dims(y_img,axis=0)
            if i == 0 and j==0:
                y = y_img
            else:
                y = np.append(y,y_img,axis=0)

#---------------------------------------------------------------------------------------------------------------------------
#Flatten the label density map
inputs2 =  Input(shape=(256,256,1))
out = Flatten()(inputs2)
fy = Model(inputs = inputs2, outputs = out)
y_fl = fy.predict(y)

# Alternate flatttening scheme
# y_fl = y.reshape(818,65536)

#split into train, val & val
n_samples = 818
train_size = 698
test_size = 40
val_size = 80
im_size = 256
batch_size = 8

#Seperation into train and validation sets. Subtract mean & std of train data

# x_shaped = x.reshape(n_samples,im_size*im_size*3)
# y_shaped = y.reshape(n_samples,im_size*im_size*1)
x_shaped = x
y_shaped = y_fl

x_shaped = x_shaped/255
# Having ground truth scaled at 255 only
# y_shaped = y_shaped/255;
x_train = x_shaped[:train_size,:,:,:]
y_train = y_shaped[:train_size,:]
train_mean = np.mean(x_train,axis=0)
train_std = np.std(x_train,axis = 0)

x_val = x_shaped[train_size:train_size + val_size,:,:,:]
y_val = y_shaped[train_size:train_size + val_size,:]
x_val -= train_mean
x_val /= train_std

x_test = x_shaped[train_size + val_size:train_size + val_size + test_size,:,:,:]
y_test = y_shaped[train_size + val_size:train_size + val_size + test_size,:]
x_test -= train_mean
x_test /= train_std



#batched image generator(Mean subtraction and normalisation of train data also done)
train_datagen= ImageDataGenerator(featurewise_center = True, featurewise_std_normalization = True, horizontal_flip =True, vertical_flip = True, rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1)
train_datagen.fit(x)
val_datagen= ImageDataGenerator()
val_datagen.fit(x_val)
test_datagen= ImageDataGenerator()
test_datagen.fit(x_test)



model = VGG16(weights='imagenet', input_shape = (256,256,3),  include_top=False)
model.layers.pop()

for layer in model.layers:
    # print(layer.name, 'trainable=', layer.trainable),
    layer.trainable = False
    # print('  after trainable=',layer.trainable)

inputs = model.layers[0].output

c5 = model.layers[17].output
c5 = Conv2DTranspose(512,(3,3),padding='same',strides=(2,2),kernel_initializer='he_uniform')(c5)

c4 = model.layers[13].output
d1 = Conv2D(256,(3,3),padding='same',kernel_initializer='he_uniform')(c4)
d1 = BatchNormalization()(d1)
d1 = Activation('relu')(d1)
d1 = Conv2D(256,(3,3),padding='same',kernel_initializer='he_uniform')(d1)
d1 = BatchNormalization()(d1)
d1 = Activation('relu')(d1)

c3 = model.layers[9].output
d2 = Conv2D(128,(3,3),padding='same',kernel_initializer='he_uniform')(c3)
d2 = BatchNormalization()(d2)
d2 = Activation('relu')(d2)
d2 = Conv2D(128,(3,3),padding='same',kernel_initializer='he_uniform')(d2)
d2 = BatchNormalization()(d2)
d2 = Activation('relu')(d2)

c2 = model.layers[5].output
d3 = Conv2D(64,(3,3),padding='same',kernel_initializer='he_uniform')(c2)
d3 = BatchNormalization()(d3)
d3 = Activation('relu')(d3)
d3 = Conv2D(64,(3,3),padding='same',kernel_initializer='he_uniform')(d3)
d3 = BatchNormalization()(d3)
d3 = Activation('relu')(d3)


deco1 = Concatenate()([d1,c5])
deco1 = Conv2D(256,(3,3),padding='same',kernel_initializer='he_uniform')(deco1)
deco1 = BatchNormalization()(deco1)
deco1 = Activation('relu')(deco1)
deco1 = Conv2D(256,(3,3),padding='same',kernel_initializer='he_uniform')(deco1)
deco1 = BatchNormalization()(deco1)
deco1 = Activation('relu')(deco1)
deco1 = Conv2DTranspose(256,(3,3),padding='same',strides=(2,2),kernel_initializer='he_uniform')(deco1)

deco2 = Concatenate()([d2,deco1])
deco2 = Conv2D(256,(3,3),padding='same',kernel_initializer='he_uniform')(deco2)
deco2 = BatchNormalization()(deco2)
deco2 = Activation('relu')(deco2)
deco2 = Conv2D(256,(3,3),padding='same',kernel_initializer='he_uniform')(deco2)
deco2 = BatchNormalization()(deco2)
deco2 = Activation('relu')(deco2)
deco2 = Conv2DTranspose(256,(3,3),padding='same',strides=(2,2),kernel_initializer='he_uniform')(deco2)

deco3 = Concatenate()([d3,deco2])
deco3 = Conv2D(256,(3,3),padding='same',kernel_initializer='he_uniform')(deco3)
deco3 = BatchNormalization()(deco3)
deco3 = Activation('relu')(deco3)
deco3 = Conv2D(256,(3,3),padding='same',kernel_initializer='he_uniform')(deco3)
deco3 = BatchNormalization()(deco3)
deco3 = Activation('relu')(deco3)
deco3 = Conv2DTranspose(256,(3,3),padding='same',strides=(2,2),kernel_initializer='he_uniform')(deco3)

deco4 = Conv2D(256,(3,3),padding='same',kernel_initializer='he_uniform')(deco3)
deco4 = BatchNormalization()(deco4)
deco4 = Activation('relu')(deco4)
deco4 = Conv2D(256,(3,3),padding='same',kernel_initializer='he_uniform')(deco4)
deco4 = BatchNormalization()(deco4)
deco4 = Activation('relu')(deco4)

deco5 = Conv2D(1,(1,1),padding='same',kernel_initializer='he_uniform')(deco4)
deco5 = Activation('linear')(deco5)
deco5 = Flatten()(deco5)
deco5 = Lambda(lambda x: x*255)(deco5)

fcrn = Model(inputs= inputs ,outputs = deco5)



#Compile and train
# def cust_mean_squared_error(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)


lr_rate = 0.01
decay_rate = lr_rate/200;
epochs = 1

#Custom loss fn
def clipped_mse(y_true, y_pred):
    return K.mean(K.square(K.clip(y_pred, 0., 255.) - K.clip(y_true, 0., 255.)), axis=-1)


rmsprop = optimizers.RMSprop(lr=lr_rate, rho=0.9, epsilon=None, decay=decay_rate)

fcrn.compile(optimizer='rmsprop',loss=clipped_mse)
f=fcrn.fit_generator(train_datagen.flow(x_train,y_train,batch_size=batch_size),steps_per_epoch=len(x_train)/batch_size,epochs=epochs,validation_data=val_datagen.flow(x_val,y_val,batch_size=batch_size),validation_steps=len(x_val)/batch_size)

#Train & test after every epoch
test = x_test[2,:,:,:].reshape(1,256,256,3)
for i in range(10):
    epochs = i + 1
    fcrn.fit_generator(train_datagen.flow(x_train,y_train,batch_size=batch_size),steps_per_epoch=len(x_train)/batch_size,epochs=1,validation_data=val_datagen.flow(x_val,y_val,batch_size=batch_size),validation_steps=len(x_val)/batch_size)
    # Image testing
    pred = fcrn.predict(test,batch_size=1)
    pred_img = image.array_to_img(np.clip(pred.reshape(256,256,1),0,255))
    pred_img.save('tr/ep' + str(epochs)+'.jpg')


# Image testing
test = x_test[1,:,:,:].reshape(1,256,256,3)
pred = fcrn.predict(test,batch_size=1)
pred_img = image.array_to_img(np.clip(pred.reshape(256,256,1),0,255))

test_img = image.array_to_img(test.reshape(256,256,3))
test_gr_img = image.array_to_img(y_test[1,:].reshape(256,256,1))

