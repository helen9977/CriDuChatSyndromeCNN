
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
import h5py
import time
from keras import backend as K
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 100, 100

left_train_data_dir = '/mnt/hde/gao/helen/picture/LL/train'
left_validation_data_dir = '/mnt/hde/gao/helen/picture/LL/validation'
right_train_data_dir = '/mnt/hde/gao/helen/picture/RR/train'
right_validation_data_dir = '/mnt/hde/gao/helen/picture/RR/validation'

nb_train_samples = 3642
nb_validation_samples = 1214
epochs = 20
batch_size = 20


if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type,fig_name):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig(fig_name+epochs+'_'+batch_size+'.jpg')
        print('savefig',fig_name)
        plt.close()



def smooth_curve(points, factor=0.8):
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous * factor + point * (1 - factor))
		else:
			smoothed_points.append(point)
	return smoothed_points



def layers(model)

	model.add(Conv2D(32, (3, 3), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.summary()

	model.compile(loss='binary_crossentropy',
		          optimizer='rmsprop',
		          metrics=['accuracy'])
	return model

def FitModel(model,train_data_dir,validation_data_dir,name,history):
		# this is the augmentation configuration we will use for training
	train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,)
	# this is the augmentation configuration we will use for testing:
	# only rescaling
	test_datagen = ImageDataGenerator(rescale=1. / 255)


	train_generator = train_datagen.flow_from_directory(
		train_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='binary')

	validation_generator = test_datagen.flow_from_directory(
		validation_data_dir,
		target_size=(img_width, img_height),
	 	batch_size=batch_size,
		class_mode='binary')

	model.fit_generator(
		train_generator,
		steps_per_epoch=nb_train_samples // batch_size,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=nb_validation_samples // batch_size,
		callbacks=[history])

	model.save(name+epochs+'_'+batch_size+'.h5')

	print('save model',name)


#-------train-------

left_model_name='left_cnn'
right_model_name='right_cnn'

print('Building left model....')
model_left = Sequential()
model_left = layers(model_left)

print('Building right model...')
model_right = Sequential()
model_right = layers(model_right)

history_1 = LossHistory()
history_2 = LossHistory()

t1=time.time()
FitModel(model_left,left_train_data_dir,left_validation_data_dir,left_model_name,history_1)
print('left model is finished...')
FitModel(model_right,right_train_data_dir,right_validation_data_dir,right_model_name,history_2)
print('right model is finished...')
t2=time.time()

print ("training data uses time: "+str(t2-t1))

history_1.loss_plot('epoch',left_model_name)
history_2.loss_plot('epoch',right_model_name)





