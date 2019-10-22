
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import load_model
from keras import backend as K
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

model  = load_model('first_try.h5')

test_dir='/mnt/hde/gao/helen/pictur  e/RR/test'


test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(100, 100),
batch_size=20,
class_mode='binary')

labels = (test_generator.class_indices)

test_loss,test_acc = model.evaluate_generator(test_generator, steps=50)
#print (result)
print('test acc:', test_acc)
print('test loss:',test_loss)

Lable =  model.predict_generator(test_generator,steps=50)
print (len(Lable))


filenames = test_generator.filenames
for idx in range(900,930):
    if 	Lable[idx]>0.5:
      print("1")
    else:
      print("0")
    print(Lable[idx])
    print(lables[idx])
    print('title    %s' % filenames[idx])
    print('')
