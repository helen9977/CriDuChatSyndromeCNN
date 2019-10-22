
from keras.models import load_model
import numpy as np
from PIL import Image
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


pwd_right_test ='/mnt/hde/gao/helen/picture/RR/test'

pwd_left_test ='/mnt/hde/gao/helen/picture/LL/test'

nb_test_samples = 1212

def data(pwd):
	images=[]
	labels=[]

	def read_image(name):
		img = np.array(load_img(name, target_size=(100,100)))
		data = np.array(img, dtype="float") / 255.0
		return data

	text = os.listdir(pwd) # 0 1

	for textpath in text:
		for fn in os.listdir(os.path.join(pwd,textpath)): # test/0  test/1
			if fn.endswith('.png'):
				fd = os.path.join(pwd,textpath,fn)
				images.append(read_image(fd))
				labels.append(textpath)

	labels=np.array(list(map(int, labels)))
	images=np.array(images)

	return labels,images


#--- test-----
model_left  = load_model('left_cnn.h5')
model_right  = load_model('right_cnn.h5')

labels_left,images_left = data(pwd_left_test)
labels_right,images_right  = data(pwd_right_test)

result_left= model_left.predict_classes(images_left)
result_right= model_right.predict_classes(images_right)



i = 532

print("Test PNG are 1212...")
print("This PNG is "+str(i)+'th. Check it in helen/picture/.')

if result_right[i] == 1 and  result_left[i] == 1:
	print("This gene graph is recognized as non-5p-deletion,the real label is "+str( labels_right[i]))
elif result_right[i] == 0 and result_left[i] == 0:
	print("This gene graph is recognized as 5p-deletion,the real label is "+ str(labels_right[i]))
else:
	print("MISS!!!")
