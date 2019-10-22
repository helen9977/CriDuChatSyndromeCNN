
from keras.models import load_model
import numpy as np
from PIL import Image
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


pwd_right_test ='/mnt/hde/gao/helen/picture/RR/test'

pwd_left_test ='/mnt/hde/gao/helen/picture/LL/test'

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
model_left  = load_model('left_cnn10_20.h5')
model_right  = load_model('right_cnn10_20.h5')

labels_left,images_left = data(pwd_left_test)
labels_right,images_right  = data(pwd_right_test)

result_left= model_left.predict_classes(images_left)
result_right= model_right.predict_classes(images_right)


j_0_0 = 0
j_0_1 = 0
j_1_0 = 0
j_1_1 = 0
j_T_F = 0
j_F_T = 0

for ele in zip(result_left,result_right,labels_left):
    if ele[0]== ele[1] and ele[1]==ele[2]:
        if ele[2]==0:
            j_0_0 +=1
        elif ele[2]==1:
            j_1_1 +=1
    elif ele[0]==ele[1] and (not ele[1]==ele[2]):
        if ele[2]==0 and ele[0]==1: 
            j_0_1+=1
        if ele[2]==1 and ele[0]==0:
            j_1_0+=1
    elif ele[0]!=ele[1]:
    	if ele[0]==0:
    	    j_T_F+=1
    	elif ele[0]==1:
    	    j_F_T_=1


print("judge 0 as 0 = " + str(j_0_0))
print("judge 0 as 1 = " + str(j_0_1))
print("judge 1 as 0 = " + str(j_1_0))
print("judge 1 as 1 = " + str(j_1_1))

print("left judge 0 while right judge 1 = " + str(j_T_F))
print("left judge 1 while right judge 0 = " + str(j_F_T))

print("acc=" + str(( j_0_0+j_1_1) / 1212.))

