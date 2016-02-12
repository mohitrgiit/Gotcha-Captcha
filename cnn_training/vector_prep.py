import sys,os
from PIL import Image, ImageFilter
import numpy as np
import h5py

root = "/home/ashar/virtualenvironment/theanoTime/Captcha"
path = os.path.join(root, "extra")

def save_hdf5(m1,m2,m3,m4):
	with h5py.File('training.h5', 'w') as hf:
		hf.create_dataset('X_train', data=m1)
		del m1
		print 'X_train done'

		y0_train = m2[:,0]
		hf.create_dataset('y0_train', data=y0_train)
		del y0_train

		y1_train = m2[:,1]
		hf.create_dataset('y1_train', data=y1_train)
		del y1_train

		y2_train = m2[:,2]
		hf.create_dataset('y2_train', data=y2_train)
		del y2_train

		y3_train = m2[:,3]
		hf.create_dataset('y3_train', data=y3_train)
		del y3_train

		y4_train = m2[:,4]
		hf.create_dataset('y4_train', data=y4_train)
		del y4_train

		y5_train = m2[:,5]
		hf.create_dataset('y5_train', data=y5_train)
		del y5_train

		print 'y_train done'


		hf.create_dataset('X_test', data=m3)
		del m3
		print 'X_test done'

		y0_test = m4[:,0]
		hf.create_dataset('y0_test', data=y0_test)
		del y0_test

		y1_test = m4[:,1]
		hf.create_dataset('y1_test', data=y1_test)
		del y1_test

		y2_test = m4[:,2]
		hf.create_dataset('y2_test', data=y2_test)
		del y2_test

		y3_test = m4[:,3]
		hf.create_dataset('y3_test', data=y3_test)
		del y3_test

		y4_test = m4[:,4]
		hf.create_dataset('y4_test', data=y4_test)
		del y4_test

		y5_test = m4[:,5]
		hf.create_dataset('y5_test', data=y5_test)
		del y5_test

		print 'y_test done'

def prepDataset():
	data={}
	with open('output_extra.txt') as f:
		for i,line in enumerate(f):
			line=line.split(' ')
			line.pop()
			name = line.pop()
			vec = map(int,line)
			vec += [0 for _ in range(5-vec[0])]
			data[name] = vec
	return data

def prepVectors(data):
	nb_train_samples=190000
	nb_test_samples=12356
	c=0
	X_train = np.zeros((nb_train_samples, 3, 50, 50), dtype="uint8")
	X_test = np.zeros((nb_test_samples, 3, 50, 50), dtype="uint8")
	y_train = []
	y_test	= []
	j=0
	for i,item in enumerate(data):
		print i
		img=Image.open(path+'/'+item)
		img=img.resize((50,50),Image.ANTIALIAS)
		arr=np.array(img)
		arr=np.rollaxis(arr,2,0)
		if i>=190000:
			X_test[j]=arr
			j+=1
			y_test.append(data[item])
			continue
		X_train[i]=arr
		y_train.append(data[item])

	y_train=np.array(y_train)
	y_test=np.array(y_test)
	print '\n\n'
	print 'X_train: ' + str(X_train.shape)
	print 'Y_train: ' + str(y_train.shape)
	print '\n'
	print 'X_test: ' + str(X_test.shape)
	print 'Y_test: ' + str(y_test.shape)
	print '\n\n'
	print 'Saving to hdf5 files...'
	save_hdf5(X_train,y_train,X_test,y_test)

print 'Preparing data...'
data = prepDataset()
print 'Data prepared.\nPreparing vectors...'
prepVectors(data)