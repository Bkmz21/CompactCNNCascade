# -*- coding: utf-8 -*-

import os 
import cv2

class_path = ['train_data_example/background/', 
              'train_data_example/face_frontal/', 
              'train_data_example/face_profile/']

fout = 'train_data.txt'
			  
train_img_list = open(fout, 'w')    
for i, src_path in enumerate(class_path):
	dst_path = src_path[0:-1] + '_gray/'
	if not os.path.exists(dst_path):
		os.makedirs(dst_path)

	files = os.listdir(src_path)
	files = [ file for file in files if file.endswith( ('.jpg','.png','.bmp','.tif') ) ]

	for j, file_name in enumerate(files):
		img = cv2.imread(src_path + '/' + file_name)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img[:,:,0] = img_gray
		img[:,:,1] = img_gray
		img[:,:,2] = img_gray
		cv2.imwrite(dst_path + file_name, img)

		val = 0
		if i >= 1:
			val = 1
		if i >= 2:
			val = 2         

		train_img_list.write(dst_path + file_name + '	' + str(val) + '\n')      
train_img_list.close()
