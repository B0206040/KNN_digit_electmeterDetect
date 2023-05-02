import cv2
import numpy as np
import sys
import os


def sharpen(img, sigma):    
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    
    return usm

samples = np.loadtxt(r"D:\0216_photo\data\general_samples.data", np.float32)
responses = np.loadtxt(r"D:\0216_photo\data\general_responses.data", np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples=samples, layout=cv2.ml.ROW_SAMPLE, responses=responses)

image = cv2.imread('2_5_1110_154444.jpg')


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY ) 
#print(image.shape)
image = sharpen(image, 100)
kernel = np.ones((3,3), np.uint8)
        
th1 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
th1 = cv2.blur(th1,(5,5))
th1 = cv2.erode(th1, kernel, iterations = 1)
th1 = cv2.dilate(th1, kernel, iterations = 1)
th1 = cv2.blur(th1,(5,5))
th1 = cv2.erode(th1, kernel, iterations = 1)
th1 = cv2.dilate(th1, kernel, iterations = 1)
th1 = sharpen(th1,100)
th1 = cv2.dilate(th1, kernel, iterations = 1)
th1 = cv2.erode(th1, kernel, iterations = 1)
th1 = cv2.blur(th1,(5,5))
th1 = cv2.dilate(th1, kernel, iterations = 1)
th1 = cv2.erode(th1, kernel, iterations = 1)
th1 = sharpen(th1,100)
cv2.imshow('sharpen', th1)
cv2.waitKey(0)

image = cv2.resize(th1,(10,10))
image = image.reshape((1,100))
image = np.float32(image)

retval, results, neigh_resp, dists = model.findNearest(image, k= 5)
print('results = ', results)
print('neigh_resp = ', neigh_resp)
