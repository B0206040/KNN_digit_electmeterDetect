import numpy as np
import os
import cv2

# OPEN TRAINING IMAGE FOR PROCESSING------------------------------------------------------------------------------------
samples =  np.empty((0, 100))
responses = []

def sharpen(img, sigma):    
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img, (0, 0), sigma)
    usm = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)
    
    return usm

Num = ['9','8','7','6','5','4','3','2','1','0','11']
for N_P in Num:
    tmp = os.listdir("./samples/" + N_P + "/") 
    for file in tmp:
        filename = "./samples/" + N_P + "/" + file
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY ) 
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
        cv2.imwrite(('mean_results/'+ N_P + "/" + file),th1)

        roi_small = cv2.resize(th1, (10, 10))
        key = int(N_P)
        sample = roi_small.reshape((1,100))
        samples = np.append(samples,sample,0)
        responses.append(key)

print ("training complete")
np.savetxt('data/general_samples.data', samples)
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size,1))
np.savetxt('data/general_responses.data', responses)
