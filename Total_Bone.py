from os import listdir
import random
from os.path import isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
mypath = '/Users/jonathanmairena/Documents/Fall2019/MachineLearning/microCT/mritopng-master/00026159_png'

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) , 0)
  
dpi = 613

images[67] = images[66]
areas_bone= []
area_mm = []
for i,original in enumerate(images):
    #Pyramid Down Sample -- ALGORITHM
    img1 = cv2.pyrDown(original)
    
    #Threshhold
    ret,binary = cv2.threshold(img1,60,255,cv2.THRESH_BINARY)
    
    #FIND CONTOURS
    contours, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #DRAW MINIMUM ENCLOSING RECTANGLE WITH ROTATION  -- ALGORITHM   
    minimum_area = [] 
    areamm = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img1,[box],0,(0,0,255),2)
        rectarea = rect[1][0] * rect[1][1]
        minimum_area.append(rectarea)
        areamm.append((rect[1][0])/dpi * (rect[1][1])/dpi)
        
    area_mm.append(sum(areamm))
    areas_bone.append(sum(minimum_area))
    print("Image: ", i , "-- Area of Bone" , area_mm[i]*645 , " mm^2")
    
idx = random.randint(0,len(images))
plt.figure()
plt.imshow(images[idx],cmap = 'gray')
plt.title("Image: "+ str(idx) + " Area: " + str(area_mm[idx]*645) + " mm^2")