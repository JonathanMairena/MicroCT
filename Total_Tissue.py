from os import listdir
from os.path import isfile, join
import cv2
import numpy as np



mypath= '/Users/jonathanmairena/Documents/Fall2019/MachineLearning/microCT/mritopng-master/00026159_png'

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) , 0)
  
dpi = 613

images[67] = images[66]
rect_areas = []
circle_area = []
areas_tissue = []
rect_inches = []
circle_inches = []
for i,original in enumerate(images):
    
    #READ IMAGE
    img1 = cv2.pyrDown(original)
    
    #Threshhold
    ret,binary = cv2.threshold(img1,30,255,cv2.THRESH_BINARY)
    
    #Blur
    dst = cv2.GaussianBlur(binary,(5,5),cv2.BORDER_DEFAULT)
    
    # histogram Equilization   -- ALGORITHM
    equ = cv2.equalizeHist(dst) 
    
    #EdgeDetection   -- ALGORITHM
    edges = cv2.Canny(dst,0,255) 
    
    #Another Blur
    dst2  =  cv2.GaussianBlur(edges,(9,9),cv2.BORDER_DEFAULT)
    
    #FIND CONTOURS
    contours, hier = cv2.findContours(dst2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #DRAW MINIMUM ENCLOSING RECTANGLE WITH ROTATION and claculate area-- ALGORITHM    
    rect = cv2.minAreaRect(contours[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img1,[box],0,(0,0,255),2)
    rectarea = rect[1][0] * rect[1][1]
    rectinches = (rect[1][0])/dpi * (rect[1][1])/dpi
    rect_inches.append(rectinches)
    rect_areas.append(rectarea)
    
    #DRAW MINIMUM ENCLOSING CIRCLE and calculate area  -- ALGORITHM    
    (x,y),radius = cv2.minEnclosingCircle(contours[0])
    center = (int(x),int(y))
    radius = int(radius)
    circlearea = (radius**2) * 3.14
    circleinches = ((radius/dpi)**2) * 3.14
    circle_inches.append(circleinches)
    circle_area.append(circlearea)
    
    #PICK THE SMALLEST AREA
    if(circlearea  > rectarea):
        minimumarea = rectarea
        incharea  = rectinches
        areas_tissue.append(rectarea)
    else: 
        minimumarea = circlearea
        incharea = circleinches
        areas_tissue.append(circlearea)
    
    print("Image: " , i , "-- Minimum Area: ", incharea*645, " mm^2" )
    
    
    

