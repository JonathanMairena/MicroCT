import cv2
import numpy as np
import matplotlib.pyplot as plt

#READ IMAGE
original = cv2.imread('C0024205_00004.DCM.png', cv2.IMREAD_UNCHANGED)
img1 = cv2.pyrDown(original)
#DPI is 613 Use to calculate area in inches later divide by  pixels to find how many inches
dpi = 613
#Threshhold   CHANGE TO 60, 255 to get the bone
ret,binary = cv2.threshold(img1,30,255,cv2.THRESH_BINARY)

#Blur
dst = cv2.GaussianBlur(binary,(5,5),cv2.BORDER_DEFAULT)

# histogram Equilization   -- ALGORITHM
equ = cv2.equalizeHist(dst) 

#EdgeDetection   -- ALGORITHM
edges = cv2.Canny(equ,0,255) 

#Another Blur
dst2  =  cv2.GaussianBlur(edges,(9,9),cv2.BORDER_DEFAULT)

#PLOTTING EDGE DETECTION
plt.figure(1)
plt.subplot(131),plt.imshow(original,cmap = 'gray')
plt.subplot(132),plt.imshow(edges,cmap = 'gray')
plt.subplot(133),plt.imshow(dst2,cmap ='gray')
plt.show()

#FIND CONTOURS
contours, hier = cv2.findContours(dst2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#DRAW MINIMUM ENCLOSING RECTANGLE WITH ROTATION  -- ALGORITHM    
rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img1,[box],0,(0,0,255),2)
rectarea = rect[1][0] * rect[1][1]
rect_inches = (rect[1][0])/dpi * (rect[1][1])/dpi

#DRAW MINIMUM ENCLOSING CIRCLE and calculate area  -- ALGORITHM    
(x,y),radius = cv2.minEnclosingCircle(contours[0])
center = (int(x),int(y))
radius = int(radius)
circlearea = (radius**2) * 3.14
circle_inches = ((radius/dpi)**2) * 3.14

#PICK THE SMALLEST AREA
if(circlearea  > rectarea):
    minimumarea = rectarea
    inch_area = rect_inches
else: 
    minimumarea = circlearea
    inch_area = circle_inches
    
plt.figure(2)
plt.contour(np.flip(img1, axis = 0))
plt.xlim([0,400])
plt.ylim([0,400])
plt.show()
print("Tissue Area: " , inch_area*645, " mm^2")

##################################################################################
################################ BONE ############################################
##################################################################################

img2 = cv2.pyrDown(original)
#DPI is 613 Use to calculate area in inches later divide by  pixels to find how many inches
dpi = 613
#Threshhold   CHANGE TO 60, 255 to get the bone
ret2,binary2 = cv2.threshold(img2,60,255,cv2.THRESH_BINARY)

#Blur
dst2 = cv2.GaussianBlur(binary2,(5,5),cv2.BORDER_DEFAULT)

# histogram Equilization   -- ALGORITHM
equ2 = cv2.equalizeHist(dst2) 

#EdgeDetection   -- ALGORITHM
edges2 = cv2.Canny(equ2,0,255) 

#Another Blur
dst3  =  cv2.GaussianBlur(edges2,(9,9),cv2.BORDER_DEFAULT)

#PLOTTING EDGE DETECTION
plt.figure(3)
plt.subplot(131),plt.imshow(original,cmap = 'gray')
plt.subplot(132),plt.imshow(edges2,cmap = 'gray')
plt.subplot(133),plt.imshow(dst3,cmap ='gray')
plt.show()
#FIND CONTOURS
contours, hier = cv2.findContours(dst3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#DRAW MINIMUM ENCLOSING RECTANGLE WITH ROTATION  -- ALGORITHM    
rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img2,[box],0,(0,0,255),2)
rectarea = rect[1][0] * rect[1][1]
rect_inches = (rect[1][0])/dpi * (rect[1][1])/dpi

#DRAW MINIMUM ENCLOSING CIRCLE and calculate area  -- ALGORITHM    
(x,y),radius = cv2.minEnclosingCircle(contours[0])
center = (int(x),int(y))
radius = int(radius)
circlearea = (radius**2) * 3.14
circle_inches = ((radius/dpi)**2) * 3.14

#PICK THE SMALLEST AREA
if(circlearea  > rectarea):
    minimumarea = rectarea
    inch_area = rect_inches
else: 
    minimumarea = circlearea
    inch_area = circle_inches
    
plt.figure(4)
plt.contour(np.flip(img2, axis = 0))
plt.xlim([0,400])
plt.ylim([0,400])
plt.show()
print("Bone Area: " , inch_area*645, " mm^2")

