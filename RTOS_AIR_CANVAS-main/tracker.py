import cvzone
from cvzone.ColorModule import ColorFinder
import cv2



cap = cv2.VideoCapture(0)

success, img = cap.read()
h,w,_ = img.shape

myColorFinder = ColorFinder(True)

hsvVals =  {'hmin': 70, 'smin': 67, 'vmin': 51, 'hmax': 88, 'smax': 255, 'vmax': 255}
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) #flip the image horizontally
    imgColor, mask = myColorFinder.update(img, hsvVals)  
    imgContour, contours = cvzone.findContours(img, mask, minArea=1000)
    
    if contours:
        data = contours[0]['center'][0], h-contours[0]['center'][1], int(contours[0]['area'])
        print(data)   

    imgStack = cvzone.stackImages([img, imgColor, mask, imgContour],2,0.5)
    cv2.imshow("Image",imgStack)
    cv2.waitKey(1)
