import numpy as np
import time
import os
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
from predict_shapes import predict_shape



########################
brushThickness = 7
eraserThickness = 50
movementThreshold = 5  # Adjust the threshold for movement
########################

folderPath = "header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (0, 0, 255) #red


cap = cv2.VideoCapture(0)
cap.set(3, 653)
cap.set(4, 600)
success, img = cap.read()
h,w,_ = img.shape
myColorFinder = ColorFinder(False)

# hsvVals =  {'hmin': 75, 'smin': 55, 'vmin': 0, 'hmax': 90, 'smax': 196, 'vmax': 255} #n pens
# hsvVals = {'hmin': 48, 'smin': 97, 'vmin': 37, 'hmax': 92, 'smax': 255, 'vmax': 255}   #phone camera
# hsvVals = {'hmin': 45, 'smin': 227, 'vmin': 86, 'hmax': 91, 'smax': 255, 'vmax': 255}  #webcam
hsvVals =  {'hmin': 70, 'smin': 67, 'vmin': 51, 'hmax': 88, 'smax': 255, 'vmax': 255}
# hsvVals = {'hmin': 80, 'smin': 77, 'vmin': 105, 'hmax': 179, 'smax': 236, 'vmax': 255} #webcam


center_x, center_y = 0, 0
xp, yp = 0, 0
imgCanvas = np.zeros((600,653,3), np.uint8)

tracking_enabled = False  # Start/stop tracking flag

stabilization_enabled = False  # Start/stop video stabilization flag

# Initialize Kalman filter parameters
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1e-6, 0, 0, 0],
                                   [0, 1e-6, 0, 0],
                                   [0, 0, 1e-6, 0],
                                   [0, 0, 0, 1e-6]], np.float32)
kalman.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
kalman.statePre = np.zeros((4, 1), dtype=np.float32)
kalman.statePost = np.zeros((4, 1), dtype=np.float32)


def clear_canvas():
    global imgCanvas
    imgCanvas = np.zeros((600,653,3),np.uint8)
    # imgCanvas = np.full((600,653,3), (255, 255, 255), np.uint8)

# Calculate the dimensions of the rectangle and position it in the bottom-right corner
rectWidth, rectHeight = 130, 50
rectX = img.shape[1] - rectWidth - 20  # 10 pixels offset from the right edge
rectY = img.shape[0] - rectHeight - 10  # 10 pixels offset from the bottom edge

# Set the position of the text inside the rectangle
textOffsetX = rectX + 2
textOffsetY = rectY + 14

drawing_mode = False
show_text_timer = 0
show_text_duration = 3  # Display text for 3 seconds

while True:
    #1. import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    # 2. find object
    if tracking_enabled:
        imgColor, mask = myColorFinder.update(img, hsvVals)  
        img, contours = cvzone.findContours(img, mask, minArea=500)
    
    else:
        contours = []

    # clear all button
    buttonColor = (235, 204, 168)
    buttonText = "Clear All"
    buttonSize = cv2.getTextSize(buttonText, cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
    buttonX, buttonY = 35, img.shape[0] - 30

    # Check if the "Clear All" button is clicked
    if center_y > buttonY - buttonSize[1] - 10 and buttonX - 10 < center_x < buttonX + buttonSize[0] + 10 and buttonY + 10 > center_y > buttonY - buttonSize[1] - 10:
        clear_canvas()
        # Change the button color to a brighter color
        buttonColor = (255, 255, 153)
    else:
        # Reset the button color to its original color
        buttonColor = (235, 204, 168)

    # Draw the button with the buttonColor variable
    cv2.rectangle(img, (buttonX - 10, buttonY - buttonSize[1] - 10), (buttonX + buttonSize[0] + 10, buttonY + 10), buttonColor, cv2.FILLED)
    cv2.putText(img, buttonText, (buttonX, buttonY - 1), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

    ###################################################################################################################
    #shortcut box
    # Draw the rectangle
    cv2.rectangle(img, (rectX, rectY), (rectX + rectWidth, rectY + rectHeight), (255, 255, 255, 500), cv2.FILLED)
    
    # Choose font properties for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1

    # Add the text inside the rectangle
    cv2.putText(img, "d: Draw on/off", (textOffsetX, textOffsetY), font, fontScale, (0, 0, 0), fontThickness)
    cv2.putText(img, "p: Predict", (textOffsetX, textOffsetY + 15), font, fontScale, (0, 0, 0), fontThickness)
    cv2.putText(img, "q: Quit", (textOffsetX, textOffsetY + 30), font, fontScale, (0, 0, 0), fontThickness)
    #####################################################################################################################

    #3. selection mode
    if contours:

        data = contours[0]['center'][0], h-contours[0]['center'][1], int(contours[0]['area'])
        # print(data) 

        center_x, center_y = contours[0]['center'] 
        # print(contours)
        # xp, yp = 0, 0

        if center_y < 100:
            if 50<center_x<150:
                header = overlayList[0]
                drawColor = (0, 0, 255) #red
            elif 170 < center_x < 270:
                header = overlayList[1]
                drawColor = (255, 0, 0) #blue
            elif 290 < center_x < 390:
                header = overlayList[2]
                drawColor = (0, 255, 0) #green
            elif 410 < center_x < 510:
                header = overlayList[3]
                drawColor = (0, 0, 0) #eraser
        
    else:
        center_x, center_y = 0, 0
        xp, yp = 0, 0


    #4. drawing mode
    if xp==0 and yp==0:
        xp, yp = center_x, center_y

    if stabilization_enabled:
        # Kalman filter prediction
        kalman_prediction = kalman.predict()
        kalman_x, kalman_y = kalman_prediction[0], kalman_prediction[1]

        # Update measurement
        kalman_measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])
        kalman.correct(kalman_measurement)

        # Get corrected state
        kalman_state = kalman.statePost
        center_x, center_y = kalman_state[0], kalman_state[1]
    
    
    # Check if the movement is below the threshold
    if abs(center_x - xp) < movementThreshold and abs(center_y - yp) < movementThreshold:
        # No movement, do not update the drawing position
        pass
    # Check if the header is being displayed
    if center_y < 100:
    # Skip drawing on imgCanvas if paint brushes are being selected
        pass
    
    else:
        if drawColor == (0,0,0):
            cv2.line(img, (xp, yp),(center_x,center_y),drawColor,eraserThickness)
            cv2.line(imgCanvas, (xp, yp),(center_x,center_y),drawColor,eraserThickness)
        else:
            cv2.line(img, (xp, yp),(center_x,center_y),drawColor,brushThickness)
            cv2.line(imgCanvas, (xp, yp),(center_x,center_y),drawColor,brushThickness)

    xp,yp = center_x, center_y

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    imgInv = cv2.resize(imgInv, (img.shape[1], img.shape[0]))
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.resize(img, (imgCanvas.shape[1], imgCanvas.shape[0]))
    img = cv2.bitwise_or(img,imgCanvas)

    key = cv2.waitKey(1) & 0xFF
    

    if not os.path.exists('saved'):
        os.makedirs('saved')

    if key == ord('d'):
        tracking_enabled = not tracking_enabled  # Toggle tracking on/off   
        show_text_timer = time.time()

        if tracking_enabled:
            drawing_mode = False
        else:
            drawing_mode = True

    if tracking_enabled:
        drawing_mode = True
        if time.time() - show_text_timer < show_text_duration:
            cv2.putText(img, "Drawing mode On", (230, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    else:
        drawing_mode = False
        if time.time() - show_text_timer < show_text_duration:
            cv2.putText(img, "Drawing mode Off", (230, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


    if key == ord('p'):
        cv2.imwrite(f'saved/saved.png', imgCanvas)
    
        
        image='saved/saved.png'
        # predict(image)

        predicted_label, highest_probability = predict_shape(image)
        
        # Draw the predicted label on the canvas
        if predicted_label is None:
            cv2.putText(imgCanvas, "Sorry, no shapes detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (135, 206, 235), 2)
        else:
            text = f"{predicted_label} (Prob: {highest_probability:.2f})"
            cv2.putText(imgCanvas, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (135, 206, 235), 2)

        # if predicted_label is None:
        #     cv2.putText(imgCanvas, "Sorry, no shapes detected", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (135, 206, 235), 2)
        # else:
        #     cv2.putText(imgCanvas, predicted_label, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (135, 206, 235), 2)

    
    if key == ord('q'):
        break

#setting the header image
    header = cv2.resize(header, (653, 90))
    img[0:90, 0:653] = header
    imgStack = cv2.hconcat([img , imgCanvas])
    # cv2.imshow("Image",img)
    cv2.imshow("Image",imgStack)
    # cv2.imshow("Canvas",imgCanvas)
    # cv2.imshow("Image",imgContour)



cap.release()
cv2.destroyAllWindows()
