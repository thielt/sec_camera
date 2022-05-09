import cv2
#library to help drawing process with OpenCV
import time
import datetime

#access the camera where parameters is the index of the video device
cap = cv2.VideoCapture(0)

#classifiers:
#cv2.CascadeClassifier(path)
#https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
eye_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
#pre-trained haarcascades

detection = False
detection_stopped_time = None
timer_started = False

TIME_RECORDING_AFTER_STOP = 2
#Time(seconds) it takes to stop recording after detection has stopped

frame_size = (int(cap.get(3)), int(cap.get(4))) #width, height
fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
#args passes it when saved and appends to end of video name

while True:
    #reading a frame from video device and display
    #_ is a placeholder variable
    _, frame = cap.read()


    #https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters
    #once haarcascade is loaded into memory
    #we can make predictions using detectMultiScale
    #grayscale is needed, converted from frame, shows location/presence in grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #(gray, scalefactor=accuracy and speed of algo should be 1.1-1.5(slower but more accurate the lower), 
    # min neighbors = copies of faces surrounding the actual face to get accuracy(lower# is more faces))
    bodies = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = bodies = face_cascade.detectMultiScale(gray, 1.3, 5)

    #RECORDING LOGIC
    if len(faces) + len(bodies) > 0:
        if detection: #nothing detected
            timer_started = False
        else:
            detection = True #detected
            #start time and recording video if movement detected
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size) #where 20 is the frame rate
            print("Recording...")

    elif detection: #nothing detected and detection = True, turn it off/stop recording
        if timer_started:
            if time.time() - detection_stopped_time >= TIME_RECORDING_AFTER_STOP: #time.time() is the current time
                detection = False
                timer_started = False
                out.release()
                print('STOPPED PROGRAM')
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)
    # ---------------------------------------
    #'''
    #Drawing rectangle on faces in camera
    for (x, y, width, height) in faces:
       cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 3)

    for (x, y, width, height) in bodies:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)

    for (x, y, width, height) in eyes:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)
    #'''

    #Shows output frame
    cv2.imshow("Camera", frame)
    
    #if 'q' is pressed, break from loop
    if cv2.waitKey(1) == ord('q'):
        break

#closing windows opened by OpenCV
#release will take control off the camera so that other programs can use it
out.release()
cap.release()
cv2.destroyAllWindows()