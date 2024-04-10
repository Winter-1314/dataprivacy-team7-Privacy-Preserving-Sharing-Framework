#import libraries of python opencv
import cv2
import sys
import datetime #used for fomatting timestamps

#uncomment once ready to handle full inputs

# if (len(sys.argv) !=4):
#     print("3 arguments required \n format should be python .\car_detection.py starttimestamp endtimestamp carcolor")
    
# else:

#getting time stamps to look at video between these times


# capture video/ video path
cap = cv2.VideoCapture('media/20090-307115630_small.mp4') #name of video you want to look at

#get fps of video used to calculate timestamp
fps = cap.get(cv2.CAP_PROP_FPS)

#use trained cars XML classifiers
car_cascade = cv2.CascadeClassifier('models/haarcascade_cars.xml')


#read until video is completed
while True:
    #capture frame by frame
    ret, frame = cap.read()
    #convert video into gray scale of each frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect cars in the video
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)
    #cv2.im_write(cars)

    ####################added code section for creating timestamps#######################

    #for calculating time stamp
    timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

    #format timestamps
    timestamps_formatted = []
    timestamp_sec = timestamp_msec / 1000.0
    timestamp_datetime = datetime.datetime.fromtimestamp(timestamp_sec)
    timestamps_formatted.append(timestamp_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"))
    
    
    ###########prints time stamps for each frame for testing
    print(timestamps_formatted[0]) 
    

    ####### another option for calculating time stamps you will want to change current_time to the actual timestamp of the video########
    # timestamp_msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    # timestamp_sec = timestamp_msec / 1000.0

    # current_time = datetime.datetime.now()
    # actual_timestamp = current_time - datetime.timedelta(seconds=timestamp_sec)
    # formatted_timestamp = actual_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
    # print(f"Actual Timestamp: {formatted_timestamp}")

    
    #to draw a rectangle in each cars 
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('video', frame)
        crop_img = frame[y:y+h,x:x+w]
    
    
    #########################################end of edited section##########################
    
    #press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break



#release the video-capture object
cap.release()
#close all the frames
cv2.destroyAllWindows()
