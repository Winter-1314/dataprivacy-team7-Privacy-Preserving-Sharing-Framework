import sys
import os
import cv2
import datetime #used for formatting timestamps

inital_model = cv2.CascadeClassifier('models/haarcascade_cars.xml')


def frames_to_video(frames, path, fps):
    path = path.split("/")
    path = "output/" + path[1] + "/"
    os.makedirs(path, exist_ok=True)

    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(path, f"frame_{i}.jpg"), frame)


#python3 trimmer.py car_stealing.mp4 "color" "timestamps?"

#timestamp frames
def trim_footage(file_name, color, interval): 
    if (os.path.exists(file_name)):

        #video data 
        cap = cv2.VideoCapture(file_name)
        fps = cap.get(cv2.CAP_PROP_FPS) # get fps of frames
        saved_frames = [] #list of frames ided by models 

        #format input from command line for time stamp
        for i in range(2):
            if i == 0:
                start_timestamp = sys.argv[1]
            else:
                end_timestamp = sys.argv[3]

            # Split the string into parts (year, month, day, hour, minute, second)
            if(len(sys.argv[1].split("-")) == 6 and len(sys.argv[3].split("-")) == 6):
                if i == 0:
                    year, month, day, hour, minute, second = start_timestamp.split("-")
                    second = second.split(".")[0] #removes the mp4 from file name
                    # Create a datetime object
                    start_timestamp = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))

                    # Print the timestamp in a desired format (e.g., ISO 8601)
                    print(start_timestamp)

                else:
                    year, month, day, hour, minute, second = end_timestamp.split("-")
                    second = second.split(".")[0] #removes the mp4 from file name
                    # Create a datetime object
                    end_timestamp = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))

                    # Print the timestamp in a desired format (e.g., ISO 8601)
                    print(end_timestamp)
            else:
                print("invalid timestamp format")

        #initilize timestamp to basically zero for comparisons
        actual_timestamp = datetime.datetime(1960,1,1,0,0,0,0)

        while cap.isOpened() and actual_timestamp < end_timestamp:
            ret, frame = cap.read()
            if(ret):#if frames 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #scale must be > 1, and lower scaling leads to more false positives 
                cars = inital_model.detectMultiScale(gray, 1.1, 3)#frame, scale, neighbor


                ####### another option for calculating time stamps########
                
                # #for calculating time stamp
                timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_sec = timestamp_msec / 1000.0

                # get actual time stamp building off of input from user for the start and stop time
                # provided by argv[1] and argv[3] above into vars start_timestamp and stop_timestamp
                actual_timestamp = start_timestamp + datetime.timedelta(seconds=timestamp_sec)
                formatted_timestamp = actual_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
                print("Actual Timestamp::", formatted_timestamp)
                
                
                if(len(cars) != 0):
                    #cars detected, pass to next model with key words

                    if(True):
                        saved_frames.append(frame)
                        #save to file, append to video?
                        
            else:
                print("Trimming complete")
                cap.release()
            
            #write frames to file   
        if(len(saved_frames) > 1):
            frames_to_video(saved_frames, file_name, fps)

    else:
        print("no media file")

if __name__ == "__main__":
    if len(sys.argv) == 4:
        path = "media/" + sys.argv[1]
        print(path)
        trim_footage(path,sys.argv[2], sys.argv[3]  )
    else:
        print("using preset")
        trim_footage("media/deliveryDriver.mp4", "red", "timestamps"  )

    
