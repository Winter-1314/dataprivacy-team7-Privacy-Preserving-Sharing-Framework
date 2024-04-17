import sys
import os
import cv2

inital_model = cv2.CascadeClassifier('models/haarcascade_cars.xml')


def frames_to_video(frames, path, fps):
    path = path.split("/")
    path = "output/" + path[1] + "/"
    os.makedirs(path, exist_ok=True)

    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(path, f"frame_{i}.jpg"), frame)


#python3 trimmer.py car_stealing.mp4 "color" "timestamps?"

#todo, timestamp the footage so we dont need to pass the filename
#timestamp frames
def trim_footage(file_name, color, interval): 
    if (os.path.exists(file_name)):

        #video data 
        cap = cv2.VideoCapture(file_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        saved_frames = [] #list of frames ided by models 
        while cap.isOpened():
            ret, frame = cap.read()
            if(ret):#if frames 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #scale must be > 1, and lower scaling leads to more false positives 
                cars = inital_model.detectMultiScale(gray, 1.1, 3)#frame, scale, neighbor
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

    
