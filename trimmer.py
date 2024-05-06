import sys
import os
import cv2
import datetime #used for formatting timestamps

import torch
from fastsam import FastSAM, FastSAMPrompt

inital_model = cv2.CascadeClassifier('.weights/haarcascade_cars.xml')


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
FastSAM_model = FastSAM('.weights/FastSAM-s.pt')


#saves frames to images stored in output/filename 
def frames_to_images(frames, path, fps):
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

        #####formatting for timestamps#####
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



        #initilize timestamp to zero equivalent for comparison
        actual_timestamp = datetime.datetime(1960,1,1,0,0,0,0)

        while cap.isOpened() and actual_timestamp < end_timestamp:
            success, frame = cap.read()
            if(success):#if frames 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #scale must be > 1, and lower scaling leads to more false positives 
                cars = inital_model.detectMultiScale(gray, 1.1, 3)#frame, scale, neighbor
                
                # #for calculating time stamp
                timestamp_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_sec = timestamp_msec / 1000.0

                # get actual time stamp building off of input from user for the start and stop time
                # provided by argv[1] and argv[3] above into vars start_timestamp and stop_timestamp
                actual_timestamp = start_timestamp + datetime.timedelta(seconds=timestamp_sec)
                formatted_timestamp = actual_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
                print("At Timestamp::", formatted_timestamp)

                if(len(cars) != 0):
                    #cars detected, pass to next model with key words
                    keywords = color + "cars in the photo"

                    DEVICE = 'cpu'
                    everything_results = FastSAM_model(frame,
                                        device=DEVICE,
                                        retina_masks=True,
                                        imgsz=(640,480),
                                        conf=0.4,
                                        iou=0.9)
                    prompt_process = FastSAMPrompt(frame, everything_results, device=DEVICE)
                    ann = prompt_process.text_prompt(text=keywords)
                    #prints the blots around "cars" stored in ann, kept to generate reference photos for presentaiton 
                    prompt_process.plot(annotations=ann,output_path='analysis.jpg',)
                    if(ann.any()):
                        saved_frames.append(frame)
                        print("\nsaving image")
                        
            else:
                print("Video trimming has completed sucessfully")
                cap.release()

        #write frames to file   
        if(len(saved_frames) > 1):
            ################################################################################################################################################            
            frames_to_images(saved_frames, file_name, fps)



            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            for frame in saved_frames:
                # Display the frame
                cv2.imshow("Video", frame)

                # Simulate delay for video playback (adjust delay as needed)
                key = cv2.waitKey(1)  # Wait for 1 millisecond (adjust for desired frame rate) 

                # Exit on 'q' key press
                if key == ord('q'):
                    break

            # Close the window
            cv2.destroyAllWindows()



            frames_to_video(saved_frames)
            
            cap.release()

    else:
        print("no media file")


#takes images from output file and converts them to a video format to provide both frames and video
def frames_to_video(frames):
    


    # Set video frame size (adjust based on your images)
    frame_width = frames[0].shape[1]  # Get width from the first frame
    frame_height = frames[0].shape[0]  # Get height from the first frame


    size = (frame_width, frame_height) 
   
    # Below VideoWriter object will create 
    # a frame of above defined The output  
    # is stored in 'filename.avi' file. 
    video_writer = cv2.VideoWriter('result_video.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 10, size) 

    # Write each frame to the video
    for i, frame in enumerate(frames):
        # Convert frame to BGR format if needed (depending on your array format)
        if frame.ndim == 3 and frame.shape[-1] == 4:  # Check for RGBA format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        video_writer.write(frame)   

    # Release video writer
    video_writer.release()



if __name__ == "__main__":
    if len(sys.argv) == 4:
        path = "media/" + sys.argv[1]
        print(path)
        trim_footage(path,sys.argv[2], sys.argv[3])
        # frames_to_video(path)
    else:
        print("using preset")
        trim_footage("media/deliveryDriver.mp4", "red", "timestamps"  )

    