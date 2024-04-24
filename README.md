trimmer.py is the project file 

    commands to run code
    python3 trimmer.py footage.mp4 color endtimestamp 

    example command to run code 

        python ./trimmer.py 2024-04-17-06-30-02.mp4 red "2024-04-17-06-30-04"

        -filename acts as the start timestamp and the last arugment is the ending timestamp.
        -timestamps need to follow the same format as above with hyphens inbetween integers in         year-month-day-hour-minute-second format. 
        -this example input will return all frames between the start and stop timestamp with cars that are red


car_detection.py and cars_counter.py are references from another project 



Built with python3 and opencv 


        pip install opencv-python 