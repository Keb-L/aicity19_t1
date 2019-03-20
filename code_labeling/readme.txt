Kelvin Lin
AI City 2019

Ground-truth labeling and Video Combination

main.py
usage: main.py [-h] [-c CAMERA] [-d DELIMITER] [-cf CAMERA_FILE] [-do DO]
               [-gt GROUND_TRUTH] [-cfg CONFIG_FILE] [-scale SCALE_FACTOR]

optional arguments:
  -h, --help            show this help message and exit
  -c CAMERA, --camera CAMERA
                        Camera ID (text file)
  -d DELIMITER, --delimiter DELIMITER
                        file delimiter (default space)
  -cf CAMERA_FILE, --camera_file CAMERA_FILE
                        camera path file
  -do DO                actions to do (label, combine, both)
  -gt GROUND_TRUTH, --ground_truth GROUND_TRUTH
                        Ground truth bounding box path
  -cfg CONFIG_FILE, --config_file CONFIG_FILE
                        compiled video config file
  -scale SCALE_FACTOR, --scale_factor SCALE_FACTOR
                        video resolution scale factor

CAMERA - text file that lists the cameras to perform DO on.
  This is just a list of cameras. If this is not specified, it sets all testing videos.

DO - actions to perform
  label - generated labeled ground truth videos (labeled_vid/labeled_c***.avi files) 
  combine - generate combined video videos (combined_vid/*.avi)
  both - do both label and combine

CONFIG_FILE - determines the placement of each video in the combined video.
  Each line represents one combined video.  SEE cfg/vset.txt
  Format: <location> <camera id>
    <location> is a x-y coordinate pair encoded as a number. (0, 0) corresponds to the
    top-left corner and has location value 0. (1, 0) would be row 1, column 0.
