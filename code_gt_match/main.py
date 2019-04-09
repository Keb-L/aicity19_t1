"""
Kelvin Lin
AI City 2019

Ground Truth Trajectory Matching
main.py
    top-level script to generate the transitions from the ground truth file.
"""
from functions import *

# PARAMETERS
# Set to true if cross-camera vehicles need to be verified (Setting this to true will filter out all vehicle ids which do not appear in at least 2 camera
verify = True

"""
Single Camera Tracking results - GROUND TRUTH FILE
    Input file: SCT ground truth
    Format
        camID carID FrameID X Y W H
    -----------------------------------------------------------------------------------------------------------
    sct_gt
    list of lists
    
    Parsed ground truth file. Each line is one line in the ground truth.
"""
sct_gt = list()
with open('./ref/0402gt_result_v2.txt', 'r') as f:
# with open('./sct_gt_verified.txt', 'r') as f:
    for line in f:
        sct_gt.append([int(x) for x in line.split(' ')])

"""
Camera to Waypoint
    Input file: camera to waypoint description 
    Format:
        <camera_id> <waypoint1_id> ... <waypointN_id>
    -----------------------------------------------------------------------------------------------------------
    lut_camera_waypoint
    dictionary structure
    
    Lookup table for the waypoints associated with each camera.  
    
    Usage
        lut_camera_waypoint[camera_id] = [waypointA, waypointB, ... , waypointN]      
"""
lut_camera_waypoint = dict()
with open('./ref/cam_to_waypoint_v4.txt', 'r') as f:
    for line in f:
        line_num = [int(x) for x in line.split(' ')]
        lut_camera_waypoint[line_num[0]] = line_num[1:]


"""
Waypoint to Vector
    Input file: waypoint to vector matching file
    Format
        <waypoint id> <vector1_direction> <vector1_ID> ... <vectorN_direction> <vectorN_ID>
    -----------------------------------------------------------------------------------------------------------
    lut_waypoint_vector
    dictionary structure
    
    Lookup table for the vectors associated with each waypoint. Query using the waypoint id. Returns all vectors that
    belong to the waypoint. 
    
    Usage
        lut_waypoint_vector[waypoint_id] = [ [vectorA_direction, vectorA_ID], ... , [vectorN_direction, vectorN_ID] ]
"""
lut_waypoint_vector = dict()
with open('./ref/vector_matching.csv', 'r') as f:
    for line in f:
        line_num = [int(x) for x in line.split(',')]
        lut_waypoint_vector[line_num[0]] = list()
        for d, v in zip(*[iter(line_num[1:])]*2):
            lut_waypoint_vector[line_num[0]].append([d, v])
        # lut_waypoint_vector[line_num[0]] = dict()
        # for d, v in zip(*[iter(line_num[1:])]*2):
        #     if d not in lut_waypoint_vector[line_num[0]]:
        #         lut_waypoint_vector[line_num[0]][d] = list()
        #     lut_waypoint_vector[line_num[0]][d].append(v)

"""
Vector Information
    Input file: vector description
    Format
        <vector_uid> <camera_id> <vector direction> <start(X, Y)> <end (X, Y)> <GPS (Long, Lat, 1) > 
    -----------------------------------------------------------------------------------------------------------
    lut_vector
    double-dictionary structure
    
    Lookup table for vector information. Query using the vector direction and id. Returns the vector's associated camera,
    start point (X1, Y1), end point (X2, Y2) and GPS coordinates.
    
    Usage
        lut_vector[direction][vector_id] = [camera id, X1, Y1, X2, Y2, GPS_Longitude, GPS_Latitude, 1.0]
    
    -----------------------------------------------------------------------------------------------------------
    lut_camera_vector
    dictionary structure
    
    Lookup table for vectors associated with each camera. Query using the camera id. Returns all associated vectors
    as a list of vectors (2-element lists)
    
    Usage
        lut_camera_vector[camera id] -> [direction, vector id]
"""
lut_vector = dict()
lut_camera_vector = dict()
with open('./ref/vector_clean.txt', 'r') as f:
    for line in f:
        line_num = [float(x) for x in line.split(' ')]
        d = int(line_num[2])    # Direction
        c = int(line_num[1])    # Camera ID
        v = int(line_num[0])    # Vector ID
        if d not in lut_vector:
            lut_vector[d] = dict()
        lut_vector[d][v] = [line_num[1], *line_num[3:]]

        if c not in lut_camera_vector:
            lut_camera_vector[c] = list()
        lut_camera_vector[c].append([d, v])

"""
Link Matrix - Not used
    Input file: CSV link matrix
    Format
        Row and Column Headers in first row and column
        Zero if no link, otherwise non-zero value

    -----------------------------------------------------------------------------------------------------------
    lut_link_matrix
    double-dictionary structure
    
    Usage
        lut_link_matrix[A][B] -> (boolean) is linked
        where, A, B are waypoints
"""
lm_temp = list()
with open('./ref/link_matrix_s5.csv', 'r') as f:
    for line in f:
        lm_temp.append([float(x) for x in line.split(',')])
lut_link_matrix = convert_matrix_to_lut(lm_temp)


"""
Time Stamp
    Input file: time stamp description
    Format
        <camera id> <time offset (seconds)>
    Notes:
        <camera id> is formatted as c*** where *** is the zero-extended camera id.
        i.e. camera 5 becomes c005, camera 10 becomes c010
        > This is the same format as the AI City cam_timestamp files!
    -----------------------------------------------------------------------------------------------------------
    lut_timestamp
    dictionary structure
    
    Lookup table for the camera time offsets.
    
    Usage
        lut_timestamp[camera id] -> time offset (seconds)
"""
lut_timestamp = dict()
with open('./cam_timestamp/S05.txt', 'r') as f:
    for line in f:
        line = line.split(' ')
        lut_timestamp[int(line[0][1:])] = float(line[1])

with open('./cam_timestamp/S02.txt', 'r') as f:
    for line in f:
        line = line.split(' ')
        lut_timestamp[int(line[0][1:])] = float(line[1])


# Cleanup of temporary variables
del lm_temp, line, line_num, f

#############################################
#              MAIN START
#############################################

# Run Scenario 5 first only (Camera 10+)
# sct_gt = list(filter(lambda x: x[0] > 9, sct_gt))

# Multi-camera crossing verification
if verify:
    filter_single_camera_ids(sct_gt)
    write_file('./sct_gt_verified.txt', sct_gt)

# Camera transition identification
pairs = create_transition_pairs(sct_gt, lut_timestamp)

# Using the transition information, identify the vectors and waypoints
result = identify_vectors_waypoints(pairs, lut_camera_vector, lut_waypoint_vector, lut_vector)

# Output to file
write_file("./result.txt", result)
logfile.close()


