"""
Kelvin Lin
AI City 2019

Ground Truth Trajectory Matching
functions.py
    Helper function collection
"""
import numpy as np
import math
from scipy.spatial import distance
from itertools import combinations
from tqdm import tqdm

logfile = open('./output.log', 'a')

# PARAMETERS
framerate = 10          # Frames per second
trajectory_frames = 5  # Use up to 10 frames for trajectory

zone_width = 100     # 40 pixel W/H
IOU_weight = 0    # IOU weighting
COS_weight = 1    # cosine weighting


def convert_matrix_to_lut(M):
    """
    Converts a list of lists (Matrix) into a dictionary of dictionaries (LUT)
    Assumes the first row/column of the matrix are the labels

    Enforces a square matrix with the same labels on the row and column header.
    :return: dictionary
    """

    # Retrieve labels from first row/column. Ignore the first element at (0, 0)
    row_label = [int(x[0]) for x in M][1:]
    column_label = [int(x) for x in M[0]][1:]

    # Extract the links from the link matrix, convert to booleans
    M_vals = [[bool(y) for y in x[1:]] for x in M][1:]

    assert(row_label == column_label)  # Assert that they are the same (square matrix)

    lut = dict()
    r_i = c_i = 0   # Row/Column indexers
    for src in row_label:
        lut[src] = dict()
        for dst in column_label:
            lut[src][dst] = M_vals[r_i][c_i]
            c_i += 1
        r_i += 1
        c_i = 0

    return lut

def filter_single_camera_ids(sct_gt):
    """
    Removes all vehicle ids that do not cross cameras.
    i.e. the vehicle is only seen by one camera

    :param sct_gt: Single camera tracking ground truth
    :return: filter ground truth
    """
    # Get all the unique vehicle id numbers
    vehicle_uid = np.unique([x[1] for x in sct_gt])

    print('Removing vehicles that do not cross cameras...')
    pbar = tqdm(total=len(vehicle_uid))     # Progress bar
    del_count = 0  # Deleted records

    for uid in vehicle_uid:  # For each vehicle
        vehicle_gt = list(filter(lambda x: x[1] == uid, sct_gt))   # Retrieve all entries
        camera_uid = np.unique([x[0] for x in vehicle_gt])         # Retrieve all cameras
        if len(camera_uid) < 2:     # Remove if does not cross cameras (not cross-camera)
            del_count += 1
            logfile.write('Vehicle %d removed. Only appears in camera(s) %s. %d entries removed from ground truth.\n'
                          % (uid, ' '.join(str(e) for e in camera_uid), len(vehicle_gt)))
            [sct_gt.remove(entry) for entry in vehicle_gt]
        pbar.update(1)   # Progress bar update
    pbar.close()

    print('%d vehicles were removed from the ground truth.' % del_count)
    return sct_gt


def create_transition_pairs(sct_gt, timestamp):
    """
    Uses the ground truth file to define the transition pairs and extracts the vehicle trajectories
    associated with the transition.

    :param sct_gt: Single camera tracking ground truth [ camID carID FrameID X Y W H ]
    :param timestamp: camera offset timestamp
    :return:
    """
    print("Creating transition pairs...")
    vehicle_uid = np.unique([x[1] for x in sct_gt])
    pbar = tqdm(total=len(vehicle_uid))  # Progress bar

    vehicle_transitions = dict()
    for uid in vehicle_uid:
        # Create vehicle entry into table
        if uid not in vehicle_transitions:
            vehicle_transitions[uid] = list()

        vehicle_gt_orig = list(filter(lambda x: x[1] == uid, sct_gt))  # Retrieve all entries of vehicle id

        # Normalize frames using the timestamp values
        # Offsets are converted into frames, ASSUMING 10 frames per second. This ensures that all time stamps are in
        # integer values.
        vehicle_gt = [x + np.array([0, 0, int(np.around(timestamp[x[0]], 1) * framerate), 0, 0, 0, 0]) for x in vehicle_gt_orig]

        # Compute the frame range in which vehicle #uid is seen by each camera
        camera_frame_range = dict()
        for rec in vehicle_gt:  # for each line in the ground truth
            camera_id = rec[0]
            frame_id = rec[2]
            if camera_id not in camera_frame_range:     # Initialization to [inf, -inf]
                camera_frame_range[camera_id] = [math.inf, -math.inf]
            # Compare bounds
            camera_frame_range[camera_id] = [min(camera_frame_range[camera_id][0], frame_id), max(camera_frame_range[camera_id][1], frame_id)]

        # Compute camera pairs and trajectories
        camera_uid = list(camera_frame_range.keys())
        """
        A transition from A to B exists iff:
            1. the min frame of camera B occurs in camera A's range.
            2. the frame ranges of cameras A and B is disjoint AND B > A AND there does not exist a camera C that is closer to B. 
                i.e. A is the NEAREST NEIGHBOR to B
        """
        camera_nearest_neighbor = dict()
        inf_renorm = lambda x: x if x > 0 else math.inf     # Evaluation function for frame difference. MUST BE POSITIVE
        for cam_uid in camera_uid:
            # Using the starting frame for each camera
            # Find the nearest neighbor with a POSITIVE distance
            # NOTE: Nearest neighbor only applies for disjoint camera ranges!
            min_frame = camera_frame_range[cam_uid][0]
            # Default values
            NN_dist = math.inf
            camera_nearest_neighbor[cam_uid] = -math.inf

            # Iterate through every other camera
            for cam_neighbor in [x for x in camera_frame_range.keys() if x != cam_uid]:
                # Compute the difference between the minimum frame of the range and the first and last frame of every
                # neighboring camera.
                neighbor_dist = inf_renorm(min_frame - camera_frame_range[cam_neighbor][1])     # Disjoint camera check,  compute distance
                overlap_dist = inf_renorm(min_frame - camera_frame_range[cam_neighbor][0])      # Overlapping camera check, compute distance

                if neighbor_dist < NN_dist:  # Modify NN entry if distance < NN_dist and distance >=
                    NN_dist = neighbor_dist
                    camera_nearest_neighbor[cam_uid] = cam_neighbor

                # Edge case for overlapping ranges (no disjoint nearest neighbor)
                # Overlapping ranges CANNOT be nearest neighbors. Saves the overlap distance and sets the neighbor to -inf
                if overlap_dist < NN_dist:
                    NN_dist = overlap_dist
                    camera_nearest_neighbor[cam_uid] = -math.inf
        """
        Transition pair verification
        Generate every combinations of camera pairs and verify
        """
        camera_comb = combinations(list(camera_frame_range.keys()), 2)
        for pair in camera_comb:
            is_transition = False
            src, dest = pair

            # direction determines which way we sample frames to get the trajectory
            #   -1 samples the previous N frames
            #    1 samples the next N frames, where N = trajectory_frames

            # CASE 1a: Overlapping ranges
            """
                  Transition
            src   ||||/|||||||||||||||||      Enter
            dest      /||||||||||||||||||||   Enter
            
            dest start frame occurs in src frame range
            """
            if camera_frame_range[dest][0] in range(*np.array(camera_frame_range[src]) + np.array([1, -1])):
                is_transition = True
                direction = [1, 1]  # Enter, Enter
                frame_enter = camera_frame_range[dest][0]
                frame_exit = camera_frame_range[dest][0]

            # CASE 1b: Overlapping ranges
            """
                                    Transition
            src       |||||||||||||||||/|||   Enter
            dest  |||||||||||||||||||||/      Exit
            
            dest end frame occurs in src frame range
            """
            if camera_frame_range[dest][1] in range(*np.array(camera_frame_range[src]) + np.array([1, -1])):
                is_transition = True
                direction = [1, -1]  # Enter, Exit
                frame_enter = camera_frame_range[dest][1]
                frame_exit = camera_frame_range[dest][1]

            # CASE: Disjoint ranges
            """
                       Transition
            src   |||||||/              Exit
            dest               /|||||   Enter
            
            dest start frame occurs after src end frame
            """
            if camera_nearest_neighbor[dest] == src:
                direction = [-1, 1]  # Exit Enter
                is_transition = True
                frame_enter = camera_frame_range[dest][0]
                frame_exit = camera_frame_range[src][1]

            if not is_transition:
                continue

            # Add entry into vehicle transitions
            # Compute trajectories

            # "Exit frame" = minimum of max frame or entry frame
            # Covers case where vehicle is seen by 2 cameras simultaneously
            trajectory_exit, loc_exit, transition_bb_exit = get_vehicle_trajectory(vehicle_gt, src, frame_exit, direction[0])
            trajectory_enter, loc_enter, transition_bb_enter = get_vehicle_trajectory(vehicle_gt, dest, frame_enter, direction[1])

            # Transition Entry: Camera1 Camera2 Frame1 Frame2 Trajectory1 Trajectory2 BBOX1 BBOX2
            transition_record = [src, dest, frame_exit, frame_enter, *trajectory_exit, *trajectory_enter, *loc_exit, *loc_enter, transition_bb_exit, transition_bb_enter]
            vehicle_transitions[uid].append(transition_record)

        pbar.update(1)   # Progress bar update
    pbar.close()
    return vehicle_transitions


def get_vehicle_trajectory(vehicle_gt, camera_id, framenumber, direction, sample_frames=trajectory_frames):
    """
    Computes the trajectory using sample_frames number of frames.

    :param vehicle_gt: ground truth for a single vehicle id
    :param camera_id: camera id
    :param framenumber: transition frame number
    :param direction: sampling direction
    :param sample_frames: number of frames to sample
    :return: averaged vehicle trajectory, averaged vehicle position, vehicle bounding box (in all sampled frames)
    """

    camera_vehicle_gt = list(filter(lambda x: x[0] == camera_id, vehicle_gt))  # Retrieve all entries of vehicle id
    camera_vehicle_gt.sort(key=lambda x: x[2])  # Sort by frame number

    camera_vehicle_frame = [x[2] for x in camera_vehicle_gt]

    if framenumber not in camera_vehicle_frame:     # Case for when the frame number is not in the list
        framenumber = max(x for x in camera_vehicle_frame if x < framenumber)
    start_index = camera_vehicle_frame.index(framenumber)

    if direction < 0:
        sample_index = range(max(0, start_index - sample_frames), start_index)  # End Frame - Sample Frames to End Frame (cutoff edges)
    elif direction > 0:
        sample_index = range(start_index, min(len(camera_vehicle_frame), start_index + sample_frames))  # End Frame to End Frame + Sample Frames (cutoff edges)
    else:
        print("Illegal argument to get_vehicle_trajectory")
        exit(1)

    # Trajectory for last sample_frame frames
    # X Y W H
    rect_center = lambda r: np.array([r[0] + r[2]//2, r[1] + r[3]//2])
    transition_bb = [camera_vehicle_gt[x][-4:] for x in sample_index]
    transition_path = list(map(rect_center, transition_bb))

    trajectory = np.array([0, 0], dtype=np.float32)
    loc = np.sum(transition_path, axis=0) // sample_frames

    # Compute path difference
    prev = transition_path[0]
    for next in transition_path[1:]:
        trajectory += np.subtract(next, prev)
        prev = next

    # Checks to see if the vehicle moved (0 magnitude check)
    if np.linalg.norm(trajectory) == 0:
        return trajectory, loc, transition_bb

    trajectory /= np.linalg.norm(trajectory)    # Normalize to unit vector
    return trajectory, loc, transition_bb


def identify_vectors_waypoints(transition_pairs, lut_camera_vector, lut_waypoint_vector, lut_vector):
    """
    Uses brute-force evaluation of every possible combination to identify the most likely match. See match_vectors
    Populates the result structure for writing out to the file system.

    :param transition_pairs: list of all transitions
    :param lut_camera_vector: camera to vector lookup table
    :param lut_waypoint_vector: waypoint to vector lookup table
    :param lut_vector: vector descriptions
    :return: See below for result format.
    """

    # Generate camera to vector
    # Incorrect
    # lut_camera_vector = dict()
    # for cam in lut_camera_waypoint.keys():
    #     lut_camera_vector[cam] = list()
    #     for waypoint in lut_camera_waypoint[cam]:
    #         lut_camera_vector[cam] += lut_waypoint_vector[waypoint]
    full_result = list()

    """
    For each trajectory, find all vectors associated with the camera
    Use cosine similarity to find the most likely vector 
    Match the vector with the waypoints, record
    """
    vehicle_uid = list(transition_pairs.keys())
    for uid in vehicle_uid:
        vehicle_trajectory = transition_pairs[uid]
        for t in vehicle_trajectory:
            camA, camB = t[0:2]
            delta_frame = t[3] - t[2]

            pathA, pathB = [t[4:6], t[6:8]]
            locA, locB = [t[8:10], t[10:12]]

            transition_path_A = t[-2]
            transition_path_B = t[-1]

            # Exit and Enter vectors ID, BBOX zone, trajectory zone
            vectIDA, vectA_zone, pathA_zone = match_vector(lut_camera_vector[camA], lut_vector, pathA, locA, transition_path_A, 1)
            vectIDB, vectB_zone, pathB_zone = match_vector(lut_camera_vector[camB], lut_vector, pathB, locB, transition_path_B, 0)

            # If either vector is invalid, throw out the result.
            if vectIDA is None or vectIDB is None:
                continue

            # Retrieve vector start/end points and GPS
            vectA = lut_vector[vectIDA[0]][vectIDA[1]][1:]
            vectB = lut_vector[vectIDB[0]][vectIDB[1]][1:]

            vectA_start, vectB_start = np.array([vectA[0:2], vectB[0:2]], dtype=np.int32)
            vectA_end, vectB_end = np.array([vectA[2:4], vectB[2:4]], dtype=np.int32)
            vectA_GPS, vectB_GPS = [vectA[4:7], vectB[4:7]]

            # Retrieve waypoint values
            wayA = [int(x) for x in lut_waypoint_vector.keys() if vectIDA in lut_waypoint_vector[x]]
            wayB = [int(x) for x in lut_waypoint_vector.keys() if vectIDB in lut_waypoint_vector[x]]

            # Invalid waypoint check
            if not wayA:
                wayA = [-1]
            else:
                logfile.write("Multiple waypoint association detected! Vehicle {:d} c{:d} to c{:d}, Waypoint {:s}\n".format(uid, camA, camB, " ".join(str(x) for x in wayA)))
                wayA = [wayA[0]]
            if not wayB:
                wayB = [-1]
            else:
                logfile.write("Multiple waypoint association detected! Vehicle {:d} c{:d} to c{:d}, Waypoint {:s}\n".format(uid, camA, camB, " ".join(str(x) for x in wayB)))
                wayB = [wayB[0]]

            # Vector to Waypoint is a one-to-one relation
            assert(len(wayA) == 1 and len(wayB) == 1)
            """
            Output format (space delimited)
                index   value
                0       camA_ID 
                1       camB_ID
                2       frame_difference 
                3       waypointA_ID-vector_type 
                4       waypointB_ID-vector_type 
                5-6     waypointA_start_point
                7-8     waypointB_start_point
                9-10    waypointA_end_point 
                11-12   waypointB_end_point 
                13-15   waypointA_GPS 
                16-18   waypointB_GPS 
                19-20   waypointA_vector 
                21-22   waypointB_vector
                23-24   vectorA_bbox
                25-26   vectorB_bbox
                27-28   trajectoryA_bbox
                29-30   trajectoryB_bbox
            """
            full_result.append([uid, camA, camB, delta_frame, *wayA, *wayB, *vectA_start, *vectB_start, *vectA_end, *vectB_end, *vectA_GPS, *vectB_GPS, *vectIDA, *vectIDB, *vectA_zone, *vectB_zone, *pathA_zone, *pathB_zone])
    return full_result


def match_vector(vector_list, lut_vector, trajectory, traj_loc, transition_path, direction):
    """
    Returns the vector id that matches the trajectory most closely.
    Uses IOU and cosine similarity to evaluate vectors. The weighting of the function is controlled by
    the IOU_weight and COS_weight global variables
    :return: the vector that is most likely a match
    """
    vect_ret = None
    vect_zone = None
    vect_similarity = -math.inf

    # Helper functions
    get_midpoint = lambda v: np.array([(v[0]+v[2])//2, (v[1]+v[3])//2])
    get_trajectory = lambda v: np.array([v[2]-v[0], v[3]-v[1]]) / np.linalg.norm(np.array([v[2]-v[0], v[3]-v[1]]))

    # Evaluation functions
    eval_function = lambda a, b: IOU_weight * a + COS_weight * b
    normalize_cos = lambda x: (x+1)/2

    # Finds every vector
    # for v in filter(lambda x: x[0] == direction, vector_list):
    for v in vector_list:
        vect_temp = lut_vector[v[0]][v[1]]
        vect_loc = get_midpoint(vect_temp)
        vect_traj = get_trajectory(vect_temp)

        # Create a BBOX around the midpoint of the vector
        # TODO: Modify this if you want to use a labeled bounding zone
        vect_bb_iou = [*(vect_loc - zone_width//2).astype(np.int), zone_width, zone_width]

        # Compute the averaged IOU over the sampled frames
        iou_accumulator = 0
        for transition_bb in transition_path:
            iou_accumulator += bb_intersection_over_union(transition_bb, vect_bb_iou)
        iou_accumulator /= len(transition_path)

        # Compute cosine similarity
        cosine_sim = normalize_cos(1 - distance.cosine(trajectory, vect_traj))

        logfile.write("iou %f\n" % iou_accumulator)

        # Check against existing best match. Update if evaluation is higher.
        if eval_function(iou_accumulator, cosine_sim) > vect_similarity:
            vect_similarity = eval_function(iou_accumulator, cosine_sim)
            vect_ret = v
            vect_zone = vect_bb_iou

    # Return best match vector, the vector's bounding box, and the trajectory bounding box (sampled at the middle frame)
    return vect_ret, vect_zone, transition_path[len(transition_path)//2]


def bb_intersection_over_union(bbA, bbB):
    """
    IOU implementation I found online
    :param bbA:
    :param bbB:
    :return:
    """
    boxA = [bbA[0], bbA[1], bbA[0] + bbA[2], bbA[1] + bbA[3]]
    boxB = [bbB[0], bbB[1], bbB[0] + bbB[2], bbB[1] + bbB[3]]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def write_file(filename, data):
    """
    Writes data to file filename.

    :param filename: output file name
    :param data: data structure
    :return:
    """

    fid = open(filename, 'w')
    data_str = [' '.join(str(e) for e in l)+'\n' for l in data]
    fid.writelines(data_str)
    fid.close()
