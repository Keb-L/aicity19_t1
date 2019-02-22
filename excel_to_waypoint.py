import numpy as np
import cv2 as cv2

# Define Track parameters
train_set_1 = list(range(1, 6))  # Scenario 1, Cameras 1 to 5
train_set_3 = list(range(10, 16))  # Scenario 3, Cameras 10 to 15
train_set_4 = list(range(16, 41))  # Scenario 4, Cameras 16 to 40
train_set = [*train_set_1, *train_set_3, *train_set_4]  # Training Set

test_set_2 = list(range(6, 10))  # Scenario 2, Cameras 6 to 9
test_set_5 = list([10, *range(16, 30), *range(33, 37)])  # Scenario 5, Cameras 10, 16 to 30, 33 to 37
test_set = [*test_set_2, *test_set_5]  # Testing Set

all_set = list(range(1, 41))

def main():
    excelfn = "modified_waypoints_friendly_matrix.csv"

    # data = Link Matrix
    with open('data/'+excelfn, 'r') as f:
        data = f.read().splitlines()
    # waypoint data
    with open('data/waypoints_1_40_v2.txt', 'r') as f:
        wp = f.read().splitlines()
    # waypoint-camera relation
    with open('data/waypoint_camera_relations_1_40_v2.txt', 'r') as f:
        wp2cam = f.read().splitlines()

    # To matrix
    wplist = [np.fromstring(x, sep=' ') for x in wp]
    wplist = dict((int(k[0]), k[2:4]) for k in wplist)
    # wplist = [int(x[0]) for x in wplist]  #Grab the waypoint id

    # To matrix
    wp2camlist = [np.fromstring(x, sep=' ') for x in wp2cam]

    # Data header
    # data_head = data[0].split(sep=',')
    # data_head = [int(x) for x in data_head[1:]]
    data_mat = [x.split(sep=',') for x in data]  # Matrix

    # Create camera to waypoint hashmap
    # camera -> waypoints
    c2wp = dict()
    for wp in wp2camlist:
        for cam in wp[2:]:
            cam = int(cam)
            if cam not in c2wp:
                c2wp[cam] = [int(wp[0])]
            else:
                c2wp[cam].extend([int(wp[0])])

    # Extract numerical portion of Link Matrix
    data_mat_num = []
    for l in data_mat[1:]:
        temp = list()
        for e in l[1:]:
            if e == '':
                temp.append(0)
            else:
                temp.append(int(e))
        data_mat_num.append(temp)
    data_mat_nz = np.nonzero(data_mat_num)

    # Get waypoint pairs from non-zero data matrix
    data_pairs = []
    for x, y in zip(*data_mat_nz):
        if [y, x] not in data_pairs:
            data_pairs.append([x, y])



    """
    # Get/write waypoint set
       c2wp: dictionary mapping camera to waypoints (in the camera)
       data_pairs: waypoint pairs
       train_set*: set of camera IDs that belong to a training scenario
       test_set*: set of camera IDs that belong to testing scenario
    
        get_wp_set
            arg1: c2wp
            arg2: camera ID list
            
        write_wp_file
            arg1: save file name
            arg2: data_pairs (waypoint pairs)
            arg3: get_wp_set output
            arg4: waypoint string descriptor (train/test)
    """
    dump_wp_main(data_pairs, wplist)
    # write_wp_main(data_pairs, c2wp)
    print()

def dump_wp_main(data_pairs, wplist):
    fid = open("data/wp_gps_pairs.txt", "w")
    fid2 = open("data/wp_pairs.txt", "w")
    for x, y in data_pairs:
        fid.write("{0} {1} {2} {3}\n".format(*wplist[x], *wplist[y]))
        fid2.write("{0} {1}\n".format(x, y))
    fid.close()
    fid2.close()
    print()

def write_wp_main(data_pairs, c2wp):
    wp_train_set = get_wp_set(c2wp, train_set)
    write_wp_file("data/S134_pairs.txt", data_pairs, wp_train_set, "train")
    wp_train_set = get_wp_set(c2wp, train_set_1)
    write_wp_file("data/S1_pairs.txt", data_pairs, wp_train_set, "train")
    wp_train_set = get_wp_set(c2wp, train_set_3)
    write_wp_file("data/S3_pairs.txt", data_pairs, wp_train_set, "train")
    wp_train_set = get_wp_set(c2wp, train_set_4)
    write_wp_file("data/S4_pairs.txt", data_pairs, wp_train_set, "train")

    wp_test_set = get_wp_set(c2wp, test_set)
    write_wp_file("data/S25_pairs.txt", data_pairs, wp_test_set, "test")
    wp_test_set = get_wp_set(c2wp, test_set_2)
    write_wp_file("data/S2_pairs.txt", data_pairs, wp_test_set, "test")
    wp_test_set = get_wp_set(c2wp, test_set_5)
    write_wp_file("data/S5_pairs.txt", data_pairs, wp_test_set, "test")

    wp_all_set = get_wp_set(c2wp, all_set)
    write_wp_file("data/SAll_pairs.txt", data_pairs, wp_all_set, "all")


def get_wp_set(c2wp, cam_set):
    """
    Returns a list of waypoints that belong to the set of cameras

    :param c2wp: camera to waypoint mapping (dictionary)
    :param cam_set: list of cameras to include in waypoint set
    :return:
    """
    wp_set = [c2wp[x] for x in cam_set]
    wp_set = set([x for y in wp_set for x in y])
    return wp_set


def write_wp_file(sv_file, wp_pairs, wp_set, wp_type):
    """
    Writes the relevant waypoint pairs to the file.

    :param sv_file: save file name
    :param wp_pairs: waypoint pairs
    :param wp_set: list of waypoints to consider
    :param wp_type: waypoint type (train/test)
    :return:
    """
    fid = open(sv_file, "w")
    for pair in wp_pairs:
        # If statements enforce the waypoint conditions.

        # Condition: both endpoints in the waypoint set -> pair[0] in wp_set and pair[1] in wp_set
        # if pair[0] in wp_set and pair[1] in wp_set:  # Check to see if waypoint is in the camera set
        #     fid.write("{0} {1} {2}\n".format(*pair, wp_type))

        # Condition: any endpoint in waypoint set -> any(x in wp_set for x in pair)
        if any(x in wp_set for x in pair):
            fid.write("{0} {1} {2}\n".format(*pair, wp_type))

    fid.close()

if __name__ == "__main__":
    main()