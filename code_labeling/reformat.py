import os
import argparse

"""
Reformat from 
<cam id> <car id> <frame id> <bb>

to individual files by <cam id> with
<frame id> <car id> < bb>
"""
def main(argv):
    split_bb_to_cam(argv)

def split_bb_to_cam(argv):
    bb_file = argv.file

    dirname = "./gt_temp/"
    mkdir_ifndef(dirname)

    # Parse data into integer lists
    data = []
    with open(bb_file, 'r') as f:
        for line in f:
            data.append([int(x) for x in line.split(' ')])

    # Outputs
    u_cam = list(set([x[0] for x in data]))

    data = [[x[0], x[2], x[1], *x[3:]] for x in data]
    data.sort(key=lambda x: x[1])

    for cid in u_cam:
        fid = open(dirname + "c{:03d}.txt".format(cid), 'w')
        for line in [[*l[1:], -1] for l in data if l[0] == cid]:
            fid.write(",".join(map(str, line)) + "\n")




def mkdir_ifndef(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action="store", default=None, help="File to reformat")
    argv = parser.parse_args()

    main(argv)