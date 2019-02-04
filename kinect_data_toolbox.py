
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadtxt(path):
    with open(path, "r") as fptr:
        lines = fptr.readlines()[1:25]
        x = np.array([float(l.split(" ")[1]) for l in lines])
        y = np.array([float(l.split(" ")[2]) for l in lines])
        z = np.array([float(l.split(" ")[3]) for l in lines])
        selection = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 17, 19, 20]
        x = x[selection]
        y = y[selection]
        z = z[selection]
    return x, y, z




# Kinect joints:
# 0     Spinebase       #H36M_NAMES[0]  = 'Hip'
# 1     Spinemid        #H36M_NAMES[13] = 'Thorax'
# 2     Neck            #H36M_NAMES[14] = 'Neck/Nose'
# 3     Head            #H36M_NAMES[15] = 'Head'
# 4     Shoulderleft    #H36M_NAMES[17] = 'LShoulder'
# 5     Elbowleft       #H36M_NAMES[18] = 'LElbow'
# 6     WristLeft       #H36M_NAMES[19] = 'LWrist'
# 7     HandLeft
# 8     Shoulderright   #H36M_NAMES[25] = 'RShoulder'
# 9     Elbowright      #H36M_NAMES[26] = 'RElbow'
# 10    Wristright      #H36M_NAMES[27] = 'RWrist'
# 11    HandRight
# 12    HipLeft         #H36M_NAMES[6]  = 'LHip'
# 13    KneeLeft        #H36M_NAMES[7]  = 'LKnee'
# 14    AnkleLeft
# 15    FootLeft        #H36M_NAMES[8]  = 'LFoot'
# 16    Hipright        #H36M_NAMES[1]  = 'RHip'
# 17    KneeRight       #H36M_NAMES[2]  = 'RKnee'
# 18    AnkleRight
# 19    FootRight       #H36M_NAMES[3]  = 'RFoot'
# 20    SpineShoulder   #H36M_NAMES[12] = 'Spine'
# 21    HandtipLeft
# 22    TumbLeft
# 23    HandTipRight
# 24    ThumbRight

# dont need all kinect joints  selection = [[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16, 17, 19, 20]]

def draw_fig(path, savePath= "/home/narvis/Dev/data_kinect/kinect _fig_vis_lessJoints",debug = False):
    base = os.path.basename(path).split("_")[1] + ".png"#
    x, y, z = loadtxt(path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('common xlabel')
    ax.set_ylabel('common ylabel')
    ax.set_xlim([-1, 1])
    ax.set_ylim([1, 3])
    ax.set_zlim([-1, 1])
    ax.scatter(x, z, zs=y, zdir='z', s=20, c=None, depthshade=True)
    fig.savefig(os.path.join(savePath, base))
    print("saved: {}".format(os.path.join(savePath, base)))
    if debug:
        plt.show()
    plt.close()


def visualizeKinect():
    dir = "/home/narvis/Dev/data_kinect/pose_data/"
    directory = os.fsencode(dir)
    allFiles = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            allFiles.append(filename)

    lsorted = sorted(allFiles, key=lambda x: int(x.split('_')[1]))

    for i in lsorted:
        path = os.path.join(dir, i)
        draw_fig(path, debug=False)

    print("Done")


def viz2figs(kinect_pose, vp3d_pose):
    # make comparable --> transform vp3d poses to kinect
    vp3d_pose = vp3d_pose[:, [0, 7, 9, 10, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6, 8]]
    for i in range(0,17):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('xlabel')
        ax.set_ylabel('zlabel')
        ax.set_zlabel('ylabel')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        index = i
        ax.scatter(kinect_pose[0][index], kinect_pose[1][index], zs=kinect_pose[2][index], zdir='y', s=20, c='blue', depthshade=True, marker='s')
        ax.scatter(vp3d_pose[0][index] * -1, vp3d_pose[1][index] , zs=vp3d_pose[2][index]*-1, zdir='y', s=20, c='blue', depthshade=True, marker='s')
        kinect_pose_WoIndex = np.delete(kinect_pose, index, axis =1)
        vp3d_pose_WoIndex = np.delete(vp3d_pose, index, axis=1)

        ax.scatter(kinect_pose_WoIndex[0], kinect_pose_WoIndex[1], zs=kinect_pose_WoIndex[2], zdir='y', s=20, c='red', depthshade=True)
        ax.scatter(vp3d_pose_WoIndex[0] * -1, vp3d_pose_WoIndex[1] , zs=vp3d_pose_WoIndex[2] *-1, zdir='y', s=20, c='green', depthshade=True)

        plt.show()
        plt.close()

def compare(lsorted):
    dir_vp3d = "/home/narvis/Dev/data_kinect/compare_vp3_poses/out_3D_vp3d.npz"
    dir_kinect = "/home/narvis/Dev/data_kinect/pose_data/"
    poses_vp3d = np.load(dir_vp3d)

    poses_kinect = []
    for i in lsorted:
        path = os.path.join(dir_kinect, i)
        x, y, z = loadtxt(path)
        poses_kinect.append([x,y,z])

    poses_vp3d = poses_vp3d["arr_0"]
    #kinect started with frame 22 we need to cut poses_vp3d to be able to compare
    poses_vp3d = poses_vp3d[22:]
    poses_vp3d = np.transpose(poses_vp3d, [0, 2, 1])
    poses_vp3d = poses_vp3d[:,[0,2,1], :]
    poses_kinect = np.array(poses_kinect)

    assert(poses_vp3d.shape == poses_kinect.shape)
    print("Shape of poses_vp3d: {}; Shape of poses_kinect: {}".format(poses_vp3d.shape, poses_kinect.shape))
    #print("Head vp3D: {}; Head Kinect: {}".format(poses_vp3d, poses_kinect[0][3]))


    first_pose_vp3d = poses_vp3d[0]
    first_pose_kinect = poses_kinect[0]


    vp3d_mean = np.mean(first_pose_vp3d, axis=1)
    kinect_mean = np.mean(first_pose_kinect, axis=1)

    first_pose_vp3d = np.transpose(np.transpose(first_pose_vp3d) - vp3d_mean)
    first_pose_kinect = np.transpose(np.transpose(first_pose_kinect) - kinect_mean)



    viz2figs(first_pose_kinect, first_pose_vp3d)
    print("tese")


def test():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('xlabel')
    ax.set_ylabel('zlabel')
    ax.set_zlabel('ylabel')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.scatter([0],[0],[1], zdir='y', s=20, c='green', depthshade=True)
    plt.show()
    plt.close()

def  main():
    dir = "/home/narvis/Dev/data_kinect/pose_data/"
    directory = os.fsencode(dir)
    allFiles = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            allFiles.append(filename)

    lsorted = sorted(allFiles, key=lambda x: int(x.split('_')[1]))

    compare(lsorted)
    #test()



if __name__ == "__main__":
    main()