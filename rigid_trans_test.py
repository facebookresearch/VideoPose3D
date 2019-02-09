import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import sys

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




def getPosesKinect(dir):
    directory = os.fsencode(dir)
    allFiles = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            allFiles.append(filename)

    lsorted = sorted(allFiles, key=lambda x: int(x.split('_')[1]))
    poses_kinect = []
    for i in lsorted:
        path = os.path.join(dir, i)
        x, y, z = loadtxt(path)
        poses_kinect.append([x,y,z])
    return poses_kinect



def load_data(dir_kinect ="/Users/tobiasczempiel/Develope/kinect_data/pose_data/", dir_vp3d = "/Users/tobiasczempiel/Develope/kinect_data/compare_vp3_poses/out_3D_vp3d.npz" ):
    poses_vp3d = np.load(dir_vp3d)
    poses_kinect = getPosesKinect(dir_kinect)
    poses_vp3d = poses_vp3d["arr_0"]
    #kinect started with frame 22 we need to cut poses_vp3d to be able to compare
    poses_vp3d = poses_vp3d[22:]
    poses_vp3d = np.transpose(poses_vp3d, [0, 2, 1])
    poses_vp3d = poses_vp3d[:,[0,2,1], :]
    poses_vp3d = poses_vp3d[:,:, [0,7,9, 10,14,15,16,11,12,13,1,2,3,4,5,6,8]]
    poses_kinect = np.array(poses_kinect)
    assert(poses_vp3d.shape == poses_kinect.shape)
    print("Shape of poses_vp3d: {}; Shape of poses_kinect: {}".format(poses_vp3d.shape, poses_kinect.shape))

    t_pose_kinect = np.transpose(poses_kinect[0])
    t_pose_vp3d = np.transpose(poses_vp3d[0])

    return t_pose_kinect, t_pose_vp3d


def viz2figs(kinect_pose, vp3d_pose):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('xlabel')
    ax.set_ylabel('zlabel')
    ax.set_zlabel('ylabel')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.scatter(kinect_pose[0], kinect_pose[1], zs=kinect_pose[2], zdir='y', s=20, c='red', depthshade=True)
    ax.scatter(vp3d_pose[0] , vp3d_pose[1] , zs=vp3d_pose[2] , zdir='y', s=20, c='green', depthshade=True)
    plt.show()
    plt.close()

def vizfigs(a, b):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('xlabel')
    ax.set_ylabel('zlabel')
    ax.set_zlabel('ylabel')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.scatter(a[0], a[1], zs=a[2], zdir='y', s=20, c='red', depthshade=False)
    ax.scatter(b[0] , b[1] , zs=b[2] , zdir='y', s=20, c='green', depthshade=False)
    plt.show()
    plt.close()

def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    N = A.shape[0] # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.transpose(AA).dot(BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.transpose(Vt) * np.transpose(U)

    t = -R.dot(centroid_A.T)
    t = t + centroid_B.T

    return R, t
def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)#dot(c*R)

    return c, R, t



def random_rot2(a):
    rotation_m = np.array([[ -0.4480736,  0.0000000,  0.8939967],
   [0.7607049,  0.5253220,  0.3812674],
  [-0.4696361,  0.8509035, -0.2353829]])
    b = a.dot(rotation_m * 2)
    #b = a.dot(rotation_m )
    b = b + np.array([0,0,1])
    return b

def main():
    #a = np.array([[0,0,0], [2,0,0], [0,2,0], [2,2,0], [1,1,0]])
    #b = np.array([[0,0,1], [2,0,1], [0,2,1], [2,2,1], [1,1,1]])
    #b = random_rot2(a)

    a,b = load_data()

    vizfigs(np.transpose(a),np.transpose(b))
    #R,t  = rigid_transform_3D(b,a)

    #b2 = b.dot(R)
    #b2 = b2 + t

    c, R, t = umeyama(a, b)
    A1 = a.dot(c * R) + t


    # first arg red , second green
    vizfigs(np.transpose(A1), np.transpose(b))

if __name__ == "__main__":
    main()