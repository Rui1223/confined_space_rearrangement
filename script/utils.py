#!/usr/bin/env python
from __future__ import division
import pybullet_utils.bullet_client as bc
import pybullet as p
import pybullet_data

from collections import OrderedDict
import os
import random
import math
import numpy as np
import time
import IPython

### utils file contains some of the common functionalities
### used often by different modules

def norm2(v):
    norm = 0.0
    for i in range(len(v)):
        norm += v[i]*v[i]
    norm = math.sqrt(norm)
    return norm


def calculateInnerProduct(v1, v2):
    ip = 0.0
    for i in range(len(v1)):
        ip += v1[i]*v2[i]
    return ip


def calculateNorm2(v1, v2):
    temp_dist = 0.0
    for i in range(len(v1)):
        temp_dist += (v1[i] - v2[i])**2
    temp_dist = math.sqrt(temp_dist)
    return temp_dist


def computePoseDist_pos(pose1, pose2):
    ### Input: format for pose [x,y,z]
    pose_dist = calculateNorm2(pose1, pose2)
    return pose_dist


def computePoseDist_quat(quat1, quat2):
    lamb = calculateInnerProduct(quat1, quat2)
    return (1 - abs(lamb))


def interpolatePosition(P1, P2, f):
    ### Here pos P: [x,y,z]
    x = P1[0] + (P2[0]-P1[0]) * f
    y = P1[1] + (P2[1]-P1[1]) * f
    z = P1[2] + (P2[2]-P1[2]) * f
    P = [x, y, z]
    return P


def interpolateQuaternion(Q1, Q2, f):
    epsilon = 0.1
    ### Here quat Q: [x,y,z,w]
    x1 = Q1[0]
    y1 = Q1[1]
    z1 = Q1[2]
    w1 = Q1[3]
    x2 = Q2[0]
    y2 = Q2[1]
    z2 = Q2[2]
    w2 = Q2[3]
    ### compute the quaternion inner product
    lamb = calculateInnerProduct(Q1, Q2) 
    if lamb < 0:
        w2 = -w2
        x2 = -x2
        y2 = -y2
        z2 = -z2
        lamb = -lamb
    ### calculate interpolation factors (r, s)
    if abs(1-lamb) < epsilon:
        ### the quaternions are nearly parallel, so use linear interpolation
        r = 1 - f
        s = f
    else:
        ### calcuate spherical linear interpolation factors
        alpha = math.acos(lamb)
        gamma = 1.0 / math.sin(alpha)
        r = math.sin((1-f)*alpha) * gamma
        s = math.sin(f*alpha) * gamma

    ### set the interpolated quaternion
    w = r * w1 + s * w2
    x = r * x1 + s * x2
    y = r * y1 + s * y2
    z = r * z1 + s * z2
    Q = [x, y, z, w]
    ### normalize the result
    Q_norm = norm2(Q)
    Q = [x / Q_norm, y / Q_norm, z / Q_norm, w / Q_norm]

    return Q

def getQuaternionFromRotationMatrix(rot):
    """ This function calculates quaternion (1*4)
        given a rotation matrix (3*3)
        
        inputs
        ======
            rot(3*3 numpy array): rotation matrix
        output
        ======
            quat(1*4 list): quaternion (x,y,z,w)
        source:
        https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    """
    if rot[2][2] < 0:
        if (rot[0][0] > rot[1][1]):
            t = 1 + rot[0][0] - rot[1][1] - rot[2][2]
            quat = [t, rot[0][1]+rot[1][0], rot[2][0]+rot[0][2], rot[1][2]-rot[2][1]]
        else:
            t = 1 - rot[0][0] + rot[1][1] - rot[2][2]
            quat = [rot[0][1]+rot[1][0], t, rot[1][2]+rot[2][1], rot[2][0]-rot[0][2]]
    else:
        if (rot[0][0] < -rot[1][1]):
            t = 1 - rot[0][0] - rot[1][1] + rot[2][2]
            quat = [rot[2][0]+rot[0][2], rot[1][2]+rot[2][1], t, rot[0][1]-rot[1][0]]
        else:
            t = 1 + rot[0][0] + rot[1][1] + rot[2][2]
            quat = [rot[1][2]-rot[2][1], rot[2][0]-rot[0][2], rot[0][1]-rot[1][0], t]
    quat = [e*0.5/math.sqrt(t) for e in quat]

    return quat


def generateCombination(L, row, cur, res):
    if (row >= len(L)):
        # res.append(cur)
        if cur not in res:
            res.append(cur)
        return
    for col in range(len(L[row])):
        if L[row][col] not in cur:
            cur_new = cur+[L[row][col]]
            generateCombination(L, row+1, cur_new, res)
        else:
            generateCombination(L, row+1, cur, res)

