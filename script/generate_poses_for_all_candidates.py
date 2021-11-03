#!/usr/bin/env python
from __future__ import division

import pybullet as p
import pybullet_data

import time
import sys
import os

from PybulletPlanScene import PybulletPlanScene

import rospy
import rospkg

### This file generates multiple poses for
### each position candidate

def main(args):
    print("Let's generate multiple poses for each position candidate")
    pybullet_plan_scene = PybulletPlanScene(args)

    ### option 1: generates poses for all position candidate
    pybullet_plan_scene.planner_p.generatePosesForAllCandidates(
        pybullet_plan_scene.robot_p, pybullet_plan_scene.workspace_p, "Right_torso")
    ### option 2: generates a high-quality IK dataset
    # pybullet_plan_scene.planner_p.generatePoses_IKdataSet(
    #     pybullet_plan_scene.robot_p, pybullet_plan_scene.workspace_p, "Right_torso")

    time.sleep(10000)



if __name__ == '__main__':
    main(sys.argv)