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


### This file generates a roadmap for both
### left and right arm of the motoman robot

def main(args):
    print("Let's generate the roadmap for motoman")
    pybullet_plan_scene = PybulletPlanScene(args)

    nsamples = int(args[1])

    # generate samples
    pybullet_plan_scene.planner_p.generateSamples(
        nsamples, pybullet_plan_scene.robot_p, pybullet_plan_scene.workspace_p)
    # pybullet_plan_scene.planner_p.generateSamples(
    #     nsamples, pybullet_plan_scene.robot_p, pybullet_plan_scene.workspace_p, "hybrid_space")
    pybullet_plan_scene.planner_p.samplesConnect(
        pybullet_plan_scene.robot_p, pybullet_plan_scene.workspace_p, "Right_torso")

    time.sleep(10000)

if __name__ == '__main__':
    main(sys.argv)