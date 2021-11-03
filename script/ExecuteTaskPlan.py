#!/usr/bin/env python
from __future__ import division

import time
import sys
import os
import numpy as np

import utils2

import rospy
import rospkg

############################### description #########################################
### This class defines a TaskPlanExecutor class which
### (1) asks the execution scene to load an instance
### (2) load the pre-saved path
### (2) aks the execution scene to execute the planned path
#####################################################################################

class TaskPlanExecutor(object):
    def __init__(self, args):
        ### set the rospkg path
        rospack = rospkg.RosPack()
        self.rosPackagePath = rospack.get_path("confined_space_rearrangement")
        self.num_objects = int(args[1])
        self.instance_id = int(args[2])
        self.instanceFolder = os.path.join(
            self.rosPackagePath, "examples", str(self.num_objects), str(self.instance_id))

    def rosInit(self):
        ### This function specifies the role of a node instance for this class ###
        ### and initializes a ros node
        rospy.init_node("execute_task_plan", anonymous=True)


def main(args):
    task_plan_executor = TaskPlanExecutor(args)
    task_plan_executor.rosInit()
    rate = rospy.Rate(10) ### 10hz

    ### load an instance in the execution scene
    initialize_instance_success = utils2.serviceCall_generateInstanceCylinder(
            task_plan_executor.num_objects, task_plan_executor.instance_id, False)
    if initialize_instance_success:
        object_paths, resetHome_trajectory = utils2.loadWholePlan(task_plan_executor.instanceFolder)
        input("press enter to see the execution of the task plan")
        utils2.executeWholePlan(object_paths, resetHome_trajectory)
        print("finish the execution!")

    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    main(sys.argv)