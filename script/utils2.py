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
import pickle

import rospy
import rospkg

from confined_space_rearrangement.msg import CylinderObj
from confined_space_rearrangement.srv import GenerateInstanceCylinder, GenerateInstanceCylinderRequest
from confined_space_rearrangement.srv import CylinderPositionEstimate, CylinderPositionEstimateRequest
from confined_space_rearrangement.srv import ReproduceInstanceCylinder, ReproduceInstanceCylinderRequest
from confined_space_rearrangement.srv import GenerateConfigsForStartPositions, GenerateConfigsForStartPositionsRequest
from confined_space_rearrangement.srv import ResetPlanningInstance, ResetPlanningInstanceRequest
from confined_space_rearrangement.srv import ClearPlanningInstance, ClearPlanningInstanceRequest
from confined_space_rearrangement.srv import ClearExecutionInstance, ClearExecutionInstanceRequest
from confined_space_rearrangement.srv import ResetRoadmap, ResetRoadmapRequest
from confined_space_rearrangement.srv import ResetRobotHome, ResetRobotHomeRequest
from confined_space_rearrangement.srv import ExecuteTrajectory, ExecuteTrajectoryRequest
from confined_space_rearrangement.srv import AttachObject, AttachObjectRequest

### utils2 file contains some of the common service calls
### used by different experiment/test/example setting


def serviceCall_generateInstanceCylinder(num_objects, instance_number, isNewInstance):
    rospy.wait_for_service("generate_instance_cylinder")
    request = GenerateInstanceCylinderRequest()
    request.num_objects = num_objects
    request.instance_number = instance_number
    request.isNewInstance = isNewInstance
    try:
        generateInstanceCylinder_proxy = rospy.ServiceProxy(
                    "generate_instance_cylinder", GenerateInstanceCylinder)
        generate_instance_cylinder_response = generateInstanceCylinder_proxy(
                request.num_objects, request.instance_number, request.isNewInstance)
        return generate_instance_cylinder_response.success
    except rospy.ServiceException as e:
        print("generate_instance_cylinder service call failed: %s" % e)

def serviceCall_cylinderPositionEstimate():
    rospy.wait_for_service("cylinder_position_estimate")
    request = CylinderPositionEstimateRequest()
    try:
        cylinderPositionEstimate_proxy = rospy.ServiceProxy(
            "cylinder_position_estimate", CylinderPositionEstimate)
        cylinder_position_estimate_response = cylinderPositionEstimate_proxy(request)
        return cylinder_position_estimate_response.cylinder_objects
    except rospy.ServiceException as e:
        print("cylinder_position_estimate service call failed: %s" % e)

def serviceCall_reproduceInstanceCylinder(cylinder_objects):
    ### Input: cylinder_objects (CylinderObj[])
    rospy.wait_for_service("reproduce_instance_cylinder")
    request = ReproduceInstanceCylinderRequest(cylinder_objects)
    try:
        reproduceInstanceCylinder_proxy = rospy.ServiceProxy(
            "reproduce_instance_cylinder", ReproduceInstanceCylinder)
        reproduce_instance_cylinder_response = reproduceInstanceCylinder_proxy(request.cylinder_objects)
        return list(reproduce_instance_cylinder_response.initial_arrangement), \
                list(reproduce_instance_cylinder_response.final_arrangement), \
                reproduce_instance_cylinder_response.success
    except rospy.ServiceException as e:
        print("reproduce_instance_cylinder service call failed: %s" % e)

def serviceCall_generateConfigsForStartPositions(armType):
    rospy.wait_for_service("generate_configs_for_start_positions")
    request = GenerateConfigsForStartPositionsRequest()
    request.armType = armType
    try:
        generateConfigsForStartPositions_proxy = rospy.ServiceProxy(
            "generate_configs_for_start_positions", GenerateConfigsForStartPositions)
        generate_configs_for_start_positions_response = generateConfigsForStartPositions_proxy(request.armType)
        return generate_configs_for_start_positions_response.success
    except rospy.ServiceException as e:
        print("generate_configs_for_start_positions service call failed" % e)

def serviceCall_reset_planning_instance():
    rospy.wait_for_service("reset_planning_instance")
    request = ResetPlanningInstanceRequest()
    try:
        resetPlanningInstance_proxy = rospy.ServiceProxy("reset_planning_instance", ResetPlanningInstance)
        reset_planning_instance_response = resetPlanningInstance_proxy(request)
        return reset_planning_instance_response.success
    except rospy.ServiceException as e:
        print("reset_planning_instance service call failed: %s" % e)

def serviceCall_clear_planning_instance():
    rospy.wait_for_service("clear_planning_instance")
    request = ClearPlanningInstanceRequest()
    try:
        clearPlanningInstance_proxy = rospy.ServiceProxy("clear_planning_instance", ClearPlanningInstance)
        clear_planning_instance_response = clearPlanningInstance_proxy(request)
        return clear_planning_instance_response.success
    except rospy.ServiceException as e:
        print("clear_planning_instance service call failed: %s" % e)

def serviceCall_clear_execution_instance():
    rospy.wait_for_service("clear_execution_instance")
    request = ClearExecutionInstanceRequest()
    try:
        clearExecutionInstance_proxy = rospy.ServiceProxy("clear_execution_instance", ClearExecutionInstance)
        clear_execution_instance_response = clearExecutionInstance_proxy(request)
        return clear_execution_instance_response.success
    except rospy.ServiceException as e:
        print("clear_execution_instance service call failed: %s" % e)

def serviceCall_reset_roadmap(armType):
    rospy.wait_for_service("reset_roadmap")
    request = ResetRoadmapRequest()
    request.armType = armType

    try:
        resetRoadmap_proxy = rospy.ServiceProxy("reset_roadmap", ResetRoadmap)
        reset_roadmap_response = resetRoadmap_proxy(request)
        return reset_roadmap_response.success
    except rospy.ServiceException as e:
        print("reset_roadmap service call failed: %s" % e)

def serviceCall_reset_robot_home(armType, isLabeledRoadmapUsed=True):
    rospy.wait_for_service("reset_robot_home")
    request = ResetRobotHomeRequest()
    request.armType = armType
    request.isLabeledRoadmapUsed = isLabeledRoadmapUsed
    try:
        resetRobotHome_proxy = rospy.ServiceProxy("reset_robot_home", ResetRobotHome)
        reset_robot_home_response = resetRobotHome_proxy(request)
        return reset_robot_home_response.success, reset_robot_home_response.resetHome_trajectory
    except rospy.ServiceException as e:
        print("reset_robot_home service call failed: %s" % e)

def serviceCall_execute_trajectory(traj):
    '''call the ExecuteTrajectory service to execute the given trajectory
    inputs
    ======
        traj (an ArmTrajectory object): the trajectory to execute
    outputs
    =======
        success: indicator of whether the trajectory is executed successfully
    '''
    rospy.wait_for_service("execute_trajectory")
    request = ExecuteTrajectoryRequest()
    request.arm_trajectory = traj
    try:
        executeTraj_proxy = rospy.ServiceProxy("execute_trajectory", ExecuteTrajectory)
        executeTraj_response = executeTraj_proxy(request.arm_trajectory)
        return executeTraj_response.success
    except rospy.ServiceException as e:
        print("execute_trajectory service call failed: %s" % e)

def serviceCall_attach_object(attach, object_idx, armType):
    """call the AttachObject service to attach/detach the corresponding object
    inputs
    ======
        attach (bool): indicate the action of attach or detach
        object_idx (int): the object to attach or detach
        armType (string): "Left"/"Right"/"Left_torso"/"Right_torso"
    outputs
    =======
        success: indicator of whether the attach/detach command is fulfilled
    """
    rospy.wait_for_service("attach_object")
    request = AttachObjectRequest()
    request.attach = attach
    request.object_idx = object_idx
    request.armType = armType
    try:
        attachObject_proxy = rospy.ServiceProxy("attach_object", AttachObject)
        attachObject_response = attachObject_proxy(request.attach, request.object_idx, request.armType)
        return attachObject_response.success
    except rospy.ServiceException as e:
        print(" attach_object service call failed: %s" % e)

def executeWholePlan(whole_path, resetHome_trajectory=None):
    """ call the ExecuteTrajectory service to execute each trajectory in the path
        also tell the robot to attach or detach the object among the motions 
        inputs
        ======
            whole path (a list of ObjectRearrangementPath): a sequence of object paths
            resetHome_trajectory (ArmTrajectory): the trajectory to set the robot home
        outputs
        =======
            execute_success (bool): indicate whether success or not
    """
    for path in whole_path:
        print("rearrange object:" + str(path.object_idx))
        ### first execute the transit trajectory in the path
        execute_success = serviceCall_execute_trajectory(path.transit_trajectory)
        ### now attach the object
        attach_success = serviceCall_attach_object(
            attach=True, object_idx=path.object_idx, armType=path.transit_trajectory.armType)
        ### then execute the transfer trajectory in the path
        execute_success = serviceCall_execute_trajectory(path.transfer_trajectory)
        ### now detach the object
        attach_success = serviceCall_attach_object(
            attach=False, object_idx=path.object_idx, armType=path.transit_trajectory.armType)
        ### finally execute the finish trajectory in the path
        execute_success = serviceCall_execute_trajectory(path.finish_trajectory)

    if resetHome_trajectory != None:
        execute_success = serviceCall_execute_trajectory(resetHome_trajectory)

    return execute_success

def resetInstance(armType):
    reset_planning_success = serviceCall_reset_planning_instance()
    reset_roadmap_success = serviceCall_reset_roadmap(armType)
    if reset_planning_success and reset_roadmap_success:
        return True
    else:
        return False

def clearInstance(armType):
    clear_planning_success = serviceCall_clear_planning_instance()
    clear_execution_success = serviceCall_clear_execution_instance()
    reset_roadmap_success = serviceCall_reset_roadmap(armType)
    if clear_planning_success and clear_execution_success and reset_roadmap_success:
        return True
    else:
        return False

def saveInstance(cylinder_objects, instanceFolder):
    ### create the instance folder specified
    if not os.path.exists(instanceFolder):
        os.makedirs(instanceFolder)
    instanceFile = instanceFolder + "/" + "instance_info.txt"
    f_instance = open(instanceFile, "w")
    for cylinder_object in cylinder_objects:
        f_instance.write(str(cylinder_object.obj_idx) + "\n")
        f_instance.write(str(cylinder_object.curr_position.x) + " " + \
            str(cylinder_object.curr_position.y) + " " + str(cylinder_object.curr_position.z) + "\n")
    f_instance.close()

def saveWholePlan(object_paths, instanceFolder, resetHome_trajectory=None):
    f_path = open(instanceFolder + "/path.obj", 'wb')
    pickle.dump(object_paths, f_path)
    pickle.dump(resetHome_trajectory, f_path)

def loadWholePlan(instanceFolder):
    '''load a plan from the specified folder'''
    f_path = open(instanceFolder + "/path.obj", 'rb')
    object_paths = pickle.load(f_path)
    resetHome_trajectory = pickle.load(f_path)
    return object_paths, resetHome_trajectory

def saveSolution(all_methods_time, all_methods_success, all_method_nActions, instanceFolder):
    timeFile = instanceFolder + "/time.txt"
    f_time = open(timeFile, "w")
    for method_time in all_methods_time:
        f_time.write(str(method_time) + "\n")
    f_time.close()
    successFile = instanceFolder + "/success.txt"
    f_success = open(successFile, "w")
    for method_success in all_methods_success:
        f_success.write(str(method_success) + "\n")
    f_success.close()
    nActionsFile = instanceFolder + "/actions.txt"
    f_actions = open(nActionsFile, "w")
    for method_nActions in all_method_nActions:
        f_actions.write(str(method_nActions) + "\n")
    f_actions.close()

def saveOrderingInfo(object_ordering, instanceFolder):
    orderingFile = instanceFolder + "/ordering.txt"
    f_ordering = open(orderingFile, "w")
    for obj_idx in object_ordering:
        f_ordering.write(str(obj_idx) + " ")
    f_ordering.close()