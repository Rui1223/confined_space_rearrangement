#!/usr/bin/env python
from __future__ import division

import time
import sys
import os
import copy
import numpy as np
from collections import OrderedDict

import rospy
import rospkg

from sensor_msgs.msg import JointState

from confined_space_rearrangement.srv import RearrangeCylinderObject, RearrangeCylinderObjectRequest
from confined_space_rearrangement.srv import GetCurrRobotConfig, GetCurrRobotConfigRequest
from confined_space_rearrangement.srv import UpdateCertainObjectPose, UpdateCertainObjectPoseRequest
from confined_space_rearrangement.srv import ResetRobotCurrConfig, ResetRobotCurrConfigRequest
from confined_space_rearrangement.srv import UpdateManipulationStatus, UpdateManipulationStatusRequest
from confined_space_rearrangement.srv import SetSceneBasedOnArrangement, SetSceneBasedOnArrangementRequest


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class MonotoneLocalSolver(object):
    def __init__(self, startArrNode, target_arrangement, time_allowed, isLabeledRoadmapUsed=True):

        ### understand the local arrangement task
        self.start_arrangement = startArrNode.arrangement
        self.target_arrangement = target_arrangement
        ### a list of obj_idx of objects to arranged
        self.all_objects = [i for i in range(len(self.start_arrangement)) \
            if self.start_arrangement[i] != self.target_arrangement[i]]
        self.num_objects = len(self.all_objects)
        self.isLabeledRoadmapUsed = isLabeledRoadmapUsed

        ### initialize the tree structure
        self.tree = OrderedDict() ### key: (scalar 0,1,etc..) value: ArrNode
        self.node_idx = 0 ### start from root node (idx: 0)
        self.tree[0] = startArrNode

        ### set the time limit
        self.time_threshold = time_allowed
        self.local_planning_startTime = time.time()

        ### record the (motion planning + collision checking) time
        self.motion_planning_time = 0


    def generateLocalNode(self, parent_node_id, obj_idx, transition_path):
        '''generate a local node which has parent_node_id, given the obj_idx and the transition_path'''
        current_arrangement = self.tree[parent_node_id].arrangement
        resulting_arrangement = copy.deepcopy(current_arrangement)
        resulting_arrangement[obj_idx] = self.target_arrangement[obj_idx]
        current_ordering = self.tree[parent_node_id].object_ordering
        resulting_robot_config = self.serviceCall_getCurrRobotConfig()
        if self.tree[parent_node_id].objectTransferred_idx == None:
            ### parent node is the root node, no transit info can be obtained in this case
            resulting_transit_from_info = None
        else:
            resulting_transit_from_info = [
                self.tree[parent_node_id].objectTransferred_idx, \
                self.tree[parent_node_id].obj_transfer_position_indices[1]]
        resulting_obj_transfer_position_indices = [current_arrangement[obj_idx], self.target_arrangement[obj_idx]]
        resulting_cost_to_come = self.tree[parent_node_id].cost_to_come + 1
        resulting_object_ordering = current_ordering + [obj_idx]
        ### add this newly-generated node
        self.node_idx += 1
        self.tree[self.node_idx] = ArrNode(
            resulting_arrangement, resulting_robot_config, self.node_idx, 
            resulting_transit_from_info, resulting_obj_transfer_position_indices, obj_idx, 
            transition_path, resulting_cost_to_come, parent_node_id, resulting_object_ordering
        )

    def revertBackToParentNode(self, parent_node_id, obj_idx, obj_parent_position_idx, armType):
        '''revert back to the parent node (pop out operation in DFS) by
        put the object and robot back to the configuration they belong to
        at the beginning of the function call'''
        update_success = self.serviceCall_updateCertainObjectPose(obj_idx, obj_parent_position_idx)
        current_robot_config = self.tree[parent_node_id].robotConfig
        update_success = self.serviceCall_resetRobotCurrConfig(current_robot_config)
        update_success = self.serviceCall_updateManipulationStatus(armType)


    def serviceCall_rearrangeCylinderObject(self, obj_idx, target_position_idx, armType, isLabeledRoadmapUsed=True):
        rospy.wait_for_service("rearrange_cylinder_object")
        request = RearrangeCylinderObjectRequest()
        request.object_idx = obj_idx
        request.target_position_idx = target_position_idx
        request.armType = armType
        request.isLabeledRoadmapUsed = isLabeledRoadmapUsed
        try:
            rearrangeCylinderObject_proxy = rospy.ServiceProxy(
                "rearrange_cylinder_object", RearrangeCylinderObject)
            rearrange_cylinder_object_response = rearrangeCylinderObject_proxy(request)
            return rearrange_cylinder_object_response.success, rearrange_cylinder_object_response.path
        except rospy.ServiceException as e:
            print("rearrange_cylinder_object service call failed: %s" % e)

    def serviceCall_getCurrRobotConfig(self):
        '''call the GetCurrRobotConfig service to get the robot current config from planning
           expect output: configuration of all controllable joints (1 + 7 + 7 + 6) '''
        rospy.wait_for_service("get_curr_robot_config")
        request = GetCurrRobotConfigRequest()
        try:
            getCurrRobotConfig_proxy = rospy.ServiceProxy("get_curr_robot_config", GetCurrRobotConfig)
            getCurrRobotConfig_response = getCurrRobotConfig_proxy(request)
            return getCurrRobotConfig_response.robot_config.position
        except rospy.ServiceException as e:
            print("get_curr_robot_config service call failed: %s" % e)

    def serviceCall_updateCertainObjectPose(self, obj_idx, target_position_idx):
        '''call the UpdateCertainObjectPose service to update the object
           to the specified target_position_idx'''
        rospy.wait_for_service("update_certain_object_pose")
        request = UpdateCertainObjectPoseRequest()
        request.object_idx = obj_idx
        request.object_position_idx = target_position_idx
        try:
            updateCertainObjectPose_proxy = rospy.ServiceProxy(
                    "update_certain_object_pose", UpdateCertainObjectPose)
            updateCertainObjectPose_response = updateCertainObjectPose_proxy(
                                request.object_idx, request.object_position_idx)
            return updateCertainObjectPose_response.success
        except rospy.ServiceException as e:
            print("update_certain_object_pose service call failed: %s" % e)

    def serviceCall_resetRobotCurrConfig(self, robot_curr_config):
        '''call the ResetRobotCurrConfig service to reset the robot
           to the specified configuration'''
        rospy.wait_for_service("reset_robot_curr_config")
        request = ResetRobotCurrConfigRequest()
        request.robot_config = JointState()
        request.robot_config.position = robot_curr_config
        try:
            resetRobotCurrConfig_proxy = rospy.ServiceProxy("reset_robot_curr_config", ResetRobotCurrConfig)
            resetRobotCurrConfig_response = resetRobotCurrConfig_proxy(request.robot_config)
            return resetRobotCurrConfig_response.success
        except rospy.ServiceException as e:
            print("reset_robot_curr_config service call failed: %s" % e)
    
    def serviceCall_updateManipulationStatus(self, armType):
        '''call the UpdateManipulationStatus service to disable
           any relationship between the robot and the object'''
        rospy.wait_for_service("update_manipulation_status")
        request = UpdateManipulationStatusRequest()
        request.armType = armType
        try:
            updateManipulationStatus_proxy = rospy.ServiceProxy("update_manipulation_status", UpdateManipulationStatus)
            updateManipulationStatus_response = updateManipulationStatus_proxy(request.armType)
            return updateManipulationStatus_response.success
        except rospy.ServiceException as e:
            print("update_manipulation_status service call failed: %s" % e)

    def serviceCall_setSceneBasedOnArrangementNode(self, arrangement, robotConfig, armType):
        '''call the SetSceneBasedOnArrangement service to
           set scene based on arrangement node'''
        rospy.wait_for_service("set_scene_based_on_arrangement")
        request = SetSceneBasedOnArrangementRequest()
        request.arrangement = arrangement
        request.robot_config.position = robotConfig
        request.armType = armType
        try:
            setSceneBasedOnArrangement_proxy = rospy.ServiceProxy("set_scene_based_on_arrangement", SetSceneBasedOnArrangement)
            setSceneBasedOnArrangement_response = setSceneBasedOnArrangement_proxy(request)
            return setSceneBasedOnArrangement_response.success
        except rospy.ServiceException as e:
            print("set_scene_based_on_arrangement service call failed: %s" % e)



class ArrNode(object):
    def __init__(self, arrangement, robotConfig, node_id, 
        transit_from_info, obj_transfer_position_indices, objectTransferred_idx, 
        transition_path, cost_to_come, parent_id, object_ordering):
        self.arrangement = arrangement
        self.robotConfig = robotConfig
        self.node_id = node_id
        ### transit_from_info indicates where does the transit path come from
        ### e.g., if the robot transits from goal position 5 of object 12
        ### then transit_from_info = [12, 5]
        self.transit_from_info = transit_from_info
        ### obj_transfer_position_indices indicates the pair of position_indices for 
        ### the object transferred before and after the transition, 
        ### e.g., the object moves from position 1 to position 3, 
        ### then obj_transfer_position_indices = [1, 3]
        self.obj_transfer_position_indices = obj_transfer_position_indices
        self.objectTransferred_idx = objectTransferred_idx
        self.transition_path = transition_path
        self.cost_to_come = cost_to_come
        self.parent_id = parent_id
        self.object_ordering = object_ordering
    
    def updateNodeID(self, node_id):
        self.node_id = node_id
    
    def updateTransitFromInfo(self, transit_from_info):
        self.transit_from_info = transit_from_info

    def updateObjTransferPositionIndices(self, obj_transfer_position_indices):
        self.obj_transfer_position_indices = obj_transfer_position_indices

    def updateObjectTransferredIdx(self, objectTransferred_idx):
        self.objectTransferred_idx = objectTransferred_idx

    def updateTransitionPath(self, transition_path):
        self.transition_path = transition_path

    def updateCostToCome(self, cost_to_come):
        self.cost_to_come = cost_to_come

    def updateParent(self, parent_id):
        self.parent_id = parent_id
    
    def updateObjectOrdering(self, object_ordering):
        self.object_ordering = object_ordering

    def getParentArr(self):
        parent_arr = copy.deepcopy(self.arrangement)
        if self.parent_id == None:
            return None
        else:
            ### move to a position before the transition
            parent_arr[self.objectTransferred_idx] = self.obj_transfer_position_indices[0]
            return parent_arr


class VirtualTreeNode(object):
    def __init__(self, arrangement, node_id, objectTransferred_idx, 
        cost_to_come, parent_id):
        self.arrangement = arrangement
        self.node_id = node_id
        self.objectTransferred_idx = objectTransferred_idx
        self.cost_to_come = cost_to_come
        self.parent_id = parent_id

    def updateNodeID(self, node_id):
        self.node_id = node_id

    def updateObjectTransferredIdx(self, objectTransferred_idx):
        self.objectTransferred_idx = objectTransferred_idx

    def updateCostToCome(self, cost_to_come):
        self.cost_to_come = cost_to_come

    def updateParent(self, parent_id):
        self.parent_id = parent_id