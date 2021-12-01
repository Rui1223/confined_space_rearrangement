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
from confined_space_rearrangement.srv import SelectObjectAndBuffer, SelectObjectAndBufferRequest


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class UnidirLazyCIRSMIX3Planner(object):
    def __init__(
        self, initial_arrangement, final_arrangement, time_allowed, isLabeledRoadmapUsed=True):

        ### understand the arrangement task
        self.initial_arrangement = initial_arrangement
        self.final_arrangement = final_arrangement
        #### a list of obj_idx of objects to be rearranged
        self.all_objects = [i for i in range(len(self.initial_arrangement)) \
            if self.initial_arrangement[i] != self.final_arrangement[i]]
        self.num_objects = len(self.all_objects)
        self.isLabeledRoadmapUsed = isLabeledRoadmapUsed

        ### initialize the tree structure
        self.treeL = OrderedDict() ### key: ("L0", etc.) value: ArrHybridNode
        self.trees = {}
        self.trees["Left"] = self.treeL
        self.arrLeftRegistr = []
        self.idLeftRegistr = []
        self.orderLeftRegistr = []
        ### add the initial_arrangement as the root node for the left tree
        robot_curr_config = self.serviceCall_getCurrRobotConfig()
        self.left_idx = 0
        self.treeL["L0"] = ArrHybridNode(
            self.initial_arrangement, robotConfig, "L0",
            None, None, None, None, 0, None, [])
        self.treeL["L0"].updateReachableStatus(True) ### the root is always reachable
        self.arrLeftRegistr.append(self.initial_arrangement)
        self.idLeftRegistr.append("L0")
        self.orderLeftRegistr.append([])
        self.leftLeaves = ["L0"] ### keep track of leaves in the left tree

        ### set the time limit
        self.time_threshold = time_allowed
        self.planning_startTime = time.time()

        


class ArrHybridNode(object):
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
        ### more attributes for hybrid (actual + virtual) tree nodes
        self.child_ids = set() ### now store child_ids
        self.reachable = False
    
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

    ### more API functions for hybrid (actual + virtual) tree nodes
    def addChild(self, child_id):
        self.child_ids.add(child_id)

    def removeChild(self, child_id):
        self.child_ids.remove(child_id)

    def updateReachableStatus(self, isReachable):
        self.reachable = isReachable