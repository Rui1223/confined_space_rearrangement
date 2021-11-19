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

from MonotoneLocalSolver import MonotoneLocalSolver
from MonotoneLocalSolver import ArrNode
from MonotoneLocalSolver import VirtualTreeNode

from confined_space_rearrangement.srv import DetectInvalidArrStatesMix, DetectInvalidArrStatesMixRequest
from confined_space_rearrangement.srv import DetectInitialConstraints, DetectInitialConstraintsRequest


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class LazyCIRSMIXSolver(MonotoneLocalSolver):
    def __init__(self, startArrNode, target_arrangement, time_allowed, isLabeledRoadmapUsed=True):
        MonotoneLocalSolver.__init__(
            self, startArrNode, target_arrangement, time_allowed, isLabeledRoadmapUsed)
        rospy.logwarn("a lazy CIRSMIXSolver starts to work")
        self.explored = [] ### a list of arrangements which have been explored
        self.virtual_tree = OrderedDict() ### key: (scalar 0,1,etc..) value: VirtualNode
        self.virtual_node_idx = 0 ### start from root node (idx: 0)
        self.virtual_tree[0] = VirtualTreeNode(
            startArrNode.arrangement, 0, None, 0, None)

    def lazy_cirsmix_solve(self):
        ### before the search, given start_arrangement and target_arrangement
        ### (1) detect constraints arising from initial positions
        self.detectInitialConstraints()
        ### (2) detect all invalid arrangement at which each object to be manipulated
        self.detectInvalidArrStates_mix()
        LOCAL_TASK_SUCCESS = self.LAZY_CIDFS_DP()
        ########### ******************************** ###########
        # return LOCAL_TASK_SUCCESS, self.tree
        if not LOCAL_TASK_SUCCESS:
            self.finalNodeID = 0
        return LOCAL_TASK_SUCCESS, self.virtual_tree, self.finalNodeID
        ########### ******************************** ###########

    def detectInitialConstraints(self):
        '''This function detects constraints between 
        object initial positions ang related grasp poses in the local task'''
        start_time = time.time()
        self.serviceCall_detectInitialConstraints()
        print("total time cost: " + str(time.time() - start_time))

    def detectInvalidArrStates_mix(self):
        '''This function detects all invalid states of arrangement
        at which each object is manipulated in the local task'''
        # start_time = time.time()
        ### self.invalid_arr_states_per_obj has the following format
        ### {obj_idx: [{invalid_arr1}, {invalid_arr2}, {invalid_arr3}], ...}
        self.invalid_arr_states_per_obj = {}
        all_obj_invalid_states = self.serviceCall_detectInvalidArrStatesMix()
        for obj_arr_states_msg in all_obj_invalid_states:
            self.invalid_arr_states_per_obj[obj_arr_states_msg.obj_idx] = []
            for invalid_arr_state_msg in obj_arr_states_msg.invalid_arr_states:
                arr_state = {}
                for (obj_idx, isAtTarget) in zip(invalid_arr_state_msg.obj_indices, invalid_arr_state_msg.isAtTarget):
                    arr_state[obj_idx] = isAtTarget
                self.invalid_arr_states_per_obj[obj_arr_states_msg.obj_idx].append(arr_state)

        # print("invalid_arr_states_per_obj: ")
        # for obj_idx, arr_states in self.invalid_arr_states_per_obj.items():
        #     print(obj_idx)
        #     print(arr_states)
        # input("Press to continue...")

    def LAZY_CIDFS_DP(self):
        '''search towards final arrangement based on current arrangement'''
        '''generate a virtual tree'''
        ###### return FLAG==true if the final arrangement can be reached by the virtual tree ######
        current_node_id = copy.deepcopy(self.virtual_node_idx)
        current_arrangement = self.virtual_tree[current_node_id].arrangement
        # current_ordering = self.virtual_tree[current_node_id].object_ordering

        ### first check if we touch the base case: we are at the target_arrangement
        if (current_arrangement == self.target_arrangement):
            ### the problem is solved
            ########### ******************************** ###########
            self.finalNodeID = current_node_id
            ########### ******************************** ###########
            return True
        ### otherwise it's not solved yet. Check if time exceeds
        if time.time() - self.local_planning_startTime >= self.time_threshold:
            return False

        FLAG = False
        remaining_objects = [i for i in range(len(current_arrangement)) \
                            if current_arrangement[i] != self.target_arrangement[i]]
        for obj_idx in remaining_objects:
            ### first check if the resulting arrangement after rearranging object obj_idx
            ### has been explored before
            resulting_arrangement = copy.deepcopy(current_arrangement)
            resulting_arrangement[obj_idx] = self.target_arrangement[obj_idx]
            if resulting_arrangement in self.explored:
                ### this resulting arrangement has been explored before and
                ### turns out to be failure, so no need to do it again
                continue
            ### otherwise, this object obj_idx has not been considered
            ### BUT BEFORE REARRANGE THIS OBJECT, 
            ### check if current_arrangement belongs to one of invalid
            ### arr states for the object to be manipulated
            if self.checkInvalidArrStates(current_arrangement, obj_idx):
                ### this is not the right time to rearrange that object
                # print("I see the action of rearranging {} is invalid at current arrangement {}: ".format(obj_idx, current_arrangement))
                # input("check if it is really the case...")
                continue
            ### otherwise, this object is rearrangable: (1) follow DFS_DP (2) no invalid action 
            ### generate a virtual node for the resulting arrangement
            self.generateVirtualNode(current_node_id, obj_idx)
            ### recursive call
            FLAG = self.LAZY_CIDFS_DP()
            if FLAG:
                return FLAG
            else:
                ### first check if FLAG == False is due to timeout, if it is, just return
                if time.time() - self.local_planning_startTime >= self.time_threshold:
                    return FLAG
        
        ### the problem is not solved but there is no option
        ### the current arrangement is not the right parent
        ### from which a solution can be found, mark it as explored
        self.explored.append(current_arrangement)
        ########### ******************************** ###########
        print("backtrack")
        ########### ******************************** ###########
        return FLAG

    def checkInvalidArrStates(self, current_arrangement, obj_idx):
        for invalid_arr_state in self.invalid_arr_states_per_obj[obj_idx]:
            isInvalid = True
            for obj, isAtTarget in invalid_arr_state.items():
                if (isAtTarget == True and current_arrangement[obj] == self.target_arrangement[obj]) or \
                    (isAtTarget == False and current_arrangement[obj] != self.target_arrangement[obj]):
                    pass
                else:
                    isInvalid = False
                    break
            ### reach here as you have already finish checking this arr_state
            if isInvalid:
                ### this current_arrangement belongs to this arr_state, no need to check other ones
                return isInvalid
            else:
                ### this current_arrangement doesn't belong to this arr_state, continue to check
                pass
        ### if you reach here, you finish checking current_arrangement for all invalid arr states
        ### and current_arrangement does not belong to any invalid arr state
        return False


    def serviceCall_detectInvalidArrStatesMix(self):
        rospy.wait_for_service("detect_invalid_arr_states_mix")
        request = DetectInvalidArrStatesMixRequest()
        request.start_arrangement = self.start_arrangement
        request.target_arrangement = self.target_arrangement
        try:
            detectInvalidArrStatesMix_proxy = rospy.ServiceProxy(
                "detect_invalid_arr_states_mix", DetectInvalidArrStatesMix)
            detect_invalid_arr_states_mix_response = detectInvalidArrStatesMix_proxy(request)
            return detect_invalid_arr_states_mix_response.all_obj_invalid_arr_states
        except rospy.ServiceException as e:
            print("detect_invalid_arr_states_mix service call failed: %s" % e)

    def serviceCall_detectInitialConstraints(self):
        rospy.wait_for_service("detect_initial_constraints")
        request = DetectInitialConstraintsRequest()
        request.start_arrangement = self.start_arrangement
        request.target_arrangement = self.target_arrangement
        try:
            detectInitialConstraints_proxy = rospy.ServiceProxy(
                "detect_initial_constraints", DetectInitialConstraints)
            detect_initial_constraints_response = detectInitialConstraints_proxy(request)
            return detect_initial_constraints_response.success
        except rospy.ServiceException as e:
            print("detect_initial_constraints service call failed: %s" % e)


    def generateVirtualNode(self, current_node_id, obj_idx):
        '''generate a virtual node which has parent node id == current_node_id given the obj_idx'''
        current_arrangement = self.virtual_tree[current_node_id].arrangement
        resulting_arrangement = copy.deepcopy(current_arrangement)
        resulting_arrangement[obj_idx] = self.target_arrangement[obj_idx]
        resulting_cost_to_come = self.virtual_tree[current_node_id].cost_to_come + 1
        ### add this newly-generated node
        self.virtual_node_idx += 1
        self.virtual_tree[self.virtual_node_idx] = VirtualTreeNode(
            resulting_arrangement, self.virtual_node_idx, obj_idx,
            resulting_cost_to_come, current_node_id)
        ########### ******************************** ###########
        print("object to move: " + str(obj_idx))
        print("parent node id: " + str(current_node_id))
        print("new node id: " + str(self.virtual_node_idx))
        print("\n")
        ########### ******************************** ###########
