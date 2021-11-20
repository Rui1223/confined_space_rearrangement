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

from confined_space_rearrangement.srv import DetectInvalidArrStatesMix, DetectInvalidArrStatesMixRequest
from confined_space_rearrangement.srv import DetectInitialConstraints, DetectInitialConstraintsRequest


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class CIRSMIXSolver(MonotoneLocalSolver):
    def __init__(self, startArrNode, target_arrangement, time_allowed, isLabeledRoadmapUsed=True):
        MonotoneLocalSolver.__init__(
            self, startArrNode, target_arrangement, time_allowed, isLabeledRoadmapUsed)
        rospy.logwarn("a CIRSMIXSolver starts to work")
        self.explored = [] ### a list of arrangements which have been explored

    def cirsmix_solve(self):
        ### before the search, given start_arrangement and target_arrangement
        ### (1) detect constraints arising from initial positions
        start_time = time.time()
        self.detectInitialConstraints()
        self.motion_planning_time += (time.time() - start_time)
        ### (2) detect all invalid arrangement at which each object to be manipulated
        self.detectInvalidArrStates_mix()
        LOCAL_TASK_SUCCESS = self.CIDFS_DP()
        return LOCAL_TASK_SUCCESS, self.tree, self.motion_planning_time

    def detectInitialConstraints(self):
        '''This function detects constraints between 
        object initial positions ang related grasp poses in the local task'''
        self.serviceCall_detectInitialConstraints()

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

    def CIDFS_DP(self):
        '''search towards final arrangement based on current arrangement'''
        ###### return FLAG==true if the problem is solved by CIDFS_DP (an indication of monotonicity) ######
        current_node_id = copy.deepcopy(self.node_idx)
        current_arrangement = self.tree[current_node_id].arrangement
        current_ordering = self.tree[current_node_id].object_ordering

        ### first check if we touch the base case: we are at the target_arrangement
        if (current_arrangement == self.target_arrangement):
            ### the problem is solved
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
                # print("I see the current arrangement {} is invalid for {} to be manipulated now: ".format(current_arrangement, obj_idx))
                # input("check if it is really the case...")
                continue
            ### otherwise, let's check this object by rearranging it
            obj_curr_position_idx = current_arrangement[obj_idx]
            obj_target_position_idx = self.target_arrangement[obj_idx]
            start_time = time.time()
            rearrange_success, transition_path = self.serviceCall_rearrangeCylinderObject(
                obj_idx, obj_target_position_idx, "Right_torso", isLabeledRoadmapUsed=self.isLabeledRoadmapUsed)
            self.motion_planning_time += (time.time() - start_time)
            # print("\n====================")
            # print("rearranging_success: ", rearrange_success)
            # input("enter to continue")
            # print("====================\n")
            if rearrange_success:
                self.generateLocalNode(current_node_id, obj_idx, transition_path)
                ### recursive call
                FLAG = self.CIDFS_DP()
                if FLAG:
                    return FLAG
                else:
                    ### first check if FLAG == False is due to timeout, if it is, just return
                    if time.time() - self.local_planning_startTime >= self.time_threshold:
                        return FLAG
                    ### put the object and robot back to the configuration they belong to
                    ### at the beginning of the function call
                    self.revertBackToParentNode(current_node_id, obj_idx, obj_curr_position_idx, "Right_torso")
            else:
                ### put the object and robot back to the configuration they belong to
                ### at the beginning of the function call
                self.revertBackToParentNode(current_node_id, obj_idx, obj_curr_position_idx, "Right_torso")

        ### the problem is not solved but there is no option
        ### the current arrangement is not the right parent
        ### from which a solution can be found, mark it as exlored
        self.explored.append(current_arrangement)
        # print("backtrack")
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