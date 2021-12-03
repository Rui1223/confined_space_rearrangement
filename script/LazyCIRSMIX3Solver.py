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
from confined_space_rearrangement.srv import DetectInvalidArrStatesMix, DetectInvalidArrStatesMixRequest
from confined_space_rearrangement.srv import DetectInitialConstraints, DetectInitialConstraintsRequest


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class LazyCIRSMIX3Solver(object):
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
        self.tree[0] = copy.deepcopy(startArrNode) ### startArrNode will always be an ACTUAL node (DEEP COPY)

        ### set the time limit
        self.time_threshold = time_allowed
        self.local_planning_startTime = time.time()

        ### record the (motion planning + collision checking) time
        self.motion_planning_time = 0

        ##################### the local solver start to work #####################
        rospy.logwarn("a lazy CIRSMIXSolver version 3 starts to work")
        self.explored = [] ### a list of arrangements which have been explored
        ### this container includes all the nodes which are already in the actual tree,
        ### which are reachable from the local root
        self.reachable = [0]
        ### you will have two modes in the DFS_DP tree search
        ### (1) back-tracking (self.backTracking = True)
        ### (2) back-jumping (self.backTracking = False)
        self.backTracking = True ### initially it is back-tracking stage

    def lazy_cirsmix3_solve(self):
        ### before the search, given start_arrangement and target_arrangement
        ### (1) detect constraints arising from initial positions
        start_time = time.time()
        self.detectInitialConstraints()
        self.motion_planning_time += (time.time() - start_time)
        ### (2) detect all invalid arrangement at which each object to be manipulated
        self.detectInvalidArrStates_mix()
        LOCAL_TASK_SUCCESS = self.LAZY_CIDFS_DP3()
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

    def LAZY_CIDFS_DP3(self):
        '''search towards final arrangement based on current arrangement'''
        '''generate a virtual tree'''
        ###### return FLAG==true if the final arrangement can be reached by the virtual tree ######
        current_node_id = copy.deepcopy(self.node_idx)
        current_arrangement = self.tree[current_node_id].arrangement
        # current_ordering = self.tree[current_node_id].object_ordering

        ### first check if we touch the base case: we are at the target_arrangement
        if (current_arrangement == self.target_arrangement):
            ### at least the tree is connected from the local root to the target arrangement via virtual connection
            isSolutionValid, self.lastNodeValid_idx = self.verifySolutionBranch(current_node_id)
            if isSolutionValid:
                return True
            else:
                ### Now you should enter into back-jumping mode
                ### jump to the last valid, reachable node (lastNodeValid_idx)
                self.backTracking = False
                return False
        ### otherwise the virtual tree is yet connected to the target arrangement. Check if time exceeds
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
            FLAG = self.LAZY_CIDFS_DP3()
            if FLAG:
                return FLAG
            else:
                ### first check if FLAG == False is due to timeout, if it is, just return
                if time.time() - self.local_planning_startTime >= self.time_threshold:
                    return FLAG
                ### Now we need to see if this is a backtracking or a backjumping condition
                if self.backTracking == True:
                    pass ### keep searching for other child nodes
                else:
                    ### the search is in the backjumping mode
                    ### check if the current node is the last valid, reachable node (self.lastNodeValid_idx)
                    if (current_node_id == self.lastNodeValid_idx):
                        ### back-tracking start from here
                        self.backTracking = True
                        pass
                    else:
                        ### this node is not the last valid, reachable node (self.lastNodeValid_idx)
                        print("back-jumping")
                        return FLAG


        ### the problem is not solved but there is no option
        ### the current arrangement is not the right parent
        ### from which a solution can be found, mark it as explored
        self.explored.append(current_arrangement)
        print("backtrack")
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
        current_arrangement = self.tree[current_node_id].arrangement
        resulting_arrangement = copy.deepcopy(current_arrangement)
        resulting_arrangement[obj_idx] = self.target_arrangement[obj_idx]
        if self.tree[current_node_id].objectTransferred_idx == None:
            ### parent node (current_node_id) is the root node, no transit info can be obtained in this case
            resulting_transit_from_info = None
        else:
            resulting_transit_from_info = [
                self.tree[current_node_id].objectTransferred_idx, \
                self.tree[current_node_id].obj_transfer_position_indices[1]]
        resulting_obj_transfer_position_indices = [self.tree[current_node_id].arrangement[obj_idx], self.target_arrangement[obj_idx]]
        resulting_cost_to_come = self.tree[current_node_id].cost_to_come + 1
        ### also add ordering info to the virtual node
        resulting_object_ordering = self.tree[current_node_id].object_ordering + [obj_idx]
        ### add this newly-generated node
        self.node_idx += 1
        self.tree[self.node_idx] = ArrHybridNode(
            resulting_arrangement, None, self.node_idx, \
            resulting_transit_from_info, resulting_obj_transfer_position_indices, obj_idx, None, \
            resulting_cost_to_come, current_node_id, resulting_object_ordering)
        ### mark this new node as a child node (self.node_idx) of the parent node (current_node_id)
        self.tree[current_node_id].addChild(self.node_idx)

        # print("object to move: " + str(obj_idx))
        # print("parent node id: " + str(current_node_id))
        # print("new node id: " + str(self.node_idx))
        # print("\n")


    def verifySolutionBranch(self, current_node_id):
        '''this function verifies if the solution branch on the virtual tree
           is valid or not in terms of motion planning + collision checking'''

        ### (1) get the branch from nearest-reachable-node to current_node_id on the virtual tree (backtrack)
        branch_to_check = []
        nodeID = current_node_id ### this should be the id for the target arrangement node
        branch_to_check.append(nodeID)
        ### back track to get the branch
        while (nodeID not in self.reachable):
            ### find its parent
            nodeID = self.tree[nodeID].parent_id
            ### add the parent in the branch_to_check
            branch_to_check.append(nodeID)
        branch_to_check.reverse()

        # print("branch_to_check: " + str(branch_to_check))
        # input("checking time. wait here")

        ### (2) verify each edge (rearranging an object) on the branch_to_check
        ### (2.1) first set the scene to the nearest-reachable node
        reachable_node = self.tree[branch_to_check[0]] ### actual node
        start_time = time.time()
        set_scene_success = self.serviceCall_setSceneBasedOnArrangementNode(
                    reachable_node.arrangement, reachable_node.robotConfig, "Right_torso")
        self.motion_planning_time += (time.time() - start_time)
        ### (2.2) check each edge, up to the final node
        for edge_i in range(1, len(branch_to_check)):
            parent_node_id = branch_to_check[edge_i-1]
            curr_node_id = branch_to_check[edge_i]
            curr_node = self.tree[curr_node_id]
            obj_idx = curr_node.objectTransferred_idx
            obj_target_position_idx = self.target_arrangement[obj_idx]
            start_time = time.time()
            rearrange_success, transition_path = self.serviceCall_rearrangeCylinderObject(
                obj_idx, obj_target_position_idx, "Right_torso", isLabeledRoadmapUsed=self.isLabeledRoadmapUsed)
            self.motion_planning_time += (time.time() - start_time)
            if rearrange_success:
                self.convertVirtualToActualNode(parent_node_id, curr_node_id, obj_idx, transition_path)
            else:
                ### the current branch failed at the edge (parent_node_id --> curr_node_id)
                print("fail to rearrange object " + str(obj_idx))
                ### now we need to delete the subtree rooted at curr_node_id
                self.deleteTree(curr_node_id)
                self.tree[parent_node_id].removeChild(curr_node_id)
                return False, parent_node_id

        ### reach here as all edges lead to rearrange success
        return True, None


    def convertVirtualToActualNode(self, parent_node_id, curr_node_id, obj_idx, transition_path):
        '''This function converts a virtual node (curr_node_id) with parent (parent_node_id)
           into an actual node'''
        resulting_robot_config = self.serviceCall_getCurrRobotConfig()
        ### update these attributes to convert the virtual child node into an actual one
        self.tree[curr_node_id].updateRobotConfig(resulting_robot_config)
        self.tree[curr_node_id].updateTransitionPath(transition_path)
        self.tree[curr_node_id].updateReachableStatus(True)
        self.reachable.append(curr_node_id)

    def deleteTree(self, curr_node_id):
        '''delete a subtree rooted at curr_node_id (using DFS)'''
        for child_id in self.tree[curr_node_id].child_ids:
            self.deleteTree(child_id)
        ### reach here either (1) there are no children (reach leaf node) or (2) all children has been deleted
        del self.tree[curr_node_id]


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
    
    def updateRobotConfig(self, robotConfig):
        self.robotConfig = robotConfig
    
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