#!/usr/bin/env python
from __future__ import division

import time
import sys
import os
import copy
import numpy as np
import random
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

from LazyCIRSMIX3Solver import LazyCIRSMIX3Solver


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
            self.initial_arrangement, robot_curr_config, "L0",
            None, None, None, None, 0, None, [])
        self.treeL["L0"].updateReachableStatus(True) ### the root is always reachable
        self.arrLeftRegistr.append(self.initial_arrangement)
        self.idLeftRegistr.append("L0")
        self.orderLeftRegistr.append([])
        self.leftLeaves = ["L0"] ### keep track of leaves in the left tree

        ### set the time limit
        self.time_threshold = time_allowed
        self.planning_startTime = time.time()

        ### the solution to harvest
        self.isSolved = False
        self.totalActions = np.inf ### record the total number of actions
        self.best_solution_cost = self.totalActions
        self.object_ordering = [] ### a list of obj_idx (ordered)
        self.object_paths = [] ### a list of ObjectRearrangePath paths
        self.motion_planning_time = 0 ### record the (motion planning + collision checking) time

        ##################### the global planner start to work #####################
        rospy.logwarn("initialize an unidirectional Lazy CIRSMIX version 3 planner")
        self.heuristic_level = 0

        remaining_time_allowed = self.time_threshold - (time.time() - self.planning_startTime)
        self.growSubTree(self.treeL["L0"], self.final_arrangement, remaining_time_allowed, self.isLabeledRoadmapUsed)
        while (self.isSolved == False):
            remaining_time_allowed = self.time_threshold - (time.time() - self.planning_startTime)
            if (remaining_time_allowed > 0):
                ### do a perturbation
                perturb_success, perturb_node = self.perturbNode()
                if not perturb_success: continue
                else:
                    remaining_time_allowed = self.time_threshold - (time.time() - self.planning_startTime)
                    self.growSubTree(perturb_node, self.final_arrangement, remaining_time_allowed, self.isLabeledRoadmapUsed)
            else:
                break
        
        if self.isSolved:
            self.harvestSolution()


    def perturbNode(self):
        '''this function selects a node to perturb by
           choosing an object to put on a buffer'''
        ### (i) first randomly select a node
        temp_node_id = random.choice(self.idLeftRegistr)
        temp_node = self.treeL[temp_node_id]
        ### before performing perturbation on the selected node, 
        ### we need to make sure it is valid to perform perturbation on the selected node
        if temp_node.reachable == False:
            ### we need to verify the branch from nearest reachable node to this selected node (virtual)
            isBranchValid, lastNodeValid_idx = self.verifyBranch(temp_node_id)
            if not isBranchValid:
                return False, None
        ### reach here as the selected node is reachable
        ### either (1) it is reachable when selected or (2) becomes reachable after verifying branch
        start_time = time.time()
        set_scene_success = self.serviceCall_setSceneBasedOnArrangementNode(temp_node.arrangement, temp_node.robotConfig, "Right_torso")
        self.motion_planning_time += (time.time() - start_time)
        ### (ii) randomly select an object and then buffer
        objects_yet_to_move = [
            i for i in range(len(self.final_arrangement)) if temp_node.arrangement[i] != self.final_arrangement[i]]
        start_time = time.time()
        success, object_idx, buffer_idx, object_path = self.serviceCall_selectObjectAndBuffer(
                            objects_yet_to_move, self.final_arrangement, "Right_torso", self.heuristic_level, self.isLabeledRoadmapUsed)
        self.motion_planning_time += (time.time() - start_time)
        if success == False:
            ### the perturbation process fails either due to failure to select an object or the failure to select a buffer
            return False, None
        else:
            ### the perturbation is a success, generate a tree node (actual) for this perturbation
            perturbed_arrangement = copy.deepcopy(temp_node.arrangement)
            perturbed_arrangement[object_idx] = buffer_idx
            robot_config = self.serviceCall_getCurrRobotConfig()
            node_id = 0 ### temporarily set to 0
            if temp_node.objectTransferred_idx == None:
                ### in case perturbation happens from the very initial node
                transit_from_info = None
            else:
                transit_from_info = [temp_node.objectTransferred_idx, temp_node.obj_transfer_position_indices[1]]
            obj_transfer_position_indices = [temp_node.arrangement[object_idx], buffer_idx]
            objectTransferred_idx = object_idx
            transition_path = object_path
            cost_to_come = temp_node.cost_to_come + 1
            parent_id = temp_node.node_id
            perturbed_object_ordering = copy.deepcopy(temp_node.object_ordering)
            perturbed_object_ordering = perturbed_object_ordering + [object_idx]
            perturbation_node = ArrHybridNode(
                perturbed_arrangement, robot_config, node_id, transit_from_info,
                obj_transfer_position_indices, objectTransferred_idx, transition_path,
                cost_to_come, parent_id, perturbed_object_ordering
            )
            perturbation_node.updateReachableStatus(True) ### marked it as a reachable node
            ### before add this node to the tree, check if this resulting node is already in the tree
            isSameNodeInTheTree, same_nodeID = self.checkSameArrangementNodeInTheLeftTree(perturbation_node)
            if not isSameNodeInTheTree:
                ### then add this node in the tree
                self.left_idx += 1
                perturbation_node.updateNodeID("L"+str(self.left_idx))
                self.treeL["L"+str(self.left_idx)] = perturbation_node
                return True, perturbation_node
            else:
                return False, None


    def verifyBranch(self, current_node_id):
        '''this function verifies if the solution branch on the virtual tree
           is valid or not in terms of motion planning + collision checking'''

        ### (1) get the branch from nearest-reachable-node to current_node_id on the global tree (backtrack)
        branch_to_check = []
        nodeID = current_node_id ### this should be the id for the target arrangement node
        branch_to_check.append(nodeID)
        ### back track to get the branch
        while (not self.treeL[nodeID].reachable):
            ### find its parent
            nodeID = self.treeL[nodeID].parent_id
            ### add the parent in the branch_to_check
            branch_to_check.append(nodeID)
        branch_to_check.reverse()
        # print("branch_to_check: " + str(branch_to_check))
        # input("checking time. wait here")

        ### (2) verify each edge (rearranging an object) on the branch_to_check
        ### (2.1) first set the scene to the nearest-reachable node
        reachable_node = self.treeL[branch_to_check[0]] ### actual node
        start_time = time.time()
        set_scene_success = self.serviceCall_setSceneBasedOnArrangementNode(
                    reachable_node.arrangement, reachable_node.robotConfig, "Right_torso")
        self.motion_planning_time += (time.time() - start_time)
        ### (2.2) check each edge, up to the final node
        for edge_i in range(1, len(branch_to_check)):
            parent_node_id = branch_to_check[edge_i-1]
            curr_node_id = branch_to_check[edge_i]
            curr_node = self.treeL[curr_node_id]
            obj_idx = curr_node.objectTransferred_idx
            obj_target_position_idx = curr_node.obj_transfer_position_indices[1] ### target position
            start_time = time.time()
            rearrange_success, transition_path = self.serviceCall_rearrangeCylinderObject(
                obj_idx, obj_target_position_idx, "Right_torso", isLabeledRoadmapUsed=self.isLabeledRoadmapUsed)
            self.motion_planning_time += (time.time() - start_time)
            if rearrange_success:
                ### convert the node from virtual to actual
                self.convertVirtualToActualNode(parent_node_id, curr_node_id, obj_idx, transition_path)
            else:
                ### the current branch failed at the edge (parent_node_id --> curr_node_id)
                print("fail to rearrange object " + str(obj_idx))
                ### now we need to delete the subtree rooted at curr_node_id
                self.deleteTree(curr_node_id)
                self.treeL[parent_node_id].removeChild(curr_node_id)
                return False, parent_node_id

        ### reach here as all edges lead to rearrange success
        return True, None



    def convertVirtualToActualNode(self, parent_node_id, curr_node_id, obj_idx, transition_path):
        '''This function converts a virtual node (curr_node_id) with parent (parent_node_id)
           into an actual node'''
        resulting_robot_config = self.serviceCall_getCurrRobotConfig()
        ### update these attributes to convert the virtual child node into an actual one
        self.treeL[curr_node_id].updateRobotConfig(resulting_robot_config)
        self.treeL[curr_node_id].updateTransitionPath(transition_path)
        self.treeL[curr_node_id].updateReachableStatus(True)

    def deleteTree(self, curr_node_id):
        '''delete a subtree rooted at curr_node_id (using DFS)'''
        for child_id in self.treeL[curr_node_id].child_ids:
            self.deleteTree(child_id)
        ### reach here either (1) there are no children (reach leaf node) or (2) all children has been deleted
        self.arrLeftRegistr.remove(self.treeL[curr_node_id].arrangement)
        self.idLeftRegistr.remove(self.treeL[curr_node_id].node_id)
        self.orderLeftRegistr.append(self.treeL[curr_node_id].object_ordering)
        del self.treeL[curr_node_id]


    def growSubTree(self, rootNode, target_arrangement, time_allowed, isLabeledRoadmapUsed):
        rospy.logwarn("grow a subTree at root arrangement: %s" % str(rootNode.arrangement))
        rospy.logwarn("toward to target arrangement: %s" % str(target_arrangement))
        ### (i) set the scene to the rootNode arrangement
        start_time = time.time()
        set_scene_success = self.serviceCall_setSceneBasedOnArrangementNode(rootNode.arrangement, rootNode.robotConfig, "Right_torso")
        self.motion_planning_time += (time.time() - start_time)
        ### (ii) generate the subTree
        lazy_cirsmix3_solver = LazyCIRSMIX3Solver(
            rootNode, target_arrangement, time_allowed, isLabeledRoadmapUsed)
        local_task_success, subTree, local_motion_planning_time = lazy_cirsmix3_solver.lazy_cirsmix3_solve()
        self.motion_planning_time += local_motion_planning_time
        ### (iii) engraft the subTree to the global search tree 
        ### Now both the subTree and the global tree is a hybrid (actual + virtual) tree
        self.engraftingLeftTree(rootNode, subTree)


    def engraftingLeftTree(self, rootNode, subTree):
        if len(subTree) == 1:
            ### the subTree only contains the rootNode
            ### basically indicates the tree is not growing
            ### then there is nothing to engraft
            return 

        ### use BFS to add the subTree to the entire global tree structure
        idToID = OrderedDict()
        queue = [0]
        idToID[0] = rootNode.node_id

        while (len(queue) != 0):
            parent_id = queue.pop()
            parent_nodeID = idToID[parent_id]
            parent_arrangement = self.treeL[parent_nodeID].arrangement
            ### loop through all the children of the parent node (parent_nodeID)
            for child_id in subTree[parent_id].child_ids:
                ### first check if this child arrangement has already been in the tree
                isSameNodeInTheTree, same_nodeID = self.checkSameArrangementNodeInTheLeftTree(subTree[child_id])
                if isSameNodeInTheTree:
                    idToID[child_id] = same_nodeID ### this step is CRUCIAL
                    ### different conditions to consider
                    if (subTree[child_id].reachable == True) and (self.treeL[same_nodeID].reachable == False):
                        ### (1) the node (child_id) is actual, and it has a same node in the global tree as virtual
                        self.treeL[same_nodeID].updateRobotConfig(subTree[child_id].robotConfig)
                        self.treeL[same_nodeID].updateTransitionPath(subTree[child_id].transition_path)
                        self.treeL[same_nodeID].updateReachableStatus(True)
                        self.treeL[same_nodeID].updateCostToCome(subTree[child_id].cost_to_come)
                        self.treeL[same_nodeID].updateParent(parent_nodeID)
                        self.treeL[same_nodeID].updateObjectOrdering(subTree[child_id].object_ordering)
                        self.treeL[self.treeL[same_nodeID].parent_id].removeChild(same_nodeID)
                        self.treeL[parent_nodeID].addChild(same_nodeID)
                    elif (subTree[child_id].reachable == False) and (self.treeL[same_nodeID].reachable == False):
                        ### (2) the node (child_id) is virtual, and it has a same node in the global tree as virtual
                        if self.treeL[same_nodeID].cost_to_come > subTree[child_id].cost_to_come:
                            ### It indicates that the current checked parent is a better parent since it costs less
                            ### update the corresponding infos for the child node
                            self.treeL[same_nodeID].updateCostToCome(subTree[child_id].cost_to_come)
                            self.treeL[same_nodeID].updateParent(parent_nodeID)
                            self.treeL[same_nodeID].updateObjectOrdering(subTree[child_id].object_ordering)
                            self.treeL[self.treeL[same_nodeID].parent_id].removeChild(same_nodeID)
                            self.treeL[parent_nodeID].addChild(same_nodeID)
                    elif (subTree[child_id].reachable == False) and (self.treeL[same_nodeID].reachable == True):
                        ### (3) the node (child_id) is virtual, and it has a same node in the global tree as virtual
                        pass
                    else:
                        ### (4) the node (child_id) is actual, and it has a same node in the global tree as actual
                        if self.treeL[same_nodeID].cost_to_come > subTree[child_id].cost_to_come:
                            ### It indicates that the current checked parent is a better parent since it costs less
                            ### update the corresponding infos for the child node
                            self.treeL[same_nodeID].updateCostToCome(subTree[child_id].cost_to_come)
                            self.treeL[same_nodeID].updateParent(parent_nodeID)
                            self.treeL[same_nodeID].updateObjectOrdering(subTree[child_id].object_ordering)
                            self.treeL[self.treeL[same_nodeID].parent_id].removeChild(same_nodeID)
                            self.treeL[parent_nodeID].addChild(same_nodeID)
                else:
                    ### this is a new node to be added to the search tree
                    self.left_idx += 1
                    self.treeL["L"+str(self.left_idx)] = copy.deepcopy(subTree[child_id])
                    self.treeL["L"+str(self.left_idx)].updateNodeID("L"+str(self.left_idx))
                    self.treeL["L"+str(self.left_idx)].updateParent(parent_nodeID)
                    self.treeL["L"+str(self.left_idx)].child_ids = set() ### reset child_ids
                    self.treeL[parent_nodeID].addChild("L"+str(self.left_idx)) ### add child to the parent
                    self.arrLeftRegistr.append(subTree[child_id].arrangement)
                    self.idLeftRegistr.append("L"+str(self.left_idx))
                    self.orderLeftRegistr.append(subTree[child_id].object_ordering)
                    idToID[child_id] = "L"+str(self.left_idx)
                    ### check if we reach the FINAL_ARRANGEMENT
                    if subTree[child_id].arrangement == self.final_arrangement:
                        rospy.logwarn("SOLUTION HAS BEEN FOUND")
                        self.isSolved = True
                        self.finalNodeID = "L"+str(self.left_idx)
                        return
                
                ### before move on to other children, add this child into the queue for future expansion
                queue.insert(0, child_id)
            
            ### reach here as all the children have been explored. Move on to next parent in the queue



    def checkSameArrangementNodeInTheLeftTree(self, arr_node):
        '''This function checks if an arrangement node is already in the search tree (left)
        It returns (1) same or not (bool) (2) if same, the node ID (string)'''
        arrangement = arr_node.arrangement
        objectTransferred_idx = arr_node.objectTransferred_idx
        obj_transfer_position_indices = arr_node.obj_transfer_position_indices
        transit_from_info = arr_node.transit_from_info
        ### check if this arrangement has already been in the tree
        similar_arrangement_indices = [i for i in range(len(self.arrLeftRegistr)) if self.arrLeftRegistr[i] == arrangement]
        if len(similar_arrangement_indices) == 0:
            return False, None
        for similar_arrangement_idx in similar_arrangement_indices:
            similar_arrangement_nodeID = self.idLeftRegistr[similar_arrangement_idx]
            if objectTransferred_idx == self.treeL[similar_arrangement_nodeID].objectTransferred_idx:
                if obj_transfer_position_indices == self.treeL[similar_arrangement_nodeID].obj_transfer_position_indices:
                    if transit_from_info == self.treeL[similar_arrangement_nodeID].transit_from_info:
                        ### then we can say these two arrangement are same
                        return True, similar_arrangement_nodeID
        return False, None


    def harvestSolution(self):
        '''This function is called when it indicates a solution has been found
        The function harvest the solution (solution data)'''
        nodeID = self.finalNodeID
        ### back track to get the object_ordering and object_path
        while (self.treeL[nodeID].parent_id != None):
            self.object_ordering.append(self.treeL[nodeID].objectTransferred_idx)
            self.object_paths.append(self.treeL[nodeID].transition_path)
            nodeID = self.treeL[nodeID].parent_id
        ### reverse the object_ordering and object_paths
        self.object_ordering.reverse()
        self.object_paths.reverse()
        self.totalActions = len(self.object_ordering)
        self.best_solution_cost = self.totalActions


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

    def serviceCall_selectObjectAndBuffer(self, objects_to_move, final_arrangement, armType, heuristic_level, isLabeledRoadmapUsed):
        '''call the SelectObjectAndBuffer service to
           select object and buffer'''
        rospy.wait_for_service("select_object_and_buffer")
        request = SelectObjectAndBufferRequest()
        request.objects_to_move = objects_to_move
        request.final_arrangement = final_arrangement
        request.armType = armType
        request.heuristic_level = heuristic_level
        request.isLabeledRoadmapUsed = isLabeledRoadmapUsed
        try:
            selectObjectAndBuffer_proxy = rospy.ServiceProxy("select_object_and_buffer", SelectObjectAndBuffer)
            selectObjectAndBuffer_response = selectObjectAndBuffer_proxy(request)
            return selectObjectAndBuffer_response.success, selectObjectAndBuffer_response.object_idx, \
                selectObjectAndBuffer_response.buffer_idx, selectObjectAndBuffer_response.path
        except rospy.ServiceException as e:
            print("select_object_and_buffer service call failed: %s" % e)

        


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