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

from RearrangementTaskPlanner import RearrangementTaskPlanner
from RearrangementTaskPlanner import ArrNode
from DFSDPSolver import DFSDPSolver


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class UnidirDFSDPPlanner(RearrangementTaskPlanner):
    def __init__(
        self, initial_arrangement, final_arrangement, time_allowed, isLabeledRoadmapUsed=True):
        RearrangementTaskPlanner.__init__(
            self, initial_arrangement, final_arrangement, time_allowed, isLabeledRoadmapUsed)
        rospy.logwarn("initialize an unidirectional DFSDP planner")
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
        set_scene_success = self.serviceCall_setSceneBasedOnArrangementNode(temp_node.arrangement, temp_node.robotConfig, "Right_torso")
        ### (ii) randomly select an object and then buffer
        objects_yet_to_move = [
            i for i in range(len(self.final_arrangement)) if temp_node.arrangement[i] != self.final_arrangement[i]]
        success, object_idx, buffer_idx, object_path = self.serviceCall_selectObjectAndBuffer(
                            objects_yet_to_move, self.final_arrangement, "Right_torso", self.heuristic_level, self.isLabeledRoadmapUsed)
        if success == False:
            ### the perturbation process fails either due to failure to select an object or the failure to select a buffer
            return False, None
        else:
            ### the perturbation is a success, generate a tree node for this perturbation
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
            perturbation_node = ArrNode(
                perturbed_arrangement, robot_config, node_id, transit_from_info, 
                obj_transfer_position_indices, objectTransferred_idx, transition_path,
                cost_to_come, parent_id, perturbed_object_ordering
            )
            ### before add this node to the tree, check it this resulting node is already in the tree
            isSameNodeInTheTree, same_nodeID = self.checkSameArrangementNodeInTheLeftTree(perturbation_node)
            if not isSameNodeInTheTree:
                ### then add this node in the tree
                self.left_idx += 1
                perturbation_node.updateNodeID("L"+str(self.left_idx))
                self.treeL["L"+str(self.left_idx)] = perturbation_node
                return True, perturbation_node
            else:
                return False, None


    def growSubTree(self, rootNode, target_arrangement, time_allowed, isLabeledRoadmapUsed):
        rospy.logwarn("grow a subTree at root arrangement: %s" % str(rootNode.arrangement))
        rospy.logwarn("toward to target arrangement: %s" % str(target_arrangement))
        ### (i) set the scene to the rootNode arrangement
        set_scene_success = self.serviceCall_setSceneBasedOnArrangementNode(rootNode.arrangement, rootNode.robotConfig, "Right_torso")
        ### (ii) generate the subTree
        dfsdp_solver = DFSDPSolver(
            rootNode, target_arrangement, time_allowed, isLabeledRoadmapUsed)
        local_task_success, subTree = dfsdp_solver.dfsdp_solve()
        ### (iii) engraft the subTree to the global search tree
        self.engraftingLeftTree(rootNode, subTree)

    def engraftingLeftTree(self, rootNode, subTree):
        if len(subTree) == 1:
            ### the subTree only contains the rootNode
            ### basically indicates the tree is not growing
            ### then there is nothing to engraft
            return 
        
        ### first construct a child dict
        ### so we can use BFS to traverse the tree during engrafting
        child_dict = {} ### key: parent, value: children (list)
        for node_id, arr_node in subTree.items():
            child_id = node_id
            parent_id = arr_node.parent_id
            if parent_id == None:
                continue
            if parent_id not in child_dict.keys():
                child_dict[parent_id] = []
            child_dict[parent_id].append(child_id)

        ### use BFS to add the subTree to the entire global tree structure
        idToID = OrderedDict()
        queue = [0]
        idToID[0] = rootNode.node_id
    
        while (len(queue) != 0):
            parent_id = queue.pop()
            parent_nodeID = idToID[parent_id]
            parent_arrangement = self.treeL[parent_nodeID].arrangement
            ### get all the children of this parent node
            if parent_id not in child_dict.keys():
                ### no children, this is leaf node
                children_ids = []
            else:
                children_ids = child_dict[parent_id]
            for child_id in children_ids:
                ### first check if this child arrangement has already been in the tree
                isSameNodeInTheTree, same_nodeID = self.checkSameArrangementNodeInTheLeftTree(subTree[child_id])
                if isSameNodeInTheTree:
                    idToID[child_id] = same_nodeID
                    ### we have this node in the tree before,
                    ### we don't add duplicate nodes BUT we may rewire it to a better parent
                    if self.treeL[same_nodeID].cost_to_come > subTree[child_id].cost_to_come:
                        ### It indicates that the current checked parent is a better parent since it costs less
                        ### update the corresponding infos for the child node
                        self.treeL[same_nodeID].updateCostToCome(subTree[child_id].cost_to_come)
                        self.treeL[same_nodeID].updateParent(parent_nodeID)
                        self.treeL[same_nodeID].updateObjectOrdering(subTree[child_id].object_ordering)
                else:
                    ### this is a new node to be added to the search tree
                    self.left_idx += 1
                    self.treeL["L"+str(self.left_idx)] = copy.deepcopy(subTree[child_id])
                    self.treeL["L"+str(self.left_idx)].updateNodeID("L"+str(self.left_idx))
                    self.treeL["L"+str(self.left_idx)].updateParent(parent_nodeID)
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
