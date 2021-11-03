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


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class MRSSolver(MonotoneLocalSolver):
    def __init__(self, startArrNode, target_arrangement, time_allowed, isLabeledRoadmapUsed=True):
        MonotoneLocalSolver.__init__(
            self, startArrNode, target_arrangement, time_allowed, isLabeledRoadmapUsed)
        rospy.logwarn("a MRSSolver start to work")
        
    def mrs_solve(self):
        LOCAL_TASK_SUCCESS = self.DFS()
        return LOCAL_TASK_SUCCESS, self.tree

    def DFS(self):
        '''search towards final arrangement based on current arrangement'''
        ###### return FLAG==true if the problem is solved by mRS (an indication of monotonicity) ######
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
            ### let's check this object by rearranging it
            obj_curr_position_idx = current_arrangement[obj_idx]
            obj_target_position_idx = self.target_arrangement[obj_idx]
            rearrange_success, transition_path = self.serviceCall_rearrangeCylinderObject(
                obj_idx, obj_target_position_idx, "Right_torso", isLabeledRoadmapUsed=self.isLabeledRoadmapUsed)
            if rearrange_success:
                self.generateLocalNode(current_node_id, obj_idx, transition_path)
                ### recursive call
                FLAG = self.DFS()
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
        return FLAG