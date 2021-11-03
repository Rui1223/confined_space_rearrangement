#!/usr/bin/env python
from __future__ import division

import time
import sys
import os
import numpy as np

import utils2

import rospy
import rospkg

from UnidirMRSPlanner import UnidirMRSPlanner
from UnidirDFSDPPlanner import UnidirDFSDPPlanner
from UnidirCIRSPlanner import UnidirCIRSPlanner

############################### description ###########################################
### This class defines a SideExperimenter class which
### conducts large-scale experiments with different methods (labeled vs. unlabeled)
### with the number of experiments specified for each case (#objects)
### It
### (1) asks the execution scene to generate an instance
### (2) asks the pose estimator to get the object poses
### (3) reproduces the instance in task planner 
### (4) solves it with all methods
### (5) collect statistics (e.g., time/actions) from each method for comparison purpose
#######################################################################################

class SideExperimenter(object):

    def __init__(self, args):
        ### set the rospkg path
        rospack = rospkg.RosPack()
        self.rosPackagePath = rospack.get_path("confined_space_rearrangement")
        self.ExperimentsFolder = os.path.join(self.rosPackagePath, "side_experiments")
        if not os.path.exists(self.ExperimentsFolder):
            os.makedirs(self.ExperimentsFolder)
        self.numObjects_options = [7,8,9,10]
        self.numExperiments_perObject = int(args[1])
        self.maxInstancesNeed_perObject = int(args[2])
        self.time_allowed = int(args[3])


    def createNumObjectsFolder(self, num_objects):
        ### create a folder denoted with specified num_objects
        self.objectFolder = os.path.join(self.ExperimentsFolder, str(num_objects))
        if not os.path.exists(self.objectFolder):
            os.makedirs(self.objectFolder)

    def initializeObjLevelStat(self):
        ### initialize obj-level statistics variable
        self.CIRS_time_obj = []
        self.CIRS_nonlabeled_time_obj = []
        self.DFSDP_time_obj = []
        self.DFSDP_nonlabeled_time_obj = []
        self.mRS_time_obj = []
        self.mRS_nonlabeled_time_obj = []
        self.CIRS_success_obj = []
        self.CIRS_nonlabeled_success_obj = []
        self.DFSDP_success_obj = []
        self.DFSDP_nonlabeled_success_obj = []
        self.mRS_success_obj = []
        self.mRS_nonlabeled_success_obj = []
        self.CIRS_nActions_obj = []
        self.CIRS_nonlabeled_nActions_obj = []
        self.DFSDP_nActions_obj = []
        self.DFSDP_nonlabeled_nActions_obj = []
        self.mRS_nActions_obj = []
        self.mRS_nonlabeled_nActions_obj = []

    def saveAverageSolutionPerNumObject(self):
        ################### average time ###################
        all_methods_average_time_obj = []
        if len(self.CIRS_time_obj) != 0:
            average_cirs_time_obj = sum(self.CIRS_time_obj) / len(self.CIRS_time_obj)
        else:
            average_cirs_time_obj = 10000
        all_methods_average_time_obj.append(average_cirs_time_obj)
        if len(self.CIRS_nonlabeled_time_obj) != 0:
            average_cirs_nonlabeled_time_obj = sum(self.CIRS_nonlabeled_time_obj) / len(self.CIRS_nonlabeled_time_obj)
        else:
            average_cirs_nonlabeled_time_obj = 10000
        all_methods_average_time_obj.append(average_cirs_nonlabeled_time_obj)
        if len(self.DFSDP_time_obj) != 0:
            average_DFSDP_time_obj = sum(self.DFSDP_time_obj) / len(self.DFSDP_time_obj)
        else:
            average_DFSDP_time_obj = 10000
        all_methods_average_time_obj.append(average_DFSDP_time_obj)
        if len(self.DFSDP_nonlabeled_time_obj) != 0:
            average_DFSDP_nonlabeled_time_obj = sum(self.DFSDP_nonlabeled_time_obj) / len(self.DFSDP_nonlabeled_time_obj)
        else:
            average_DFSDP_nonlabeled_time_obj = 10000
        all_methods_average_time_obj.append(average_DFSDP_nonlabeled_time_obj)
        if len(self.mRS_time_obj) != 0:
            average_mRS_time_obj = sum(self.mRS_time_obj) / len(self.mRS_time_obj)
        else:
            average_mRS_time_obj = 10000
        all_methods_average_time_obj.append(average_mRS_time_obj)
        if len(self.mRS_nonlabeled_time_obj) != 0:
            average_mRS_nonlabeled_time_obj = sum(self.mRS_nonlabeled_time_obj) / len(self.mRS_nonlabeled_time_obj)
        else:
            average_mRS_nonlabeled_time_obj = 10000
        all_methods_average_time_obj.append(average_mRS_nonlabeled_time_obj)
        ################### average success ###################
        all_methods_average_success_obj = []
        if len(self.CIRS_success_obj) != 0:
            average_cirs_success_obj = sum(self.CIRS_success_obj) / len(self.CIRS_success_obj)
        else:
            average_cirs_success_obj = 0.0
        all_methods_average_success_obj.append(average_cirs_success_obj)
        if len(self.CIRS_nonlabeled_success_obj) != 0:
            average_cirs_nonlabeled_success_obj = sum(self.CIRS_nonlabeled_success_obj) / len(self.CIRS_nonlabeled_success_obj)
        else:
            average_cirs_nonlabeled_success_obj = 0.0
        all_methods_average_success_obj.append(average_cirs_nonlabeled_success_obj)
        if len(self.DFSDP_success_obj) != 0:
            average_DFSDP_success_obj = sum(self.DFSDP_success_obj) / len(self.DFSDP_success_obj)
        else:
            average_DFSDP_success_obj = 0.0
        all_methods_average_success_obj.append(average_DFSDP_success_obj)
        if len(self.DFSDP_nonlabeled_success_obj) != 0:
            average_DFSDP_nonlabeled_success_obj = sum(self.DFSDP_nonlabeled_success_obj) / len(self.DFSDP_nonlabeled_success_obj)
        else:
            average_DFSDP_nonlabeled_success_obj = 0.0
        all_methods_average_success_obj.append(average_DFSDP_nonlabeled_success_obj)
        if len(self.mRS_success_obj) != 0:
            average_mRS_success_obj = sum(self.mRS_success_obj) / len(self.mRS_success_obj)
        else:
            average_mRS_success_obj = 0.0
        all_methods_average_success_obj.append(average_mRS_success_obj)
        if len(self.mRS_nonlabeled_success_obj) != 0:
            average_mRS_nonlabeled_success_obj = sum(self.mRS_nonlabeled_success_obj) / len(self.mRS_nonlabeled_success_obj)
        else:
            average_mRS_nonlabeled_success_obj = 0.0
        all_methods_average_success_obj.append(average_mRS_nonlabeled_success_obj)
        ################### average nActions ###################
        all_methods_average_nActions_obj = []
        if len(self.CIRS_nActions_obj) != 0:
            average_cirs_nActions_obj = sum(self.CIRS_nActions_obj) / len(self.CIRS_nActions_obj)
        else:
            average_cirs_nActions_obj = 0.0
        all_methods_average_nActions_obj.append(average_cirs_nActions_obj)
        if len(self.CIRS_nonlabeled_nActions_obj) != 0:
            average_cirs_nonlabeled_nActions_obj = sum(self.CIRS_nonlabeled_nActions_obj) / len(self.CIRS_nonlabeled_nActions_obj)
        else:
            average_cirs_nonlabeled_nActions_obj = 0.0
        all_methods_average_nActions_obj.append(average_cirs_nonlabeled_nActions_obj)
        if len(self.DFSDP_nActions_obj) != 0:
            average_DFSDP_nActions_obj = sum(self.DFSDP_nActions_obj) / len(self.DFSDP_nActions_obj)
        else:
            average_DFSDP_nActions_obj = 0.0
        all_methods_average_nActions_obj.append(average_DFSDP_nActions_obj)
        if len(self.DFSDP_nonlabeled_nActions_obj) != 0:
            average_DFSDP_nonlabeled_nActions_obj = sum(self.DFSDP_nonlabeled_nActions_obj) / len(self.DFSDP_nonlabeled_nActions_obj)
        else:
            average_DFSDP_nonlabeled_nActions_obj = 0.0
        all_methods_average_nActions_obj.append(average_DFSDP_nonlabeled_nActions_obj)
        if len(self.mRS_nActions_obj) != 0:
            average_mRS_nActions_obj = sum(self.mRS_nActions_obj) / len(self.mRS_nActions_obj)
        else:
            average_mRS_nActions_obj = 0.0
        all_methods_average_nActions_obj.append(average_mRS_nActions_obj)
        if len(self.mRS_nonlabeled_nActions_obj) != 0:
            average_mRS_nonlabeled_nActions_obj = sum(self.mRS_nonlabeled_nActions_obj) / len(self.mRS_nonlabeled_nActions_obj)
        else:
            average_mRS_nonlabeled_nActions_obj = 0.0
        all_methods_average_nActions_obj.append(average_mRS_nonlabeled_nActions_obj)
        ### save average results
        utils2.saveSolution(
            all_methods_average_time_obj, all_methods_average_success_obj, all_methods_average_nActions_obj, self.objectFolder)


    def rosInit(self):
        ### This function specifies the role of a node instance for this class ###
        ### and initializes a ros node
        rospy.init_node("side_experiments", anonymous=True)


def main(args):
    side_experimenter = SideExperimenter(args)
    side_experimenter.rosInit()
    rate = rospy.Rate(10) ### 10hz

    for num_objects in side_experimenter.numObjects_options:
        ### create a folder for current num_objects
        side_experimenter.createNumObjectsFolder(num_objects)
        side_experimenter.initializeObjLevelStat()
        num_monotoneInstancesSaved = 0

        for experiment_id in range(1, side_experimenter.numExperiments_perObject+1):
            all_methods_time_instance = []
            all_methods_success_instance = []
            all_methods_nActions_instance = []
            ### first see if we already have enough instances
            if (num_monotoneInstancesSaved >= side_experimenter.maxInstancesNeed_perObject): break
            ### generate an instance in the execution scene
            initialize_instance_success = utils2.serviceCall_generateInstanceCylinder(
                                                num_objects, num_monotoneInstancesSaved+1, True)
            if not initialize_instance_success: continue
            ### object pose estimation
            cylinder_objects = utils2.serviceCall_cylinderPositionEstimate()
            ### reproduce the estimated object poses in the planning scene
            initial_arrangement, final_arrangement, reproduce_instance_success = \
                    utils2.serviceCall_reproduceInstanceCylinder(cylinder_objects)
            ### generate IK config for start positions for all objects
            ik_generate_success = utils2.serviceCall_generateConfigsForStartPositions("Right_torso")

            ########################## now using different methods to solve the instance ##########################
            ### use CIRS method first just to make sure
            ### (1) if the instance is monotone, all the methods will be compared
            ### (2) if the instance is non-monotone, no comparison will be made

            ### (i) CIRS
            start_time = time.time()
            unidir_cirs_planner = UnidirCIRSPlanner(
                initial_arrangement, final_arrangement, side_experimenter.time_allowed)
            cirs_planning_time = time.time() - start_time
            cirs_isSolved = unidir_cirs_planner.isSolved
            cirs_nActions = unidir_cirs_planner.best_solution_cost

            if (not cirs_isSolved) or (cirs_isSolved and cirs_nActions > num_objects):
                ### the problem is not monotone, clear the instance and move on to the next instance
                clear_instance_success = utils2.clearInstance("Right_torso")
                # input("check the instance clearance!!!")
                continue

            ### otherwise, the problem is monotone and is solved by CIRS
            side_experimenter.CIRS_time_obj.append(cirs_planning_time)
            side_experimenter.CIRS_success_obj.append(float(cirs_isSolved))
            if cirs_nActions != np.inf:
                side_experimenter.CIRS_nActions_obj.append(cirs_nActions)
            else:
                cirs_nActions = 5000
            all_methods_time_instance.append(cirs_planning_time)
            all_methods_success_instance.append(float(cirs_isSolved))
            all_methods_nActions_instance.append(cirs_nActions)

            #####################################################################
            reset_instance_success = utils2.resetInstance("Right_torso")
            #####################################################################

            ###### try other methods now ######
            ### (ii) CIRS_nonlabeled
            start_time = time.time()
            unidir_cirs_planner = UnidirCIRSPlanner(
                initial_arrangement, final_arrangement, side_experimenter.time_allowed,
                isLabeledRoadmapUsed=False)
            cirs_nonlabeled_planning_time = time.time() - start_time
            cirs_nonlabeled_isSolved = unidir_cirs_planner.isSolved
            cirs_nonlabeled_nActions = unidir_cirs_planner.best_solution_cost
            side_experimenter.CIRS_nonlabeled_time_obj.append(cirs_nonlabeled_planning_time)
            side_experimenter.CIRS_nonlabeled_success_obj.append(float(cirs_nonlabeled_isSolved))
            if cirs_nonlabeled_nActions != np.inf:
                side_experimenter.CIRS_nonlabeled_nActions_obj.append(cirs_nonlabeled_nActions)
            else:
                cirs_nonlabeled_nActions = 5000
            all_methods_time_instance.append(cirs_nonlabeled_planning_time)
            all_methods_success_instance.append(float(cirs_nonlabeled_isSolved))
            all_methods_nActions_instance.append(cirs_nonlabeled_nActions)

            #####################################################################
            reset_instance_success = utils2.resetInstance("Right_torso")
            #####################################################################

            ### (iii) DFSDP
            start_time = time.time()
            unidir_dfsdp_planner = UnidirDFSDPPlanner(
                initial_arrangement, final_arrangement, side_experimenter.time_allowed)
            DFSDP_planning_time = time.time() - start_time
            DFSDP_isSolved = unidir_dfsdp_planner.isSolved
            DFSDP_nActions = unidir_dfsdp_planner.best_solution_cost
            side_experimenter.DFSDP_time_obj.append(DFSDP_planning_time)
            side_experimenter.DFSDP_success_obj.append(float(DFSDP_isSolved))
            if DFSDP_nActions != np.inf:
                side_experimenter.DFSDP_nActions_obj.append(DFSDP_nActions)
            else:
                DFSDP_nActions = 5000
            all_methods_time_instance.append(DFSDP_planning_time)
            all_methods_success_instance.append(float(DFSDP_isSolved))
            all_methods_nActions_instance.append(DFSDP_nActions)

            #####################################################################
            reset_instance_success = utils2.resetInstance("Right_torso")
            #####################################################################

            ### (iv) DFSDP_nonlabeled
            start_time = time.time()
            unidir_dfsdp_planner = UnidirDFSDPPlanner(
                initial_arrangement, final_arrangement, side_experimenter.time_allowed,
                isLabeledRoadmapUsed=False)
            DFSDP_nonlabeled_planning_time = time.time() - start_time
            DFSDP_nonlabeled_isSolved = unidir_dfsdp_planner.isSolved
            DFSDP_nonlabeled_nActions = unidir_dfsdp_planner.best_solution_cost
            side_experimenter.DFSDP_nonlabeled_time_obj.append(DFSDP_nonlabeled_planning_time)
            side_experimenter.DFSDP_nonlabeled_success_obj.append(float(DFSDP_nonlabeled_isSolved))
            if DFSDP_nonlabeled_nActions != np.inf:
                side_experimenter.DFSDP_nonlabeled_nActions_obj.append(DFSDP_nonlabeled_nActions)
            else:
                DFSDP_nonlabeled_nActions = 5000
            all_methods_time_instance.append(DFSDP_nonlabeled_planning_time)
            all_methods_success_instance.append(float(DFSDP_nonlabeled_isSolved))
            all_methods_nActions_instance.append(DFSDP_nonlabeled_nActions)

            #####################################################################
            reset_instance_success = utils2.resetInstance("Right_torso")
            #####################################################################

            ### (v) mRS
            start_time = time.time()
            unidir_mrs_planner = UnidirMRSPlanner(
                initial_arrangement, final_arrangement, side_experimenter.time_allowed)
            mRS_planning_time = time.time() - start_time
            mRS_isSolved = unidir_mrs_planner.isSolved
            mRS_nActions = unidir_mrs_planner.best_solution_cost
            side_experimenter.mRS_time_obj.append(mRS_planning_time)
            side_experimenter.mRS_success_obj.append(float(mRS_isSolved))
            if mRS_nActions != np.inf:
                side_experimenter.mRS_nActions_obj.append(mRS_nActions)  
            else:
                mRS_nActions = 5000
            all_methods_time_instance.append(mRS_planning_time)
            all_methods_success_instance.append(float(mRS_isSolved))
            all_methods_nActions_instance.append(mRS_nActions)
            
            #####################################################################
            reset_instance_success = utils2.resetInstance("Right_torso")
            #####################################################################

            ### (vi) mRS_nonlabeled
            start_time = time.time()
            unidir_mrs_planner = UnidirMRSPlanner(
                initial_arrangement, final_arrangement, side_experimenter.time_allowed, 
                isLabeledRoadmapUsed=False)
            mRS_nonlabeled_planning_time = time.time() - start_time
            mRS_nonlabeled_isSolved = unidir_mrs_planner.isSolved
            mRS_nonlabeled_nActions = unidir_mrs_planner.best_solution_cost
            side_experimenter.mRS_nonlabeled_time_obj.append(mRS_nonlabeled_planning_time)
            side_experimenter.mRS_nonlabeled_success_obj.append(float(mRS_nonlabeled_isSolved))
            if mRS_nonlabeled_nActions != np.inf:
                side_experimenter.mRS_nonlabeled_nActions_obj.append(mRS_nonlabeled_nActions)
            else:
                mRS_nonlabeled_nActions = 5000
            all_methods_time_instance.append(mRS_nonlabeled_planning_time)
            all_methods_success_instance.append(float(mRS_nonlabeled_isSolved))
            all_methods_nActions_instance.append(mRS_nonlabeled_nActions)

            #####################################################################
            reset_instance_success = utils2.resetInstance("Right_torso")
            #####################################################################

            #############################################################################################
            ### this monotone instance has been tested on all methods
            num_monotoneInstancesSaved += 1
            tempInstanceFolder = os.path.join(side_experimenter.objectFolder, str(num_monotoneInstancesSaved))
            utils2.saveInstance(cylinder_objects, tempInstanceFolder)
            utils2.saveSolution(
                all_methods_time_instance, all_methods_success_instance, all_methods_nActions_instance, tempInstanceFolder)

            ### Before moving on to the next instance, clear the current instance
            clear_instance_success = utils2.clearInstance("Right_torso")
            # input("check the instance clearance!!!")

        ### reach here as all experiments have been finished for the #objects specified
        ### we need to save the avarage results for each method for the #objects specified
        side_experimenter.saveAverageSolutionPerNumObject()
        ### after that, move on to the next parameter for #objects
    
    ### reach here as you finish all experiments, congrats!

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    main(sys.argv)