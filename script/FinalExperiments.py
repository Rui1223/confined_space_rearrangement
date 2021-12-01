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
from UnidirCIRSMIXPlanner import UnidirCIRSMIXPlanner
from UnidirLazyCIRSMIXPlanner import UnidirLazyCIRSMIXPlanner
from UnidirLazyCIRSMIX2Planner import UnidirLazyCIRSMIX2Planner

############################### description ###########################################
### This class defines a FinalExperimenter class which
### conducts large-scale experiments with different methods
### with the number of experiments specified for each case (#objects)
### It
### (1) asks the execution scene to generate an instance
### (2) asks the pose estimator to get the object poses
### (3) reproduces the instance in task planner 
### (4) solves it with all methods
### (5) collect statistics (e.g., time/actions) from each method for comparison purpose
#######################################################################################

class FinalExperimenter(object):

    def __init__(self, args):
        rospack = rospkg.RosPack()
        self.rosPackagePath = rospack.get_path("confined_space_rearrangement")
        self.ExperimentsFolder = os.path.join(self.rosPackagePath, "final_experiments")
        if not os.path.exists(self.ExperimentsFolder):
            os.makedirs(self.ExperimentsFolder)
        self.numObjects_options = [11,12,13]
        self.numExperiments_perObject = int(args[1])
        self.maxInstancesNeed_perObject = int(args[2])
        self.time_allowed_monotone = 120
        self.time_allowed_nonMonotone = 240


    def createNumObjectsFolder(self, num_objects):
        ### create a folder denoted with specified num_objects
        self.objectFolder = os.path.join(self.ExperimentsFolder, str(num_objects))
        if not os.path.exists(self.objectFolder):
            os.makedirs(self.objectFolder)

        self.monotoneFolder = os.path.join(self.objectFolder, "monotone_instances")
        if not os.path.exists(self.monotoneFolder):
            os.makedirs(self.monotoneFolder)

        self.nonMonotoneFolder = os.path.join(self.objectFolder, "non_monotone_instances")
        if not os.path.exists(self.nonMonotoneFolder):
            os.makedirs(self.nonMonotoneFolder)

    def initializeObjLevelStat(self):
        ### initialize obj-level statistics variable
        self.Lazy_CIRSMIX_time_obj_m = []
        # self.CIRSMIX_time_obj_m = []
        self.CIRS_time_obj_m = []
        self.Lazy_CIRSMIX2_time_obj_m = []
        # self.DFSDP_time_obj_m = []
        # self.mRS_time_obj_m = []

        self.Lazy_CIRSMIX_motionTime_obj_m = []
        # self.CIRSMIX_motionTime_obj_m = []
        self.CIRS_motionTime_obj_m = []
        self.Lazy_CIRSMIX2_motionTime_obj_m = []
        # self.DFSDP_motionTime_obj_m = []
        # self.mRS_motionTime_obj_m = []

        self.Lazy_CIRSMIX_taskTime_obj_m = []
        # self.CIRSMIX_taskTime_obj_m = []
        self.CIRS_taskTime_obj_m = []
        self.Lazy_CIRSMIX2_taskTime_obj_m = []
        # self.DFSDP_taskTime_obj_m = []
        # self.mRS_taskTime_obj_m = []

        self.Lazy_CIRSMIX_success_obj_m = []
        # self.CIRSMIX_success_obj_m = []
        self.CIRS_success_obj_m = []
        self.Lazy_CIRSMIX2_success_obj_m = []
        # self.DFSDP_success_obj_m = []
        # self.mRS_success_obj_m = []

        self.Lazy_CIRSMIX_nActions_obj_m = []
        # self.CIRSMIX_nActions_obj_m = []
        self.CIRS_nActions_obj_m = []
        self.Lazy_CIRSMIX2_nActions_obj_m = []
        # self.DFSDP_nActions_obj_m = []
        # self.mRS_nActions_obj_m = []

        self.Lazy_CIRSMIX_time_obj_nm = []
        # self.CIRSMIX_time_obj_nm = []
        self.CIRS_time_obj_nm = []
        self.Lazy_CIRSMIX2_time_obj_nm = []
        # self.DFSDP_time_obj_nm = []
        # self.mRS_time_obj_nm = []

        self.Lazy_CIRSMIX_motionTime_obj_nm = []
        # self.CIRSMIX_motionTime_obj_nm = []
        self.CIRS_motionTime_obj_nm = []
        self.Lazy_CIRSMIX2_motionTime_obj_nm = []
        # self.DFSDP_motionTime_obj_nm = []
        # self.mRS_motionTime_obj_nm = []

        self.Lazy_CIRSMIX_taskTime_obj_nm = []
        # self.CIRSMIX_taskTime_obj_nm = []
        self.CIRS_taskTime_obj_nm = []
        self.Lazy_CIRSMIX2_taskTime_obj_nm = []
        # self.DFSDP_taskTime_obj_nm = []
        # self.mRS_taskTime_obj_nm = []

        self.Lazy_CIRSMIX_success_obj_nm = []
        # self.CIRSMIX_success_obj_nm = []
        self.CIRS_success_obj_nm = []
        self.Lazy_CIRSMIX2_success_obj_nm = []
        # self.DFSDP_success_obj_nm = []
        # self.mRS_success_obj_nm = []

        self.Lazy_CIRSMIX_nActions_obj_nm = []
        # self.CIRSMIX_nActions_obj_nm = []
        self.CIRS_nActions_obj_nm = []
        self.Lazy_CIRSMIX2_nActions_obj_nm = []
        # self.DFSDP_nActions_obj_nm = []
        # self.mRS_nActions_obj_nm = []

    def saveAverageSolutionPerNumObject(self):
        ######################################## monotone ########################################
        ################### average time ###################
        all_methods_average_time_obj_m = []

        if len(self.Lazy_CIRSMIX_time_obj_m) != 0:
            average_lazy_cirsmix_time_obj_m = sum(self.Lazy_CIRSMIX_time_obj_m) / len(self.Lazy_CIRSMIX_time_obj_m)
        else:
            average_lazy_cirsmix_time_obj_m = 10000
        all_methods_average_time_obj_m.append(average_lazy_cirsmix_time_obj_m)

        # if len(self.CIRSMIX_time_obj_m) != 0:
        #     average_cirsmix_time_obj_m = sum(self.CIRSMIX_time_obj_m) / len(self.CIRSMIX_time_obj_m)
        # else:
        #     average_cirsmix_time_obj_m = 10000
        # all_methods_average_time_obj_m.append(average_cirsmix_time_obj_m)

        if len(self.CIRS_time_obj_m) != 0:
            average_cirs_time_obj_m = sum(self.CIRS_time_obj_m) / len(self.CIRS_time_obj_m)
        else:
            average_cirs_time_obj_m = 10000
        all_methods_average_time_obj_m.append(average_cirs_time_obj_m)

        if len(self.Lazy_CIRSMIX2_time_obj_m) != 0:
            average_lazy_cirsmix2_time_obj_m = sum(self.Lazy_CIRSMIX2_time_obj_m) / len(self.Lazy_CIRSMIX2_time_obj_m)
        else:
            average_lazy_cirsmix2_time_obj_m = 10000
        all_methods_average_time_obj_m.append(average_lazy_cirsmix2_time_obj_m)

        # if len(self.DFSDP_time_obj_m) != 0:
        #     average_DFSDP_time_obj_m = sum(self.DFSDP_time_obj_m) / len(self.DFSDP_time_obj_m)
        # else:
        #     average_DFSDP_time_obj_m = 10000
        # all_methods_average_time_obj_m.append(average_DFSDP_time_obj_m)

        # if len(self.mRS_time_obj_m) != 0:
        #     average_mRS_time_obj_m = sum(self.mRS_time_obj_m) / len(self.mRS_time_obj_m)
        # else:
        #     average_mRS_time_obj_m = 10000
        # all_methods_average_time_obj_m.append(average_mRS_time_obj_m)

        ################### average motion planning time ###################
        all_methods_average_motionTime_obj_m = []

        if len(self.Lazy_CIRSMIX_motionTime_obj_m) != 0:
            average_lazy_cirsmix_motionTime_obj_m = sum(self.Lazy_CIRSMIX_motionTime_obj_m) / len(self.Lazy_CIRSMIX_motionTime_obj_m)
        else:
            average_lazy_cirsmix_motionTime_obj_m = 10000
        all_methods_average_motionTime_obj_m.append(average_lazy_cirsmix_motionTime_obj_m)

        # if len(self.CIRSMIX_motionTime_obj_m) != 0:
        #     average_cirsmix_motionTime_obj_m = sum(self.CIRSMIX_motionTime_obj_m) / len(self.CIRSMIX_motionTime_obj_m)
        # else:
        #     average_cirsmix_motionTime_obj_m = 10000
        # all_methods_average_motionTime_obj_m.append(average_cirsmix_motionTime_obj_m)

        if len(self.CIRS_motionTime_obj_m) != 0:
            average_cirs_motionTime_obj_m = sum(self.CIRS_motionTime_obj_m) / len(self.CIRS_motionTime_obj_m)
        else:
            average_cirs_motionTime_obj_m = 10000
        all_methods_average_motionTime_obj_m.append(average_cirs_motionTime_obj_m)

        if len(self.Lazy_CIRSMIX2_motionTime_obj_m) != 0:
            average_lazy_cirsmix2_motionTime_obj_m = sum(self.Lazy_CIRSMIX2_motionTime_obj_m) / len(self.Lazy_CIRSMIX2_motionTime_obj_m)
        else:
            average_lazy_cirsmix2_motionTime_obj_m = 10000
        all_methods_average_motionTime_obj_m.append(average_lazy_cirsmix2_motionTime_obj_m)

        # if len(self.DFSDP_motionTime_obj_m) != 0:
        #     average_DFSDP_motionTime_obj_m = sum(self.DFSDP_motionTime_obj_m) / len(self.DFSDP_motionTime_obj_m)
        # else:
        #     average_DFSDP_motionTime_obj_m = 10000
        # all_methods_average_motionTime_obj_m.append(average_DFSDP_motionTime_obj_m)

        # if len(self.mRS_motionTime_obj_m) != 0:
        #     average_mRS_motionTime_obj_m = sum(self.mRS_motionTime_obj_m) / len(self.mRS_motionTime_obj_m)
        # else:
        #     average_mRS_motionTime_obj_m = 10000
        # all_methods_average_motionTime_obj_m.append(average_mRS_motionTime_obj_m)

        ################### average task planning time ###################
        all_methods_average_taskTime_obj_m = []

        if len(self.Lazy_CIRSMIX_taskTime_obj_m) != 0:
            average_lazy_cirsmix_taskTime_obj_m = sum(self.Lazy_CIRSMIX_taskTime_obj_m) / len(self.Lazy_CIRSMIX_taskTime_obj_m)
        else:
            average_lazy_cirsmix_taskTime_obj_m = 10000
        all_methods_average_taskTime_obj_m.append(average_lazy_cirsmix_taskTime_obj_m)

        # if len(self.CIRSMIX_taskTime_obj_m) != 0:
        #     average_cirsmix_taskTime_obj_m = sum(self.CIRSMIX_taskTime_obj_m) / len(self.CIRSMIX_taskTime_obj_m)
        # else:
        #     average_cirsmix_taskTime_obj_m = 10000
        # all_methods_average_taskTime_obj_m.append(average_cirsmix_taskTime_obj_m)

        if len(self.CIRS_taskTime_obj_m) != 0:
            average_cirs_taskTime_obj_m = sum(self.CIRS_taskTime_obj_m) / len(self.CIRS_taskTime_obj_m)
        else:
            average_cirs_taskTime_obj_m = 10000
        all_methods_average_taskTime_obj_m.append(average_cirs_taskTime_obj_m)

        if len(self.Lazy_CIRSMIX2_taskTime_obj_m) != 0:
            average_lazy_cirsmix2_taskTime_obj_m = sum(self.Lazy_CIRSMIX2_taskTime_obj_m) / len(self.Lazy_CIRSMIX2_taskTime_obj_m)
        else:
            average_lazy_cirsmix2_taskTime_obj_m = 10000
        all_methods_average_taskTime_obj_m.append(average_lazy_cirsmix2_taskTime_obj_m)

        # if len(self.DFSDP_taskTime_obj_m) != 0:
        #     average_DFSDP_taskTime_obj_m = sum(self.DFSDP_taskTime_obj_m) / len(self.DFSDP_taskTime_obj_m)
        # else:
        #     average_DFSDP_taskTime_obj_m = 10000
        # all_methods_average_taskTime_obj_m.append(average_DFSDP_taskTime_obj_m)

        # if len(self.mRS_taskTime_obj_m) != 0:
        #     average_mRS_taskTime_obj_m = sum(self.mRS_taskTime_obj_m) / len(self.mRS_taskTime_obj_m)
        # else:
        #     average_mRS_taskTime_obj_m = 10000
        # all_methods_average_taskTime_obj_m.append(average_mRS_taskTime_obj_m)

        ################### average success ###################
        all_methods_average_success_obj_m = []

        if len(self.Lazy_CIRSMIX_success_obj_m) != 0:
            average_lazy_cirsmix_success_obj_m = sum(self.Lazy_CIRSMIX_success_obj_m) / len(self.Lazy_CIRSMIX_success_obj_m)
        else:
            average_lazy_cirsmix_success_obj_m = 10000
        all_methods_average_success_obj_m.append(average_lazy_cirsmix_success_obj_m)

        # if len(self.CIRSMIX_success_obj_m) != 0:
        #     average_cirsmix_success_obj_m = sum(self.CIRSMIX_success_obj_m) / len(self.CIRSMIX_success_obj_m)
        # else:
        #     average_cirsmix_success_obj_m = 10000
        # all_methods_average_success_obj_m.append(average_cirsmix_success_obj_m)

        if len(self.CIRS_success_obj_m) != 0:
            average_cirs_success_obj_m = sum(self.CIRS_success_obj_m) / len(self.CIRS_success_obj_m)
        else:
            average_cirs_success_obj_m = 0.0
        all_methods_average_success_obj_m.append(average_cirs_success_obj_m)

        if len(self.Lazy_CIRSMIX2_success_obj_m) != 0:
            average_lazy_cirsmix2_success_obj_m = sum(self.Lazy_CIRSMIX2_success_obj_m) / len(self.Lazy_CIRSMIX2_success_obj_m)
        else:
            average_lazy_cirsmix2_success_obj_m = 10000
        all_methods_average_success_obj_m.append(average_lazy_cirsmix2_success_obj_m)

        # if len(self.DFSDP_success_obj_m) != 0:
        #     average_DFSDP_success_obj_m = sum(self.DFSDP_success_obj_m) / len(self.DFSDP_success_obj_m)
        # else:
        #     average_DFSDP_success_obj_m = 0.0
        # all_methods_average_success_obj_m.append(average_DFSDP_success_obj_m)

        # if len(self.mRS_success_obj_m) != 0:
        #     average_mRS_success_obj_m = sum(self.mRS_success_obj_m) / len(self.mRS_success_obj_m)
        # else:
        #     average_mRS_success_obj_m = 0.0
        # all_methods_average_success_obj_m.append(average_mRS_success_obj_m)

        ################### average nActions ###################
        all_methods_average_nActions_obj_m = []

        if len(self.Lazy_CIRSMIX_nActions_obj_m) != 0:
            average_lazy_cirsmix_nActions_obj_m = sum(self.Lazy_CIRSMIX_nActions_obj_m) / len(self.Lazy_CIRSMIX_nActions_obj_m)
        else:
            average_lazy_cirsmix_nActions_obj_m = 10000
        all_methods_average_nActions_obj_m.append(average_lazy_cirsmix_nActions_obj_m)

        # if len(self.CIRSMIX_nActions_obj_m) != 0:
        #     average_cirsmix_nActions_obj_m = sum(self.CIRSMIX_nActions_obj_m) / len(self.CIRSMIX_nActions_obj_m)
        # else:
        #     average_cirsmix_nActions_obj_m = 10000
        # all_methods_average_nActions_obj_m.append(average_cirsmix_nActions_obj_m)

        if len(self.CIRS_nActions_obj_m) != 0:
            average_cirs_nActions_obj_m = sum(self.CIRS_nActions_obj_m) / len(self.CIRS_nActions_obj_m)
        else:
            average_cirs_nActions_obj_m = 0.0
        all_methods_average_nActions_obj_m.append(average_cirs_nActions_obj_m)

        if len(self.Lazy_CIRSMIX2_nActions_obj_m) != 0:
            average_lazy_cirsmix2_nActions_obj_m = sum(self.Lazy_CIRSMIX2_nActions_obj_m) / len(self.Lazy_CIRSMIX2_nActions_obj_m)
        else:
            average_lazy_cirsmix2_nActions_obj_m = 10000
        all_methods_average_nActions_obj_m.append(average_lazy_cirsmix2_nActions_obj_m)

        # if len(self.DFSDP_nActions_obj_m) != 0:
        #     average_DFSDP_nActions_obj_m = sum(self.DFSDP_nActions_obj_m) / len(self.DFSDP_nActions_obj_m)
        # else:
        #     average_DFSDP_nActions_obj_m = 0.0
        # all_methods_average_nActions_obj_m.append(average_DFSDP_nActions_obj_m)

        # if len(self.mRS_nActions_obj_m) != 0:
        #     average_mRS_nActions_obj_m = sum(self.mRS_nActions_obj_m) / len(self.mRS_nActions_obj_m)
        # else:
        #     average_mRS_nActions_obj_m = 0.0
        # all_methods_average_nActions_obj_m.append(average_mRS_nActions_obj_m)

        ### save average results
        utils2.saveSolution2(
            all_methods_average_time_obj_m, all_methods_average_motionTime_obj_m, all_methods_average_taskTime_obj_m, \
            all_methods_average_success_obj_m, all_methods_average_nActions_obj_m, self.monotoneFolder)
        ##########################################################################################

        ######################################## non-monotone ########################################
        ################### average time ###################
        all_methods_average_time_obj_nm = []

        if len(self.Lazy_CIRSMIX_time_obj_nm) != 0:
            average_lazy_cirsmix_time_obj_nm = sum(self.Lazy_CIRSMIX_time_obj_nm) / len(self.Lazy_CIRSMIX_time_obj_nm)
        else:
            average_lazy_cirsmix_time_obj_nm = 10000
        all_methods_average_time_obj_nm.append(average_lazy_cirsmix_time_obj_nm)

        # if len(self.CIRSMIX_time_obj_nm) != 0:
        #     average_cirsmix_time_obj_nm = sum(self.CIRSMIX_time_obj_nm) / len(self.CIRSMIX_time_obj_nm)
        # else:
        #     average_cirsmix_time_obj_nm = 10000
        # all_methods_average_time_obj_nm.append(average_cirsmix_time_obj_nm)

        if len(self.CIRS_time_obj_nm) != 0:
            average_cirs_time_obj_nm = sum(self.CIRS_time_obj_nm) / len(self.CIRS_time_obj_nm)
        else:
            average_cirs_time_obj_nm = 10000
        all_methods_average_time_obj_nm.append(average_cirs_time_obj_nm)

        if len(self.Lazy_CIRSMIX2_time_obj_nm) != 0:
            average_lazy_cirsmix2_time_obj_nm = sum(self.Lazy_CIRSMIX2_time_obj_nm) / len(self.Lazy_CIRSMIX2_time_obj_nm)
        else:
            average_lazy_cirsmix2_time_obj_nm = 10000
        all_methods_average_time_obj_nm.append(average_lazy_cirsmix2_time_obj_nm)

        # if len(self.DFSDP_time_obj_nm) != 0:
        #     average_DFSDP_time_obj_nm = sum(self.DFSDP_time_obj_nm) / len(self.DFSDP_time_obj_nm)
        # else:
        #     average_DFSDP_time_obj_nm = 10000
        # all_methods_average_time_obj_nm.append(average_DFSDP_time_obj_nm)

        # if len(self.mRS_time_obj_nm) != 0:
        #     average_mRS_time_obj_nm = sum(self.mRS_time_obj_nm) / len(self.mRS_time_obj_nm)
        # else:
        #     average_mRS_time_obj_nm = 10000
        # all_methods_average_time_obj_nm.append(average_mRS_time_obj_nm)

        ################### average motion planning time ###################
        all_methods_average_motionTime_obj_nm = []

        if len(self.Lazy_CIRSMIX_motionTime_obj_nm) != 0:
            average_lazy_cirsmix_motionTime_obj_nm = sum(self.Lazy_CIRSMIX_motionTime_obj_nm) / len(self.Lazy_CIRSMIX_motionTime_obj_nm)
        else:
            average_lazy_cirsmix_motionTime_obj_nm = 10000
        all_methods_average_motionTime_obj_nm.append(average_lazy_cirsmix_motionTime_obj_nm)

        # if len(self.CIRSMIX_motionTime_obj_nm) != 0:
        #     average_cirsmix_motionTime_obj_nm = sum(self.CIRSMIX_motionTime_obj_nm) / len(self.CIRSMIX_motionTime_obj_nm)
        # else:
        #     average_cirsmix_motionTime_obj_nm = 10000
        # all_methods_average_motionTime_obj_nm.append(average_cirsmix_motionTime_obj_nm)

        if len(self.CIRS_motionTime_obj_nm) != 0:
            average_cirs_motionTime_obj_nm = sum(self.CIRS_motionTime_obj_nm) / len(self.CIRS_motionTime_obj_nm)
        else:
            average_cirs_motionTime_obj_nm = 10000
        all_methods_average_motionTime_obj_nm.append(average_cirs_motionTime_obj_nm)

        if len(self.Lazy_CIRSMIX2_motionTime_obj_nm) != 0:
            average_lazy_cirsmix2_motionTime_obj_nm = sum(self.Lazy_CIRSMIX2_motionTime_obj_nm) / len(self.Lazy_CIRSMIX2_motionTime_obj_nm)
        else:
            average_lazy_cirsmix2_motionTime_obj_nm = 10000
        all_methods_average_motionTime_obj_nm.append(average_lazy_cirsmix2_motionTime_obj_nm)

        # if len(self.DFSDP_motionTime_obj_nm) != 0:
        #     average_DFSDP_motionTime_obj_nm = sum(self.DFSDP_motionTime_obj_nm) / len(self.DFSDP_motionTime_obj_nm)
        # else:
        #     average_DFSDP_motionTime_obj_nm = 10000
        # all_methods_average_motionTime_obj_nm.append(average_DFSDP_motionTime_obj_nm)

        # if len(self.mRS_motionTime_obj_nm) != 0:
        #     average_mRS_motionTime_obj_nm = sum(self.mRS_motionTime_obj_nm) / len(self.mRS_motionTime_obj_nm)
        # else:
        #     average_mRS_motionTime_obj_nm = 10000
        # all_methods_average_motionTime_obj_nm.append(average_mRS_motionTime_obj_nm)

        ################### average task planning time ###################
        all_methods_average_taskTime_obj_nm = []

        if len(self.Lazy_CIRSMIX_taskTime_obj_nm) != 0:
            average_lazy_cirsmix_taskTime_obj_nm = sum(self.Lazy_CIRSMIX_taskTime_obj_nm) / len(self.Lazy_CIRSMIX_taskTime_obj_nm)
        else:
            average_lazy_cirsmix_taskTime_obj_nm = 10000
        all_methods_average_taskTime_obj_nm.append(average_lazy_cirsmix_taskTime_obj_nm)

        # if len(self.CIRSMIX_taskTime_obj_nm) != 0:
        #     average_cirsmix_taskTime_obj_nm = sum(self.CIRSMIX_taskTime_obj_nm) / len(self.CIRSMIX_taskTime_obj_nm)
        # else:
        #     average_cirsmix_taskTime_obj_nm = 10000
        # all_methods_average_taskTime_obj_nm.append(average_cirsmix_taskTime_obj_nm)

        if len(self.CIRS_taskTime_obj_nm) != 0:
            average_cirs_taskTime_obj_nm = sum(self.CIRS_taskTime_obj_nm) / len(self.CIRS_taskTime_obj_nm)
        else:
            average_cirs_taskTime_obj_nm = 10000
        all_methods_average_taskTime_obj_nm.append(average_cirs_taskTime_obj_nm)

        if len(self.Lazy_CIRSMIX2_taskTime_obj_nm) != 0:
            average_lazy_cirsmix2_taskTime_obj_nm = sum(self.Lazy_CIRSMIX2_taskTime_obj_nm) / len(self.Lazy_CIRSMIX2_taskTime_obj_nm)
        else:
            average_lazy_cirsmix2_taskTime_obj_nm = 10000
        all_methods_average_taskTime_obj_nm.append(average_lazy_cirsmix2_taskTime_obj_nm)

        # if len(self.DFSDP_taskTime_obj_nm) != 0:
        #     average_DFSDP_taskTime_obj_nm = sum(self.DFSDP_taskTime_obj_nm) / len(self.DFSDP_taskTime_obj_nm)
        # else:
        #     average_DFSDP_taskTime_obj_nm = 10000
        # all_methods_average_taskTime_obj_nm.append(average_DFSDP_taskTime_obj_nm)

        # if len(self.mRS_taskTime_obj_nm) != 0:
        #     average_mRS_taskTime_obj_nm = sum(self.mRS_taskTime_obj_nm) / len(self.mRS_taskTime_obj_nm)
        # else:
        #     average_mRS_taskTime_obj_nm = 10000
        # all_methods_average_taskTime_obj_nm.append(average_mRS_taskTime_obj_nm)

        ################### average success ###################
        all_methods_average_success_obj_nm = []

        if len(self.Lazy_CIRSMIX_success_obj_nm) != 0:
            average_lazy_cirsmix_success_obj_nm = sum(self.Lazy_CIRSMIX_success_obj_nm) / len(self.Lazy_CIRSMIX_success_obj_nm)
        else:
            average_lazy_cirsmix_success_obj_nm = 10000
        all_methods_average_success_obj_nm.append(average_lazy_cirsmix_success_obj_nm)

        # if len(self.CIRSMIX_success_obj_nm) != 0:
        #     average_cirsmix_success_obj_nm = sum(self.CIRSMIX_success_obj_nm) / len(self.CIRSMIX_success_obj_nm)
        # else:
        #     average_cirsmix_success_obj_nm = 10000
        # all_methods_average_success_obj_nm.append(average_cirsmix_success_obj_nm)

        if len(self.CIRS_success_obj_nm) != 0:
            average_cirs_success_obj_nm = sum(self.CIRS_success_obj_nm) / len(self.CIRS_success_obj_nm)
        else:
            average_cirs_success_obj_nm = 0.0
        all_methods_average_success_obj_nm.append(average_cirs_success_obj_nm)

        if len(self.Lazy_CIRSMIX2_success_obj_nm) != 0:
            average_lazy_cirsmix2_success_obj_nm = sum(self.Lazy_CIRSMIX2_success_obj_nm) / len(self.Lazy_CIRSMIX2_success_obj_nm)
        else:
            average_lazy_cirsmix2_success_obj_nm = 10000
        all_methods_average_success_obj_nm.append(average_lazy_cirsmix2_success_obj_nm)

        # if len(self.DFSDP_success_obj_nm) != 0:
        #     average_DFSDP_success_obj_nm = sum(self.DFSDP_success_obj_nm) / len(self.DFSDP_success_obj_nm)
        # else:
        #     average_DFSDP_success_obj_nm = 0.0
        # all_methods_average_success_obj_nm.append(average_DFSDP_success_obj_nm)

        # if len(self.mRS_success_obj_nm) != 0:
        #     average_mRS_success_obj_nm = sum(self.mRS_success_obj_nm) / len(self.mRS_success_obj_nm)
        # else:
        #     average_mRS_success_obj_nm = 0.0
        # all_methods_average_success_obj_nm.append(average_mRS_success_obj_nm)

        ################### average nActions ###################
        all_methods_average_nActions_obj_nm = []

        if len(self.Lazy_CIRSMIX_nActions_obj_nm) != 0:
            average_lazy_cirsmix_nActions_obj_nm = sum(self.Lazy_CIRSMIX_nActions_obj_nm) / len(self.Lazy_CIRSMIX_nActions_obj_nm)
        else:
            average_lazy_cirsmix_nActions_obj_nm = 10000
        all_methods_average_nActions_obj_nm.append(average_lazy_cirsmix_nActions_obj_nm)

        # if len(self.CIRSMIX_nActions_obj_nm) != 0:
        #     average_cirsmix_nActions_obj_nm = sum(self.CIRSMIX_nActions_obj_nm) / len(self.CIRSMIX_nActions_obj_nm)
        # else:
        #     average_cirsmix_nActions_obj_nm = 10000
        # all_methods_average_nActions_obj_nm.append(average_cirsmix_nActions_obj_nm)

        if len(self.CIRS_nActions_obj_nm) != 0:
            average_cirs_nActions_obj_nm = sum(self.CIRS_nActions_obj_nm) / len(self.CIRS_nActions_obj_nm)
        else:
            average_cirs_nActions_obj_nm = 0.0
        all_methods_average_nActions_obj_nm.append(average_cirs_nActions_obj_nm)

        if len(self.Lazy_CIRSMIX2_nActions_obj_nm) != 0:
            average_lazy_cirsmix2_nActions_obj_nm = sum(self.Lazy_CIRSMIX2_nActions_obj_nm) / len(self.Lazy_CIRSMIX2_nActions_obj_nm)
        else:
            average_lazy_cirsmix2_nActions_obj_nm = 10000
        all_methods_average_nActions_obj_nm.append(average_lazy_cirsmix2_nActions_obj_nm)

        # if len(self.DFSDP_nActions_obj_nm) != 0:
        #     average_DFSDP_nActions_obj_nm = sum(self.DFSDP_nActions_obj_nm) / len(self.DFSDP_nActions_obj_nm)
        # else:
        #     average_DFSDP_nActions_obj_nm = 0.0
        # all_methods_average_nActions_obj_nm.append(average_DFSDP_nActions_obj_nm)

        # if len(self.mRS_nActions_obj_nm) != 0:
        #     average_mRS_nActions_obj_nm = sum(self.mRS_nActions_obj_nm) / len(self.mRS_nActions_obj_nm)
        # else:
        #     average_mRS_nActions_obj_nm = 0.0
        # all_methods_average_nActions_obj_nm.append(average_mRS_nActions_obj_nm)

        ### save average results
        utils2.saveSolution2(
            all_methods_average_time_obj_nm, all_methods_average_motionTime_obj_nm, all_methods_average_taskTime_obj_nm, \
            all_methods_average_success_obj_nm, all_methods_average_nActions_obj_nm, self.nonMonotoneFolder)
        ######################################################################################################


    def rosInit(self):
        ### This function specifies the role of a node instance for this class ###
        ### and initializes a ros node
        rospy.init_node("final_experiments", anonymous=True)


def main(args):
    final_experimenter = FinalExperimenter(args)
    final_experimenter.rosInit()
    rate = rospy.Rate(10) ### 10hz

    for num_objects in final_experimenter.numObjects_options:
        ### create a folder for current num_objects
        final_experimenter.createNumObjectsFolder(num_objects)
        final_experimenter.initializeObjLevelStat()
        num_monotoneInstancesSaved = 0
        num_nonMonotoneInstancesSaved = 0
        totalNum_instancesSaved = num_monotoneInstancesSaved + num_nonMonotoneInstancesSaved

        for experiment_id in range(1, final_experimenter.numExperiments_perObject+1):
            all_methods_time_instance = []
            all_methods_success_instance = []
            all_methods_nActions_instance = []
            all_methods_motionTime_instance = []
            all_methods_taskTime_instance = []
            ### first see if we already have enough instances
            if (totalNum_instancesSaved >= final_experimenter.maxInstancesNeed_perObject): break
            ### generate an instance in the execution scene
            initialize_instance_success = utils2.serviceCall_generateInstanceCylinder(
                                                num_objects, totalNum_instancesSaved+1, True)
            if not initialize_instance_success: continue
            ### object pose estimation
            cylinder_objects = utils2.serviceCall_cylinderPositionEstimate()
            ### reproduce the estimated object poses in the planning scene
            initial_arrangement, final_arrangement, reproduce_instance_success = \
                    utils2.serviceCall_reproduceInstanceCylinder(cylinder_objects)
            ### generate IK config for start positions for all objects
            ik_generate_success = utils2.serviceCall_generateConfigsForStartPositions("Right_torso")

            ########################## now using different methods to solve the instance ##########################
            ### use Lazy_CIRSMIX first to classify
            ### (1) monotone problems
            ### (2) non-monotone problems

            ### (0) Lazy CIRSMIX
            start_time = time.time()
            unidir_lazy_cirsmix_planner = UnidirLazyCIRSMIXPlanner(
                initial_arrangement, final_arrangement, final_experimenter.time_allowed_nonMonotone)
            lazy_cirsmix_planning_time = time.time() - start_time
            lazy_cirsmix_motion_planning_time = unidir_lazy_cirsmix_planner.motion_planning_time
            lazy_cirsmix_task_planning_time = lazy_cirsmix_planning_time - lazy_cirsmix_motion_planning_time
            lazy_cirsmix_isSolved = unidir_lazy_cirsmix_planner.isSolved
            lazy_cirsmix_nActions = unidir_lazy_cirsmix_planner.best_solution_cost
            if lazy_cirsmix_isSolved:
                ### the problem is solved by lazy CIRSMIX in given time
                if lazy_cirsmix_nActions == num_objects:
                    ### it is a monotone instance, set the time allowed for other 
                    time_allowed_for_other_methods = final_experimenter.time_allowed_monotone
                    if lazy_cirsmix_planning_time <= final_experimenter.time_allowed_monotone:
                        ### Lazy CIRSMIX solves it successfully
                        final_experimenter.Lazy_CIRSMIX_time_obj_m.append(lazy_cirsmix_planning_time)
                        final_experimenter.Lazy_CIRSMIX_motionTime_obj_m.append(lazy_cirsmix_motion_planning_time)
                        final_experimenter.Lazy_CIRSMIX_taskTime_obj_m.append(lazy_cirsmix_task_planning_time)
                        final_experimenter.Lazy_CIRSMIX_success_obj_m.append(float(lazy_cirsmix_isSolved))
                        final_experimenter.Lazy_CIRSMIX_nActions_obj_m.append(lazy_cirsmix_nActions)
                    else:
                        ### Lazy CIRSMIX does not solve it successfully (time limit exceeds)
                        final_experimenter.Lazy_CIRSMIX_time_obj_m.append(lazy_cirsmix_planning_time)
                        final_experimenter.Lazy_CIRSMIX_motionTime_obj_m.append(lazy_cirsmix_motion_planning_time)
                        final_experimenter.Lazy_CIRSMIX_taskTime_obj_m.append(lazy_cirsmix_task_planning_time)
                        lazy_cirsmix_isSolved = False
                        final_experimenter.Lazy_CIRSMIX_success_obj_m.append(float(lazy_cirsmix_isSolved))
                        final_experimenter.Lazy_CIRSMIX_nActions_obj_m.append(lazy_cirsmix_nActions)
                else:
                    ### it is a non-monotone instance, set the time allowed for other methods
                    time_allowed_for_other_methods = final_experimenter.time_allowed_nonMonotone
                    ### Lazy CIRSMIX solves it successfully
                    final_experimenter.Lazy_CIRSMIX_time_obj_nm.append(lazy_cirsmix_planning_time)
                    final_experimenter.Lazy_CIRSMIX_motionTime_obj_nm.append(lazy_cirsmix_motion_planning_time)
                    final_experimenter.Lazy_CIRSMIX_taskTime_obj_nm.append(lazy_cirsmix_task_planning_time)
                    final_experimenter.Lazy_CIRSMIX_success_obj_nm.append(float(lazy_cirsmix_isSolved))
                    final_experimenter.Lazy_CIRSMIX_nActions_obj_nm.append(lazy_cirsmix_nActions)
            else:
                ### the problem is not solved by Lazy CIRSMIX in given time
                ### the problem is deemed as non-monotone
                time_allowed_for_other_methods = final_experimenter.time_allowed_nonMonotone
                final_experimenter.Lazy_CIRSMIX_time_obj_nm.append(lazy_cirsmix_planning_time)
                final_experimenter.Lazy_CIRSMIX_motionTime_obj_nm.append(lazy_cirsmix_motion_planning_time)
                final_experimenter.Lazy_CIRSMIX_taskTime_obj_nm.append(lazy_cirsmix_task_planning_time)
                final_experimenter.Lazy_CIRSMIX_success_obj_nm.append(float(lazy_cirsmix_isSolved))
                lazy_cirsmix_nActions = 5000 ### set np.inf to 5000

            all_methods_time_instance.append(lazy_cirsmix_planning_time)
            all_methods_motionTime_instance.append(lazy_cirsmix_motion_planning_time)
            all_methods_taskTime_instance.append(lazy_cirsmix_task_planning_time)
            all_methods_success_instance.append(float(lazy_cirsmix_isSolved))
            all_methods_nActions_instance.append(lazy_cirsmix_nActions)

            #####################################################################
            reset_instance_success = utils2.resetInstance("Right_torso")
            #####################################################################

            ###### try other methods now ######
            # ### (1) CIRSMIX
            # start_time = time.time()
            # unidir_cirsmix_planner = UnidirCIRSMIXPlanner(
            #     initial_arrangement, final_arrangement, time_allowed_for_other_methods)
            # cirsmix_planning_time = time.time() - start_time
            # cirsmix_motion_planning_time = unidir_cirsmix_planner.motion_planning_time
            # cirsmix_task_planning_time = cirsmix_planning_time - cirsmix_motion_planning_time
            # cirsmix_isSolved = unidir_cirsmix_planner.isSolved
            # cirsmix_nActions = unidir_cirsmix_planner.best_solution_cost
            # if time_allowed_for_other_methods == final_experimenter.time_allowed_monotone:
            #     final_experimenter.CIRSMIX_time_obj_m.append(cirsmix_planning_time)
            #     final_experimenter.CIRSMIX_motionTime_obj_m.append(cirsmix_motion_planning_time)
            #     final_experimenter.CIRSMIX_taskTime_obj_m.append(cirsmix_task_planning_time)
            #     final_experimenter.CIRSMIX_success_obj_m.append(cirsmix_isSolved)
            #     if cirsmix_nActions != np.inf:
            #         final_experimenter.CIRSMIX_nActions_obj_m.append(cirsmix_nActions)
            #     else:
            #         cirsmix_nActions = 5000
            # else:
            #     ### non-monotone problem solution
            #     final_experimenter.CIRSMIX_time_obj_nm.append(cirsmix_planning_time)
            #     final_experimenter.CIRSMIX_motionTime_obj_nm.append(cirsmix_motion_planning_time)
            #     final_experimenter.CIRSMIX_taskTime_obj_nm.append(cirsmix_task_planning_time)
            #     final_experimenter.CIRSMIX_success_obj_nm.append(cirsmix_isSolved)
            #     if cirsmix_nActions != np.inf:
            #         final_experimenter.CIRSMIX_nActions_obj_nm.append(cirsmix_nActions)
            #     else:
            #         cirsmix_nActions = 5000
            # all_methods_time_instance.append(cirsmix_planning_time)
            # all_methods_motionTime_instance.append(cirsmix_motion_planning_time)
            # all_methods_taskTime_instance.append(cirsmix_task_planning_time)
            # all_methods_success_instance.append(float(cirsmix_isSolved))
            # all_methods_nActions_instance.append(cirsmix_nActions)

            # #####################################################################
            # reset_instance_success = utils2.resetInstance("Right_torso")
            # #####################################################################                

            ### (2) CIRS
            start_time = time.time()
            unidir_cirs_planner = UnidirCIRSPlanner(
                initial_arrangement, final_arrangement, time_allowed_for_other_methods)
            cirs_planning_time = time.time() - start_time
            cirs_motion_planning_time = unidir_cirs_planner.motion_planning_time
            cirs_task_planning_time = cirs_planning_time - cirs_motion_planning_time
            cirs_isSolved = unidir_cirs_planner.isSolved
            cirs_nActions = unidir_cirs_planner.best_solution_cost
            if time_allowed_for_other_methods == final_experimenter.time_allowed_monotone:
                final_experimenter.CIRS_time_obj_m.append(cirs_planning_time)
                final_experimenter.CIRS_motionTime_obj_m.append(cirs_motion_planning_time)
                final_experimenter.CIRS_taskTime_obj_m.append(cirs_task_planning_time)
                final_experimenter.CIRS_success_obj_m.append(cirs_isSolved)
                if cirs_nActions != np.inf:
                    final_experimenter.CIRS_nActions_obj_m.append(cirs_nActions)
                else:
                    cirs_nActions = 5000
            else:
                ### non-monotone problem solution
                final_experimenter.CIRS_time_obj_nm.append(cirs_planning_time)
                final_experimenter.CIRS_motionTime_obj_nm.append(cirs_motion_planning_time)
                final_experimenter.CIRS_taskTime_obj_nm.append(cirs_task_planning_time)
                final_experimenter.CIRS_success_obj_nm.append(cirs_isSolved)
                if cirs_nActions != np.inf:
                    final_experimenter.CIRS_nActions_obj_nm.append(cirs_nActions)
                else:
                    cirs_nActions = 5000
            all_methods_time_instance.append(cirs_planning_time)
            all_methods_motionTime_instance.append(cirs_motion_planning_time)
            all_methods_taskTime_instance.append(cirs_task_planning_time)
            all_methods_success_instance.append(float(cirs_isSolved))
            all_methods_nActions_instance.append(cirs_nActions)

            #####################################################################
            reset_instance_success = utils2.resetInstance("Right_torso")
            #####################################################################

            ### (3) Lazy CIRSMIX2
            start_time = time.time()
            unidir_lazy_cirsmix2_planner = UnidirLazyCIRSMIX2Planner(
                initial_arrangement, final_arrangement, time_allowed_for_other_methods)
            lazy_cirsmix2_planning_time = time.time() - start_time
            lazy_cirsmix2_motion_planning_time = unidir_lazy_cirsmix2_planner.motion_planning_time
            lazy_cirsmix2_task_planning_time = lazy_cirsmix2_planning_time - lazy_cirsmix2_motion_planning_time
            lazy_cirsmix2_isSolved = unidir_lazy_cirsmix2_planner.isSolved
            lazy_cirsmix2_nActions = unidir_lazy_cirsmix2_planner.best_solution_cost
            if time_allowed_for_other_methods == final_experimenter.time_allowed_monotone:
                final_experimenter.Lazy_CIRSMIX2_time_obj_m.append(lazy_cirsmix2_planning_time)
                final_experimenter.Lazy_CIRSMIX2_motionTime_obj_m.append(lazy_cirsmix2_motion_planning_time)
                final_experimenter.Lazy_CIRSMIX2_taskTime_obj_m.append(lazy_cirsmix2_task_planning_time)
                final_experimenter.Lazy_CIRSMIX2_success_obj_m.append(lazy_cirsmix2_isSolved)
                if lazy_cirsmix2_nActions != np.inf:
                    final_experimenter.Lazy_CIRSMIX2_nActions_obj_m.append(lazy_cirsmix2_nActions)
                else:
                    lazy_cirsmix2_nActions = 5000
            else:
                ### non-monotone problem solution
                final_experimenter.Lazy_CIRSMIX2_time_obj_nm.append(lazy_cirsmix2_planning_time)
                final_experimenter.Lazy_CIRSMIX2_motionTime_obj_nm.append(lazy_cirsmix2_motion_planning_time)
                final_experimenter.Lazy_CIRSMIX2_taskTime_obj_nm.append(lazy_cirsmix2_task_planning_time)
                final_experimenter.Lazy_CIRSMIX2_success_obj_nm.append(lazy_cirsmix2_isSolved)
                if lazy_cirsmix2_nActions != np.inf:
                    final_experimenter.Lazy_CIRSMIX2_nActions_obj_nm.append(lazy_cirsmix2_nActions)
                else:
                    lazy_cirsmix2_nActions = 5000
            all_methods_time_instance.append(lazy_cirsmix2_planning_time)
            all_methods_motionTime_instance.append(lazy_cirsmix2_motion_planning_time)
            all_methods_taskTime_instance.append(lazy_cirsmix2_task_planning_time)
            all_methods_success_instance.append(float(lazy_cirsmix2_isSolved))
            all_methods_nActions_instance.append(lazy_cirsmix2_nActions)

            #####################################################################
            reset_instance_success = utils2.resetInstance("Right_torso")
            #####################################################################

            # ### (4) DFSDP
            # start_time = time.time()
            # unidir_dfsdp_planner = UnidirDFSDPPlanner(
            #     initial_arrangement, final_arrangement, time_allowed_for_other_methods)
            # dfsdp_planning_time = time.time() - start_time
            # dfsdp_motion_planning_time = unidir_dfsdp_planner.motion_planning_time
            # dfsdp_task_planning_time = dfsdp_planning_time - dfsdp_motion_planning_time
            # dfsdp_isSolved = unidir_dfsdp_planner.isSolved
            # dfsdp_nActions = unidir_dfsdp_planner.best_solution_cost
            # if time_allowed_for_other_methods == final_experimenter.time_allowed_monotone:
            #     final_experimenter.DFSDP_time_obj_m.append(dfsdp_planning_time)
            #     final_experimenter.DFSDP_motionTime_obj_m.append(dfsdp_motion_planning_time)
            #     final_experimenter.DFSDP_taskTime_obj_m.append(dfsdp_task_planning_time)
            #     final_experimenter.DFSDP_success_obj_m.append(dfsdp_isSolved)
            #     if dfsdp_nActions != np.inf:
            #         final_experimenter.DFSDP_nActions_obj_m.append(dfsdp_nActions)
            #     else:
            #         dfsdp_nActions = 5000
            # else:
            #     ### non-monotone problem solution
            #     final_experimenter.DFSDP_time_obj_nm.append(dfsdp_planning_time)
            #     final_experimenter.DFSDP_motionTime_obj_nm.append(dfsdp_motion_planning_time)
            #     final_experimenter.DFSDP_taskTime_obj_nm.append(dfsdp_task_planning_time)
            #     final_experimenter.DFSDP_success_obj_nm.append(dfsdp_isSolved)
            #     if dfsdp_nActions != np.inf:
            #         final_experimenter.DFSDP_nActions_obj_nm.append(dfsdp_nActions)
            #     else:
            #         dfsdp_nActions = 5000
            # all_methods_time_instance.append(dfsdp_planning_time)
            # all_methods_motionTime_instance.append(dfsdp_motion_planning_time)
            # all_methods_taskTime_instance.append(dfsdp_task_planning_time)
            # all_methods_success_instance.append(float(dfsdp_isSolved))
            # all_methods_nActions_instance.append(dfsdp_nActions)

            # #####################################################################
            # reset_instance_success = utils2.resetInstance("Right_torso")
            # #####################################################################

            # ### (5) mRS
            # start_time = time.time()
            # unidir_mrs_planner = UnidirMRSPlanner(
            #     initial_arrangement, final_arrangement, time_allowed_for_other_methods)
            # mrs_planning_time = time.time() - start_time
            # mrs_motion_planning_time = unidir_mrs_planner.motion_planning_time
            # mrs_task_planning_time = mrs_planning_time - mrs_motion_planning_time
            # mrs_isSolved = unidir_mrs_planner.isSolved
            # mrs_nActions = unidir_mrs_planner.best_solution_cost
            # if time_allowed_for_other_methods == final_experimenter.time_allowed_monotone:
            #     final_experimenter.mRS_time_obj_m.append(mrs_planning_time)
            #     final_experimenter.mRS_motionTime_obj_m.append(mrs_motion_planning_time)
            #     final_experimenter.mRS_taskTime_obj_m.append(mrs_task_planning_time)
            #     final_experimenter.mRS_success_obj_m.append(mrs_isSolved)
            #     if mrs_nActions != np.inf:
            #         final_experimenter.mRS_nActions_obj_m.append(mrs_nActions)
            #     else:
            #         mrs_nActions = 5000
            # else:
            #     ### non-monotone problem solution
            #     final_experimenter.mRS_time_obj_nm.append(mrs_planning_time)
            #     final_experimenter.mRS_motionTime_obj_nm.append(mrs_motion_planning_time)
            #     final_experimenter.mRS_taskTime_obj_nm.append(mrs_task_planning_time)
            #     final_experimenter.mRS_success_obj_nm.append(mrs_isSolved)
            #     if mrs_nActions != np.inf:
            #         final_experimenter.mRS_nActions_obj_nm.append(mrs_nActions)
            #     else:
            #         mrs_nActions = 5000
            # all_methods_time_instance.append(mrs_planning_time)
            # all_methods_motionTime_instance.append(mrs_motion_planning_time)
            # all_methods_taskTime_instance.append(mrs_task_planning_time)
            # all_methods_success_instance.append(float(mrs_isSolved))
            # all_methods_nActions_instance.append(mrs_nActions)

            # #####################################################################
            # reset_instance_success = utils2.resetInstance("Right_torso")
            # #####################################################################
                
            #############################################################################################
            ### this instance has been tested on all methods
            if time_allowed_for_other_methods == final_experimenter.time_allowed_monotone:
                ### monotone instance
                num_monotoneInstancesSaved += 1
                tempInstanceFolder = os.path.join(final_experimenter.monotoneFolder, str(num_monotoneInstancesSaved))
            else:
                ### non-monotone instance
                num_nonMonotoneInstancesSaved += 1
                tempInstanceFolder = os.path.join(final_experimenter.nonMonotoneFolder, str(num_nonMonotoneInstancesSaved))
            totalNum_instancesSaved = num_monotoneInstancesSaved + num_nonMonotoneInstancesSaved
            utils2.saveInstance(cylinder_objects, tempInstanceFolder)
            utils2.saveSolution2(
                all_methods_time_instance, all_methods_motionTime_instance, all_methods_taskTime_instance, \
                all_methods_success_instance, all_methods_nActions_instance, tempInstanceFolder)
            
            ### Before moving on to the next instance, clear the current instance
            clear_instance_success = utils2.clearInstance("Right_torso")
            # input("check the instance clearance!!!")
        
        ### reach here as all experiments have been finished for the #objects specified
        ### we need to save the avarage results for each method for the #objects specified
        final_experimenter.saveAverageSolutionPerNumObject()
        ### after that, move on to the next parameter for #objects
    
    ### reach here as you finish all experiments, congrats!

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    main(sys.argv)