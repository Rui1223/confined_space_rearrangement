#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
import pybullet as p
import pybullet_data

import os
import sys
import random
import shutil
import math
import numpy as np
from scipy import spatial
import time
import IPython
import subprocess
from operator import itemgetter
import copy
import pickle
from collections import OrderedDict

import utils
from CollisionChecker import CollisionChecker

import rospy
from rospkg import RosPack
from confined_space_rearrangement.srv import AstarPathFindingNonLabeled, AstarPathFindingNonLabeledRequest
from confined_space_rearrangement.srv import AstarPathFindingLabeled, AstarPathFindingLabeledRequest
from confined_space_rearrangement.msg import Edge


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class Planner(object):
    def __init__(self, rosPackagePath, server,
        isObjectInLeftHand=False, isObjectInRightHand=False,
        objectInLeftHand=None, objectInRightHand=None):
        self.planningServer = server
        self.rosPackagePath = rosPackagePath
        self.roadmapFolder = os.path.join(self.rosPackagePath, "roadmaps")
        self.collisionAgent_p = CollisionChecker(self.planningServer)
        self.nodes = {}
        self.nodes["Right_torso"] = []
        self.isObjectInLeftHand = isObjectInLeftHand
        self.isObjectInRightHand = isObjectInRightHand
        self.objectInLeftHand = objectInLeftHand
        self.objectInRightHand = objectInRightHand
        self.objectInLeftHand_idx = -1
        self.objectInRightHand_idx = -1
        self.leftLocalPose = [[-1, -1, -1], [-1, -1, -1, -1]]
        self.rightLocalPose = [[-1, -1, -1], [-1, -1, -1, -1]]
        self.query_idx = 1 ### record the current planning query index
        self.loadIKdataset()
        self.deserializeCandidatesConfigPoses()


    def resetPlannerParams(self):
        self.isObjectInLeftHand = False
        self.isObjectInRightHand = False
        self.objectInLeftHand = None
        self.objectInRightHand = None
        self.objectInLeftHand_idx = -1
        self.objectInRightHand_idx = -1
        self.leftLocalPose = [[-1, -1, -1], [-1, -1, -1, -1]]
        self.rightLocalPose = [[-1, -1, -1], [-1, -1, -1, -1]]
        self.query_idx = 1 ### record the current planning query index        

    def setRobotToConfig(self, ik_config, robot, armType):
        ### this function set the robot to certain config (with stepSimulation)
        ### this function must be called in the following situation
        ### (1) right before the collision check
        ### (2) attempt to get the pose of the end effector
        if armType == "Left_torso" or armType == "Right_torso":
            robot.setSingleArmToConfig_torso(ik_config[1:8], ik_config[0], armType)
        else:
            robot.setSingleArmToConfig(ik_config, armType)

    def loadIKdataset(self):
        armType = "Right_torso"
        self.IK_dataset_Right_torso = []
        ikdatasetFile = self.roadmapFolder + "/ik_dataset_" + str(armType) + ".txt"
        f_ikdataset = open(ikdatasetFile, "r")
        for line in f_ikdataset:
            line = line.split()
            line = [float(e) for e in line]
            self.IK_dataset_Right_torso.append(line)
        f_ikdataset.close()

    def loadSamples(self):
        arms = ["Right_torso"]
        ############### load the samples ###############
        for armType in arms:
            samplesFile = self.roadmapFolder + "/samples_" + str(armType) + ".txt"
            f_samples = open(samplesFile, "r")
            for line in f_samples:
                line = line.split()
                line = [float(e) for e in line[1:]]
                self.nodes[armType].append(line)
            f_samples.close()
        ################################################
        self.nsamples = len(self.nodes["Right_torso"])
        ### specify the needed parameters
        self.neighbors_const = 3.5 * math.e * (1 + 1.0/8)
        ### use k_n to decide the number of neighbors: #neighbors = k_n * log(#samples)
        self.num_neighbors = int(self.neighbors_const * math.log(self.nsamples))
        if self.num_neighbors > self.nsamples:
            self.num_neighbors = self.nsamples
        print("nsamples: ", self.nsamples)
        print("num_neighbors: ", self.num_neighbors)

    def generateSamples(self, nsamples, robot, workspace, mode="configuration_space"):
        ### mode: decide which space do you sample from
        ### (1) configuration_space
        ### (2) cartesian_space
        ### (3) hybrid_space
        if not os.path.exists(self.roadmapFolder):
            os.makedirs(self.roadmapFolder)
        self.nsamples = nsamples
        ### specify the needed parameters
        self.neighbors_const = 3.5 * math.e * (1 + 1.0/8)
        ### use k_n to decide the number of neighbors: #neighbors = k_n * log(#samples)
        self.num_neighbors = int(self.neighbors_const * math.log(self.nsamples))
        if self.num_neighbors > self.nsamples:
            self.num_neighbors = self.nsamples
        print("nsamples: ", self.nsamples)
        print("num_neighbors: ", self.num_neighbors)
        ################ sampling ################
        arms = ["Right_torso"]
        for armType in arms:
            if mode == "configuration_space":
                self.samplingNodes_configurationSpace(robot, workspace, armType)
                samplesFile = self.roadmapFolder + "/samples_" + str(armType) + "_normal.txt"
            if mode == "hybrid_space":
                self.samplingNodes_hybridSpace(robot, workspace, armType)
                samplesFile = self.roadmapFolder + "/samples_" + str(armType) + ".txt"
            self.saveSamplesToFile(samplesFile, armType)
        ##########################################

    def samplingNodes_hybridSpace(self, robot, workspace, armType):
        temp_counter = 0
        ###### (i) let's first generate 1000 random samples in the configuration space ######
        while temp_counter < int(self.nsamples * 0.5):
            ### sample an IK configuration
            ikSolution = self.singleSampling_CSpace(robot, armType) ### this is for a single arm
            ########## move the robot to that configuration and then check ##########
            self.setRobotToConfig(ikSolution, robot, armType)
            isIKFallIntoRightRegion = self.sampleRegionCheck(robot, workspace, armType)
            if not isIKFallIntoRightRegion:
                continue
            isValid, FLAG = self.checkConfig_CollisionWithRobotAndKnownGEO(robot, workspace)
            if isValid:
                self.nodes[armType].append(ikSolution)
                temp_counter += 1
                print("finish the %s node" % str(temp_counter))
                # input("I want to see these points")
        ##########################################################################

        ###### (ii) then generate 1000 random cartesian poses of interest ######
        while temp_counter < self.nsamples:
            ### sample an workspace pose
            sample_pose = self.singleSampling_workspace(robot, workspace, armType)
            ### assign a rest_pose
            rest_pose = self.randomRestPose(robot, armType)
            isIKValid, FLAG, ikSolution = self.generateConfigBasedOnSamplePose(
                                sample_pose, rest_pose, robot, workspace, armType)
            if isIKValid:
                self.nodes[armType].append(ikSolution)
                temp_counter += 1
                print("finish the %s node" % str(temp_counter))
                # input("I want to see these points")
        ##########################################################################

    def samplingNodes_configurationSpace(self, robot, workspace, armType):
        temp_counter = 0
        ### Let's start
        while temp_counter < self.nsamples:
            ### sample an IK configuration
            ikSolution = self.singleSampling_CSpace(robot, armType) ### this is for a single arm
            ########## move the robot to that configuration and then check ##########
            self.setRobotToConfig(ikSolution, robot, armType)
            isIKFallIntoRightRegion = self.sampleRegionCheck(robot, workspace, armType)
            if not isIKFallIntoRightRegion:
                continue
            isValid, FLAG = self.checkConfig_CollisionWithRobotAndKnownGEO(robot, workspace)
            if isValid:
                self.nodes[armType].append(ikSolution)
                temp_counter += 1
                # input("I want to see these points")
                print("finish the %s node" % str(temp_counter))
            ##########################################################################

    def singleSampling_CSpace(self, robot, armType):
        if armType == "Left" or "Left_torso":
            first_joint_index = 1
        if armType == "Right" or "Right_torso":
            first_joint_index = 8
        nArmJoints = int(len(robot.rightArmHomeConfiguration))
        ikSolution = []
        ### first consider if you have torso to handle
        if armType == "Left_torso" or armType == "Right_torso":
            ikSolution.append(random.uniform(robot.ll[0], robot.ul[0]))
        ### Now handle the arm
        for i in range(nArmJoints):
            ikSolution.append(
                random.uniform(robot.ll[first_joint_index+i], robot.ul[first_joint_index+i]))

        return ikSolution

    def singleSampling_workspace(self, robot, workspace, armType):
        ### return a pose [[x,y,z], [x,y,z,w]]
        ### first sample position
        temp_pos = [
            random.uniform(workspace.constrained_area_x_limit[0], \
                workspace.constrained_area_x_limit[1] - workspace.side_clearance_x - workspace.cylinder_radius),
            random.uniform(workspace.constrained_area_y_limit[0] + workspace.side_clearance_y + workspace.cylinder_radius, \
                workspace.constrained_area_y_limit[1] - workspace.side_clearance_y - workspace.cylinder_radius),
            random.uniform(workspace.tablePosition[2] + workspace.table_dim[2]/2 + workspace.cylinder_height, \
                workspace.tablePosition[2] + workspace.table_dim[2]/2 + workspace.ceiling_height - workspace.thickness_flank*2)
        ]
        ### now sample orientation
        default_orientation=np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        possible_angles = [0, -15, -45, -30, 15, 45, 30]
        sample_angle = random.choice(possible_angles) * math.pi / 180.0
        rotation = np.array([[math.cos(sample_angle), -math.sin(sample_angle), 0], 
                                [math.sin(sample_angle), math.cos(sample_angle), 0], 
                                [0, 0, 1]])
        orientation = np.dot(default_orientation, rotation)
        temp_quat = utils.getQuaternionFromRotationMatrix(orientation)

        temp_pose = [temp_pos, list(temp_quat)]
        return temp_pose

    def sampleRegionCheck(self, robot, workspace, armType):
        if armType == "Left" or armType == "Left_torso":
            ee_idx = robot.left_ee_idx
        else:
            ee_idx = robot.right_ee_idx

        isIKFallIntoRightRegion = False
        pos_quat = p.getLinkState(robot.motomanGEO, ee_idx, physicsClientId=self.planningServer)
        pos = list(pos_quat[0])
        if (pos[2] > workspace.tablePosition[2] + workspace.table_dim[2] / 2 + workspace.ceiling_height):
            return isIKFallIntoRightRegion
        # if (pos[0] > workspace.constrained_area_x_limit[0] and pos[0] < workspace.constrained_area_x_limit[1]) and \
        #    (pos[1] > workspace.constrained_area_y_limit[0] and pos[1] < workspace.constrained_area_y_limit[1]):
        #     if (pos[2] < workspace.tablePosition[2] + workspace.table_dim[2] / 2 + workspace.cylinder_height / 2 + 0.01):
        #         return isIKFallIntoRightRegion
        return True

    def saveSamplesToFile(self, samplesFile, armType):
        f_samples = open(samplesFile, "w")
        for node_idx in range(len(self.nodes[armType])):
            node = self.nodes[armType][node_idx]
            f_samples.write(str(node_idx))
            for k in range(len(node)):
                f_samples.write(" " + str(node[k]))
            f_samples.write("\n")
        f_samples.close()

    def samplesConnect(self, robot, workspace, armType):
        connectivity = np.zeros((self.nsamples, self.nsamples))
        tree = spatial.KDTree(self.nodes[armType]) ### use KD tree to arrange neighbors assignment
        connectionsFile = self.roadmapFolder + "/connections_" + str(armType) + "_normal.txt"
        f_connection = open(connectionsFile, "w")
        ### for each node
        for node_idx in range(len(self.nodes[armType])):
            queryNode = self.nodes[armType][node_idx]
            knn = tree.query(queryNode, k=self.num_neighbors, p=2)

            neighbors_connected = 0
            ### for each potential neighbor
            for j in range(len(knn[1])):
                ### first check if this query node has already connected to enough neighbors
                if neighbors_connected >= self.num_neighbors:
                    break
                if knn[1][j] == node_idx:
                    ### if the neighbor is the query node itself
                    continue
                if connectivity[node_idx][knn[1][j]] == 1:
                    ### the connectivity has been checked before
                    neighbors_connected += 1
                    continue
                ### Otherwise, check the edge validity
                ### in terms of collision with the robot itself and all known geometries (e.g. table/shelf)
                ### between the query node and the current neighbor
                neighbor = self.nodes[armType][knn[1][j]]
                isEdgeValid, FLAG = self.checkEdgeValidity_knownGEO(queryNode, neighbor, robot, workspace, armType)
                if isEdgeValid:
                    ### write this edge information with their costs and labels into the txt file
                    f_connection.write(str(node_idx) + " " + str(knn[1][j]) + " " + str(knn[0][j]) + "\n")
                    connectivity[node_idx][knn[1][j]] = 1
                    connectivity[knn[1][j]][node_idx] = 1
                    neighbors_connected += 1
            print("Number of neighbors for current node " + str(node_idx) + ": " + str(neighbors_connected))
        f_connection.close()


    def updateMeshBasedonLocalPose(self, robot, workspace, armType):
        ### the planning version of the API "updateRealObjectBasedonLocalPose" in executor
        if armType == "Left" or armType == "Left_torso":
            # ee_idx = robot.left_ee_idx
            # objectInHand = self.objectInLeftHand
            # curr_ee_pose = robot.left_ee_pose
            # object_global_pose = self.getObjectGlobalPose(self.leftLocalPose, curr_ee_pose)
            object_global_pose = self.getObjectGlobalPose(self.leftLocalPose, robot.left_ee_pose)
            p.resetBasePositionAndOrientation(
                self.objectInLeftHand, object_global_pose[0], object_global_pose[1], 
                physicsClientId=self.planningServer)
            ### update the position and orientation of the object manipulated (ignore the orientation at this point)
            if self.objectInLeftHand_idx != -1:
                ### the object being updated is indeed an object of interest in the scene
                workspace.object_geometries[self.objectInLeftHand_idx].curr_pos = object_global_pose[0]
        if armType == "Right" or armType == "Right_torso":
            # ee_idx = robot.right_ee_idx
            # objectInHand = self.objectInRightHand
            # curr_ee_pose = robot.right_ee_pose
            # object_global_pose = self.getObjectGlobalPose(self.rightLocalPose, curr_ee_pose)
            object_global_pose = self.getObjectGlobalPose(self.rightLocalPose, robot.right_ee_pose)
            p.resetBasePositionAndOrientation(
                self.objectInRightHand, object_global_pose[0], object_global_pose[1], 
                physicsClientId=self.planningServer)
            ### update the position and orientation of the object manipulated (ignore the orientation at this point)
            if self.objectInRightHand_idx != -1:
                ### the object being updated is indeed an object of interest in the scene
                workspace.object_geometries[self.objectInRightHand_idx].curr_pos = object_global_pose[0]

    def getObjectGlobalPose(self, local_pose, ee_global_pose):
        temp_object_global_pose = p.multiplyTransforms(
            ee_global_pose[0], ee_global_pose[1],
            local_pose[0], local_pose[1])
        object_global_pose = [list(temp_object_global_pose[0]), list(temp_object_global_pose[1])]

        return object_global_pose 


    def randomRestPose(self, robot, armType):
        ### option 1: choose from IK dataset
        if armType == "Right_torso":
            temp_idx = random.randint(0, len(self.IK_dataset_Right_torso)-1)
            right_torso_rp = self.IK_dataset_Right_torso[temp_idx]
            rp = [right_torso_rp[0]] + robot.leftArmCurrConfiguration + right_torso_rp[1:8] + robot.rightHandCurrConfiguration
            return rp

        ### option 2
        # rp = []
        # for i_joint in range(len(robot.rp)):
        #     joint_value = random.uniform(robot.ll[i_joint], robot.ul[i_joint])
        #     rp.append(joint_value)
        # return rp

        ### option 3
        # rp = []
        # for i_joint in range(len(robot.rp)):
        #     if armType == 'Left_torso' or armType == "Right_torso":
        #         if i_joint == 0: 
        #             rp.append(robot.rp[i_joint])
        #             continue
        #     joint_value = robot.rp[i_joint] + random.uniform(-0.34, 0.34)
        #     ### clipping
        #     if joint_value < robot.ll[i_joint]:
        #         joint_value = robot.ll[i_joint] + 0.017
        #     if joint_value > robot.ul[i_joint]:
        #         joint_value = robot.ul[i_joint] - 0.017
        #     rp.append(joint_value)
        # return rp

    def detectInitialConstraintsWithConfigs(self, start_arrangement, target_arrangement, robot, workspace, armType):
        ### This function updates positionCandidateConfigs in terms of
        ### approaching_labels_all, grasping_labels_all, as well as total_labels_all
        ### related to the local task (start & target arrangement) 
        ### record the current configuration
        curr_robot_config = robot.getRobotCurrConfig()
        all_objects = [i for i in range(len(start_arrangement)) \
            if start_arrangement[i] != target_arrangement[i]]
        blockPrint()
        ### for each object
        for obj_idx in all_objects:
            print("obj_idx: " + str(obj_idx))
            ### get obj_start_position_idx and obj_target_position_idx
            obj_start_position_idx = start_arrangement[obj_idx]
            obj_target_position_idx = target_arrangement[obj_idx]
            print("obj_start_position_idx: " + str(obj_start_position_idx))
            print("obj_target_position_idx: " + str(obj_target_position_idx))
            ### get obj_start_configPoses and obj_target_configPoses
            if obj_start_position_idx >= workspace.num_candidates:
                ### the object is at the initial position
                obj_start_configPoses = self.object_initial_configPoses[obj_idx]
            else:
                ### the object is at certain position candidate
                obj_start_configPoses = self.position_candidates_configPoses[obj_start_position_idx]
            if obj_target_position_idx >= workspace.num_candidates:
                ### the object's goal position is an initial position (it should not happen)
                obj_target_configPoses = self.object_initial_configPoses[obj_idx]
            else:
                ### the object's goal position is at certain position candidate (definitely)
                obj_target_configPoses = self.position_candidates_configPoses[obj_target_position_idx]
            ### reset to each related approaching and grasping config
            if obj_start_configPoses.total_labels_all == []:
                for k in range(len(obj_start_configPoses.grasping_configs)):
                    print("check the kth pose, k is: " + str(k))
                    obj_start_approaching_config = obj_start_configPoses.approaching_configs[k]
                    obj_start_grasping_config = obj_start_configPoses.grasping_configs[k]
                    ### reset to approaching config first
                    self.setRobotToConfig(obj_start_approaching_config, robot, armType)
                    ### check initial labels
                    isConfigValid, FLAG, positionCollided_approaching = \
                        self.checkConfig_CollisionBetweenRobotAndStaticObjects_labeled(robot, workspace.initial_object_position_meshes)
                    print("FLAG: " + str(FLAG))
                    print("positionCollided_approaching: " + str(positionCollided_approaching))
                    ### update the labels_all
                    obj_start_configPoses.approaching_labels_all.append(
                        set(list(obj_start_configPoses.approaching_labels[k]) + positionCollided_approaching))
                    print("approaching_labels: " + str(obj_start_configPoses.approaching_labels[k]))
                    print("approaching_labels_all: " + str(obj_start_configPoses.approaching_labels_all[k]))
                    # input("press enter to continue")
                    ### rest to grasping config then
                    self.setRobotToConfig(obj_start_grasping_config, robot, armType)
                    ### check initial labels
                    isConfigValid, FLAG, positionCollided_grasping = \
                        self.checkConfig_CollisionBetweenRobotAndStaticObjects_labeled(robot, workspace.initial_object_position_meshes)
                    print("FLAG: " + str(FLAG))
                    print("positionCollided_grasping: " + str(positionCollided_grasping))
                    ### update the labels_all
                    obj_start_configPoses.grasping_labels_all.append(
                        set(list(obj_start_configPoses.grasping_labels[k]) + positionCollided_grasping))
                    print("grasping_labels: " + str(obj_start_configPoses.grasping_labels[k]))
                    print("grasping_labels_all: " + str(obj_start_configPoses.grasping_labels_all[k]))
                    # input("press enter to continue")
                    ###### finally update total_labels_all
                    obj_start_configPoses.total_labels_all.append(
                        set(list(obj_start_configPoses.approaching_labels_all[k]) + list(obj_start_configPoses.grasping_labels_all[k])))
                    print("*********start total_labels_all:" + str(obj_start_configPoses.total_labels_all))
                    # input("press enter to continue")
            # input("-------------------now check all targets-------------------")
            if obj_target_configPoses.total_labels_all == []:
                for k in range(len(obj_target_configPoses.grasping_configs)):
                    print("check the kth pose, k is: " + str(k))
                    obj_target_approaching_config = obj_target_configPoses.approaching_configs[k]
                    obj_target_grasping_config = obj_target_configPoses.grasping_configs[k]
                    ### reset to approaching config first
                    self.setRobotToConfig(obj_target_approaching_config, robot, armType)
                    ### check initial labels
                    isConfigValid, FLAG, positionCollided_approaching = \
                        self.checkConfig_CollisionBetweenRobotAndStaticObjects_labeled(robot, workspace.initial_object_position_meshes)
                    print("positionCollided_approaching: " + str(positionCollided_approaching))
                    ### update the labels_all
                    obj_target_configPoses.approaching_labels_all.append(
                        set(list(obj_target_configPoses.approaching_labels[k]) + positionCollided_approaching))
                    print("approaching_labels: " + str(obj_target_configPoses.approaching_labels[k]))
                    print("approaching_labels_all: " + str(obj_target_configPoses.approaching_labels_all[k]))
                    # input("press enter to continue")
                    ### rest to grasping config then
                    self.setRobotToConfig(obj_target_grasping_config, robot, armType)
                    ### check initial labels
                    isConfigValid, FLAG, positionCollided_grasping = \
                        self.checkConfig_CollisionBetweenRobotAndStaticObjects_labeled(robot, workspace.initial_object_position_meshes)
                    print("positionCollided_grasping: " + str(positionCollided_grasping))
                    ### update the labels_all
                    obj_target_configPoses.grasping_labels_all.append(
                        set(list(obj_target_configPoses.grasping_labels[k]) + positionCollided_grasping))
                    print("grasping_labels: " + str(obj_target_configPoses.grasping_labels[k]))
                    print("grasping_labels_all: " + str(obj_target_configPoses.grasping_labels_all[k]))
                    # input("press enter to continue")
                    ###### finally update total_labels_all
                    obj_target_configPoses.total_labels_all.append(
                        set(list(obj_target_configPoses.approaching_labels_all[k]) + list(obj_target_configPoses.grasping_labels_all[k])))
                    print("*********target total_labels_all:" + str(obj_target_configPoses.total_labels_all))
                    # input("press enter to continue")
        enablePrint()
        ### move the robot back
        robot.resetArmConfig_torso(curr_robot_config[1:15], curr_robot_config[0])
        robot.resetRightHandConfig(curr_robot_config[15:21])


    def obtainCurrObjectConfigPoses(self, workspace, object_idx):
        ### This function feteches the current object's configPoses specified by object_idx
        curr_obj_position_idx = workspace.object_geometries[object_idx].curr_position_idx
        if curr_obj_position_idx >= workspace.num_candidates:
            ### the object is at the initial position
            curr_object_configPoses = self.object_initial_configPoses[object_idx]
        else:
            ### the object is at certain position candidate
            curr_object_configPoses = self.position_candidates_configPoses[curr_obj_position_idx]
        return curr_object_configPoses

    def getConstraintsFromLabels_mix(self, configPoses, obj_idx, start_arrangement, target_arrangement, manipulation_mode):
        '''This function gets all objects start/target constraints from labels
        stored in the configPoses'''
        ### configPoses: a PositionCandidateConfigs object
        ### start_arrangement: a list/tuple of object_indices
        ### target_arrangement: a list/tuple of object_indices
        ### manipulation_mode: "picking" or "placing"
        start_arrangement = list(start_arrangement)
        target_arrangement = list(target_arrangement)
        configPoses_constraints = []
        if manipulation_mode == "picking":
            all_pose_labels = configPoses.total_labels_all
        if manipulation_mode == "placing":
            all_pose_labels = configPoses.grasping_labels_all
        # print("all_pose_labels: " + str(all_pose_labels))
        for pose_i in range(len(all_pose_labels)):
            pose_labels = all_pose_labels[pose_i]
            configPoses_constraints.append([])
            for label in pose_labels:
                if (label in target_arrangement) and (target_arrangement.index(label) != obj_idx):
                    ### get the object_idx of the object that occupied that label at target arrangement
                    configPoses_constraints[pose_i].append( (target_arrangement.index(label), True) )
                elif (label in start_arrangement) and (start_arrangement.index(label) != obj_idx):
                    ### get the object_idx of the object that occupied that label at start arrangement
                    configPoses_constraints[pose_i].append( (start_arrangement.index(label), False) )
        #     print("constraint for pose " + str(pose_i) + ": " + str(configPoses_constraints[pose_i]))
        #     input("wait")
        # print("==========")
        # print(configPoses_constraints)
        # print("\n")
        return configPoses_constraints
    
    def getConstraintsFromLabels(self, configPoses, obj_idx, target_arrangement, manipulation_mode):
        '''This function gets all objects target constraints from labels
        stored in the configPoses'''
        ### configPoses: a PositionCandidateConfigs object
        ### target_arrangement: a list/tuple of object_indices
        ### manipulation_mode: "picking" or "placing"
        target_arrangement = list(target_arrangement)
        configPoses_constraints = []
        if manipulation_mode == "picking":
            all_pose_labels = configPoses.total_labels
        if manipulation_mode == "placing":
            all_pose_labels = configPoses.grasping_labels
        for pose_i in range(len(all_pose_labels)):
            pose_labels = all_pose_labels[pose_i]
            configPoses_constraints.append([])
            for label in pose_labels:
                if (label in target_arrangement) and (target_arrangement.index(label) != obj_idx):
                    ### get the object_idx of the object that occupied that label at target arrangement
                    configPoses_constraints[pose_i].append( (target_arrangement.index(label), True) )
        return configPoses_constraints

    def addInvalidArrStates_mix(self, configPoses_constraints, object_idx):
        '''This functions add invalid arrangement states given
        configPoses_constraints [ [(obj_idx, True), ..., (obj_idx, False)], ..., []]'''
        ### first get failure reasons
        # start_time = time.time()
        failure_reasons = []
        if [] in configPoses_constraints: return
        utils.generateCombination(configPoses_constraints, 0, [], failure_reasons)
        # print("time for generate combinations: " + str(time.time() - start_time))
        ### once we get the failure reasons, construct and add invalid states
        ### (1) for reasons where all constraints are related to goal constraints, go with forward checking
        ### (2) for reasons where there is at least one start constraint, go with instant checking
        # print("\n-------object_idx: " + str(object_idx) + "--------")
        for failure_reason in failure_reasons:
            # print("current failure reason: " + str(failure_reason))
            goal_counts = sum([failure_reason[mm][1] for mm in range(len(failure_reason))])
            # print("goal_counts: " + str(goal_counts))
            if goal_counts == len(failure_reason):
                # print("all-goal-related constraints")
                ### all constraints are goal constraints, go with forward checking
                for (cstr_obj_idx, isGoal) in failure_reason:
                    invalid_state = {} ### do not make it OrderedDict()
                    invalid_state[object_idx] = False
                    for (obj_idx, isGoal) in failure_reason:
                        if obj_idx == cstr_obj_idx: continue
                        invalid_state[obj_idx] = True
                    self.invalid_arr_states_per_obj[cstr_obj_idx].append(invalid_state)
                    # print(str(cstr_obj_idx) + ": " + str(invalid_state))
            else:
                # print("not all-goal-related constraints")
                ### at least one constraint is start constraint, go with instant checking (single)
                temp_success = True
                invalid_state = {}
                for (obj_idx, isGoal) in failure_reason:
                    if obj_idx in invalid_state.keys():
                        ### you are adding conflicting events which will not concur
                        temp_success = False
                        break
                    else:
                        invalid_state[obj_idx] = isGoal
                if temp_success:
                    self.invalid_arr_states_per_obj[object_idx].append(invalid_state)
                    # print(str(object_idx) + ": " + str(invalid_state))

            # input("wait to continue")
            # print("\n")


    def addInvalidArrStates(self, configPoses_constraints, object_idx):
        '''This functions add invalid arrangement states given
        configPoses_constraints [ [(obj_idx, True), ..., (obj_idx, False)], ..., []]'''
        ### configPoses_constraints: [[obj_indices], [obj_indices], [obj_indices]] (old)
        ### first get failure reasons
        failure_reasons = []
        if [] in configPoses_constraints: return
        utils.generateCombination(configPoses_constraints, 0, [], failure_reasons)
        ### once we get the failure reasons, construct and add invalid states
        for failure_reason in failure_reasons:
            for (cstr_obj_idx, isGoal) in failure_reason:
                invalid_state = {} ### do not make it OrderedDict()
                invalid_state[object_idx] = False
                for (obj_idx, isGoal) in failure_reason:
                    if obj_idx == cstr_obj_idx: continue
                    invalid_state[obj_idx] = True
                self.invalid_arr_states_per_obj[cstr_obj_idx].append(invalid_state)


    def generatePrePickingPose(self, pose, rest_config, robot, workspace, armType):
        ### This function generates a pre-picking (post-placing) pose based on grasp pose
        ### It is 5cm (0.05m) behind the approaching direction of z axis (local axis of the end effector)
        ### Input: pose: [[x,y,z], [x,y,z,w]]
        if armType == "Left" or armType == "Left_torso":
            ee_idx = robot.left_ee_idx
            first_joint_index = 1
        if armType == "Right" or armType == "Right_torso":
            ee_idx = robot.right_ee_idx
            first_joint_index = 8

        if armType == "Left":
            rest_pose = robot.torsoCurrConfiguration + rest_config + robot.rightArmCurrConfiguration + robot.rightHandCurrConfiguration
        if armType == "Right":
            rest_pose = robot.torsoCurrConfiguration + robot.leftArmCurrConfiguration + rest_config + robot.rightHandCurrConfiguration
        if armType == "Left_torso":
            rest_pose = [rest_config[0]] + rest_config[1:8] + robot.rightArmCurrConfiguration + robot.rightHandCurrConfiguration
        if armType == "Right_torso":
            rest_pose = [rest_config[0]] + robot.leftArmCurrConfiguration + rest_config[1:8] + robot.rightHandCurrConfiguration

        temp_rot_matrix = p.getMatrixFromQuaternion(pose[1])
        ### local z-axis of the end effector
        temp_approaching_direction = [temp_rot_matrix[2], temp_rot_matrix[5], temp_rot_matrix[8]]
        temp_pos = list(np.array(pose[0]) - 0.05*np.array(temp_approaching_direction))
        new_pose = [temp_pos, pose[1]] ### the quaternion remains the same as input pose

        ### check the IK of the pose
        q_newPoseIK = p.calculateInverseKinematics(bodyUniqueId=robot.motomanGEO, 
                                endEffectorLinkIndex=ee_idx, 
                                targetPosition=new_pose[0], 
                                targetOrientation=new_pose[1], 
                                lowerLimits=robot.ll, upperLimits=robot.ul, 
                                jointRanges=robot.jr, restPoses=rest_pose,
                                maxNumIterations=2000, residualThreshold=0.0000001,
                                physicsClientId=robot.server)
        if armType == "Left" or armType == "Right":
            singleArmConfig_IK = list(q_newPoseIK[first_joint_index:first_joint_index+7])
        if armType == "Left_torso" or armType == "Right_torso":
            singleArmConfig_IK = [q_newPoseIK[0]] + list(q_newPoseIK[first_joint_index:first_joint_index+7])
        ########## move the robot to that configuration and then check ##########
        self.setRobotToConfig(singleArmConfig_IK, robot, armType)
        isIKValid, FLAG = self.checkPoseIK(new_pose, robot, workspace, armType)
        ##########################################################################

        trials = 0
        while (not isIKValid) and (trials < 1):
            ### try another IK (not specify rest pose)
            ### try another IK given curernt rest pose with random noise
            rest_pose = self.randomRestPose(robot, armType)
            q_newPoseIK = p.calculateInverseKinematics(bodyUniqueId=robot.motomanGEO,
                                    endEffectorLinkIndex=ee_idx,
                                    targetPosition=new_pose[0],
                                    targetOrientation=new_pose[1],
                                    lowerLimits=robot.ll, upperLimits=robot.ul, 
                                    jointRanges=robot.jr, restPoses=rest_pose,
                                    maxNumIterations=2000, residualThreshold=0.0000001,
                                    physicsClientId=robot.server)
            if armType == "Left" or armType == "Right":
                singleArmConfig_IK = list(q_newPoseIK[first_joint_index:first_joint_index+7])
            if armType == "Left_torso" or armType == "Right_torso":
                singleArmConfig_IK = [q_newPoseIK[0]] + list(q_newPoseIK[first_joint_index:first_joint_index+7])
            ########## move the robot to that configuration and then check ##########
            self.setRobotToConfig(singleArmConfig_IK, robot, armType)
            isIKValid, FLAG = self.checkPoseIK(new_pose, robot, workspace, armType)
            if isIKValid: break
            ### otherwise
            trials += 1
            ##########################################################################
        ### you need to return both the statement whether the pose is valid
        ### and the valid configuration the pose corresponds to
        return isIKValid, FLAG, new_pose, singleArmConfig_IK


    def generateConfigBasedOnPose(self, pose, rest_config, robot, workspace, armType):
        ### This function checks the validity of a pose by checking its corresponding config (IK)
        ### Input: pose: [[x,y,z],[x,y,z,w]]
        ###        armType: "Left" or "Right"
        ### Output: isIKValid (bool), singleArmConfig_IK (list)), FLAG
        if armType == "Left" or armType == "Left_torso":
            ee_idx = robot.left_ee_idx
            first_joint_index = 1
        if armType == "Right" or armType == "Right_torso":
            ee_idx = robot.right_ee_idx
            first_joint_index = 8
        
        if armType == "Left":
            rest_pose = robot.torsoCurrConfiguration + rest_config + robot.rightArmCurrConfiguration + robot.rightHandCurrConfiguration
        if armType == "Right":
            rest_pose = robot.torsoCurrConfiguration + robot.leftArmCurrConfiguration + rest_config + robot.rightHandCurrConfiguration
        if armType == "Left_torso":
            rest_pose = [rest_config[0]] + rest_config[1:8] + robot.rightArmCurrConfiguration + robot.rightHandCurrConfiguration
        if armType == "Right_torso":
            rest_pose = [rest_config[0]] + robot.leftArmCurrConfiguration + rest_config[1:8] + robot.rightHandCurrConfiguration

        ### we add rest pose in the IK solver to get as high-quality IK as possible
        config_IK = p.calculateInverseKinematics(bodyUniqueId=robot.motomanGEO,
                                endEffectorLinkIndex=ee_idx,
                                targetPosition=pose[0],
                                targetOrientation=pose[1],
                                lowerLimits=robot.ll, upperLimits=robot.ul, 
                                jointRanges=robot.jr, restPoses=rest_pose,
                                maxNumIterations=2000, residualThreshold=0.0000001,
                                physicsClientId=robot.server)
        if armType == "Left" or armType == "Right":
            singleArmConfig_IK = list(config_IK[first_joint_index:first_joint_index+7])
        if armType == "Left_torso" or armType == "Right_torso":
            singleArmConfig_IK = [config_IK[0]] + list(config_IK[first_joint_index:first_joint_index+7])
        ########## move the robot to that configuration and then check ##########
        self.setRobotToConfig(singleArmConfig_IK, robot, armType)
        isIKValid, FLAG = self.checkPoseIK(pose, robot, workspace, armType)
        ##########################################################################
        
        trials = 0
        while (not isIKValid) and (trials < 1):
            ### try another IK (not specify rest pose)
            ### try another IK given curernt rest pose with random noise
            rest_pose = self.randomRestPose(robot, armType)
            config_IK = p.calculateInverseKinematics(bodyUniqueId=robot.motomanGEO,
                                    endEffectorLinkIndex=ee_idx,
                                    targetPosition=pose[0],
                                    targetOrientation=pose[1],
                                    lowerLimits=robot.ll, upperLimits=robot.ul, 
                                    jointRanges=robot.jr, restPoses=rest_pose,
                                    maxNumIterations=2000, residualThreshold=0.0000001,
                                    physicsClientId=robot.server)
            if armType == "Left" or armType == "Right":
                singleArmConfig_IK = list(config_IK[first_joint_index:first_joint_index+7])
            if armType == "Left_torso" or armType == "Right_torso":
                singleArmConfig_IK = [config_IK[0]] + list(config_IK[first_joint_index:first_joint_index+7])
            ########## move the robot to that configuration and then check ##########
            self.setRobotToConfig(singleArmConfig_IK, robot, armType)
            isIKValid, FLAG = self.checkPoseIK(pose, robot, workspace, armType)
            if isIKValid: break
            ### otherwise
            trials += 1
            ##########################################################################
        ### you need to return both the statement whether the pose is valid
        ### and the valid configuration the pose corresponds to
        return isIKValid, FLAG, singleArmConfig_IK

    def checkPoseIK(self, desired_ee_pose, robot, workspace, armType):
        ######## before calling this function, don't forget to call API: setRobotToConfig ########
        ### This function checks if an IK solution is valid in terms of
        ### (1) small error from the desired pose (position error + quaternion error)
        ### (2) no collision occurred (all types of collisions)

        ### Input: desired_ee_pose: [[x,y,z], [x,y,z,w]]
        ###        armType: "Left" or "Right"
        ### Output: isConfigValid (bool) indicating whether the IK is valid
        ###         FLAG (int): indicating the validity/invalidity reason
        ###                     0: no any issues (pass all IK check)
        ###                     1: reachability
        ###                     2: robot self-collision
        ###                     3: collision between robot and known geometries
        ###                     4: collision between robot and static objects
        ###                     5: collision between robot and the moving object  
        ###                     6: the moving object collides with known geometries
        ###                     7: the moving object collides with static objects
        
        isConfigValid, FLAG = self.checkConfig_reachability(desired_ee_pose, robot, armType)
        if not isConfigValid: 
            return isConfigValid, FLAG
        ### Congrats! The IK successfully passed reachability checker. 
        ### Then check if there is collision
        isConfigValid, FLAG = self.checkConfig_AllCollisions(robot, workspace, armType)
        return isConfigValid, FLAG


    def checkConfig_reachability(self, desired_ee_pose, robot, armType):
        ######## before calling this function, don't forget to call API: setRobotToConfig ########
        ### This function purely checks if an IK solution is reachable
        ### small position and orientation error
        if armType == "Left" or armType == "Left_torso":
            actual_ee_pose = copy.deepcopy(robot.left_ee_pose)
        if armType == "Right" or armType == "Right_torso":
            actual_ee_pose = copy.deepcopy(robot.right_ee_pose)
        ### if the ee_idx is within 2.0cm(0.02m) Euclidean distance from the desired one, we accept it
        ee_dist_pos = utils.computePoseDist_pos(actual_ee_pose[0], desired_ee_pose[0])
        # print("actual_pose: " + str(actual_ee_pose[0]))
        # print("desired_pose: " + str(desired_ee_pose))
        # print("position error: " + str(ee_dist_pos))
        if ee_dist_pos > 0.012:
            # print("IK not reachable as position error exceeds 1.2cm: " + str(ee_dist_pos))
            ### raise the flag
            isConfigValid = False
            FLAG = 1
            return isConfigValid, FLAG
        else:
            ### Now check orientation error
            ee_dist_quat = utils.computePoseDist_quat(actual_ee_pose[1], desired_ee_pose[1])
            if ee_dist_quat > 0.8:
                # print("IK not reachable as quaternion error exceeds 0.8: " + str(ee_dist_quat))
                ### raise the flag
                isConfigValid = False
                FLAG = 1
                return isConfigValid, FLAG
        ### otherwise, this config (IK) is reachable
        isConfigValid = True
        FLAG = 0
        return isConfigValid, FLAG

    
    def checkConfig_AllCollisions(self, robot, workspace, armType):
        ######## before calling this function, don't forget to call API: setRobotToConfig ########
        ### This function checks all collisions
        ### Common: no robot self collision and collsions between robot and knownGEO AT ALL TIME
        ### no other collisions
        ###     (i) no object in hand (e.g., "transit"): no collision between the robot and the static object
        ###     (ii) object in hand (e.g., "transfer"): 
        ###           no collision between the robot and the moving object
        ###           no collision between the moving object and knownGEO

        ############  first check potential collision between robot and knownGEO & robot ############
        ### FLAG: (2,3) or 0
        isConfigValid, FLAG = self.checkConfig_CollisionWithRobotAndKnownGEO(robot, workspace) 
        if isConfigValid == False: return isConfigValid, FLAG
        #############################################################################################

        ############ then check potentinal collisions with objects not in hand ######################
        ### first get all the objects which are not in hand
        static_object_geometries = { obj_info.object_index : obj_info.geo \
            for obj_info in workspace.object_geometries.values() \
            if (obj_info.object_index != self.objectInLeftHand_idx) and \
                (obj_info.object_index != self.objectInRightHand_idx) }
        ### FLAG: 4
        isConfigValid, FLAG = self.checkConfig_CollisionBetweenRobotAndStaticObjects(
                                                                robot, static_object_geometries)
        if isConfigValid == False: return isConfigValid, FLAG
        #############################################################################################

        ############## then check potential collision arising from moving objects ###################
        ### first see if there exists moving objects
        if (self.isObjectInLeftHand and (armType == "Left" or armType == "Left_torso")) or \
                    (self.isObjectInRightHand and (armType == "Right" or armType == "Right_torso")):
            ### (i) first update the object in hand
            self.updateMeshBasedonLocalPose(robot, workspace, armType)
            if armType == "Left" or armType == "Left_torso":
                manipulation_objectGEO = self.objectInLeftHand
            if armType == "Right" or armType == "Right_torso":
                manipulation_objectGEO = self.objectInRightHand
            ### (ii) check the potential collision between the robot and the moving object
            ### FLAG: 5
            isConfigValid, FLAG = self.checkConfig_CollisionBetweenRobotAndMovingObject(
                                                        robot, manipulation_objectGEO, armType)
            if isConfigValid == False: return isConfigValid, FLAG
            ### (iii) check the potential collision between the moving object and known geometries
            ### FLAG: 6
            isConfigValid, FLAG = self.checkConfig_CollisionMovingObjectAndKnownGEO(
                                                            manipulation_objectGEO, workspace) 
            if isConfigValid == False: return isConfigValid, FLAG
            ### (iv) check the potential collision between the moving object and other static objects
            ### FLAG: 7
            isConfigValid, FLAG = self.checkConfig_CollisionMovingObjectAndStaticObjects(
                                                    manipulation_objectGEO, static_object_geometries)
            if isConfigValid == False: return isConfigValid, FLAG
        #############################################################################################
        ### reaching here since it pass all collision check
        return isConfigValid, FLAG

    def checkConfig_labelCollisions(self, robot, workspace, armType):
        ######## before calling this function, don't forget to call API: setRobotToConfig ########
        ### This function checks collisions for FLAG (4,5,6,7) with the prior knowledge
        ### that FLAG (2,3) has been checked before

        ############ check potentinal collisions with objects not in hand ######################
        ### first get all the objects which are not in hand
        static_object_geometries = { obj_info.object_index : obj_info.geo \
            for obj_info in workspace.object_geometries.values() \
            if (obj_info.object_index != self.objectInLeftHand_idx) and \
                (obj_info.object_index != self.objectInRightHand_idx) }
        ### FLAG: 4
        isConfigValid, FLAG, objectCollided = \
            self.checkConfig_CollisionBetweenRobotAndStaticObjects_labeled(robot, static_object_geometries)
        if isConfigValid == False: return isConfigValid, FLAG, objectCollided
        #############################################################################################

        ############## then check potential collision arising from moving objects ###################
        ### first see if there exists moving objects
        if (self.isObjectInLeftHand and (armType == "Left" or armType == "Left_torso")) or \
                    (self.isObjectInRightHand and (armType == "Right" or armType == "Right_torso")):
            ### (i) first update the object in hand
            self.updateMeshBasedonLocalPose(robot, workspace, armType)
            if armType == "Left" or armType == "Left_torso":
                manipulation_objectGEO = self.objectInLeftHand
            if armType == "Right" or armType == "Right_torso":
                manipulation_objectGEO = self.objectInRightHand
            ### (ii) check the potential collision between the robot and the moving object
            ### FLAG: 5
            isConfigValid, FLAG = self.checkConfig_CollisionBetweenRobotAndMovingObject(
                                                        robot, manipulation_objectGEO, armType)
            if isConfigValid == False: return isConfigValid, FLAG, objectCollided
            ### (iii) check the potential collision between the moving object and known geometries
            ### FLAG: 6
            isConfigValid, FLAG = self.checkConfig_CollisionMovingObjectAndKnownGEO(
                                                            manipulation_objectGEO, workspace) 
            if isConfigValid == False: return isConfigValid, FLAG, objectCollided
            ### (iv) check the potential collision between the moving object and other static objects
            ### FLAG: 7
            isConfigValid, FLAG, objectCollided = \
                self.checkConfig_CollisionMovingObjectAndStaticObjects_labeled(
                                                manipulation_objectGEO, static_object_geometries)
            if isConfigValid == False: return isConfigValid, FLAG, objectCollided
        #############################################################################################
        ### reaching here since it pass all collision check
        return isConfigValid, FLAG, objectCollided


    def checkConfig_CollisionWithRobotAndKnownGEO(self, robot, workspace):
        ######## before calling this function, don't forget to call API: setRobotToConfig ########

        ### check if there is collision
        if self.collisionAgent_p.collisionCheck_selfCollision(robot.motomanGEO) == True:
            # print("robot self collision")
            ### raise the flag
            isConfigValid = False
            FLAG = 2
            return isConfigValid, FLAG

        if self.collisionAgent_p.collisionCheck_robot_knownGEO(
                    robot.motomanGEO, workspace.known_geometries) == True:
            # print("robot collide with known geometries")
            ### raise the flag
            isConfigValid = False
            FLAG = 3
            return isConfigValid, FLAG
        ### if you reach here, the configuration passes current collision check 
        isConfigValid = True
        FLAG = 0
        return isConfigValid, FLAG

    def checkConfig_CollisionBetweenRobotAndStaticObjects(self, robot, object_geometries):
        ######## before calling this function, don't forget to call API: setRobotToConfig ########
        ### check if there is collision between the robot and the static objects (objects in hand excluded)
        ### object_geometries is a dictionary {obj_idx : obj_geo}
        if self.collisionAgent_p.collisionCheck_robot_staticObjectGEOs(robot.motomanGEO, object_geometries) == True:
            # print("collision between robot and the static objects")
            ### raise the flag
            isConfigValid = False
            FLAG = 4
            return isConfigValid, FLAG
        ### if you reach here, the configuration passes current collision check 
        isConfigValid = True
        FLAG = 0
        return isConfigValid, FLAG

    def checkConfig_CollisionBetweenRobotAndMovingObject(self, robot, movingObjectGEO, armType):
        if self.collisionAgent_p.collisionCheck_robot_movingObjectGEO(robot.motomanGEO, movingObjectGEO, armType) == True:
            # print("collision between robot and the moving object")
            ### raise the flag
            isConfigValid = False
            FLAG = 5
            return isConfigValid, FLAG
        ### if you reach here, the configuration passes current collision check 
        isConfigValid = True
        FLAG = 0
        return isConfigValid, FLAG
    
    def checkConfig_CollisionMovingObjectAndKnownGEO(self, movingObjectGEO, workspace):
        if self.collisionAgent_p.collisionCheck_object_knownGEO(movingObjectGEO, workspace.known_geometries) == True:
            # print("moving object collide with known geomtries")
            ### raise the flag
            isConfigValid = False
            FLAG = 6
            return isConfigValid, FLAG
        ### if you reach here, the configuration passes current collision check 
        isConfigValid = True
        FLAG = 0
        return isConfigValid, FLAG

    def checkConfig_CollisionMovingObjectAndStaticObjects(self, movingObjectGEO, object_geometries):
        ### object_geometries is a dictionary {obj_idx : obj_geo}
        if self.collisionAgent_p.collisionCheck_object_objectGEO(movingObjectGEO, object_geometries) == True:
            # print("moving object collide with remaining object geometries")
            ### raise the flag
            isConfigValid = False
            FLAG = 7
            return isConfigValid, FLAG
        ### if you reach here, the configuration passes current collision check 
        isConfigValid = True
        FLAG = 0
        return isConfigValid, FLAG

    def checkEdgeValidity_AllCollisions(self, n1, n2, robot, workspace, armType):
        ### Input: n1, n2: node (a list of 7 or 8 joint values)
        ### Output: bool value indicates whether the transition from n1 to n2 is valid 
        ###         and a flag which indicates the failure explanation

        # nseg = 5
        # min_degree = math.pi / 90
        min_degree = math.pi / 180 * 5 ### make it sparsely interpolated to speed up collision check

        if armType == "Left" or armType == "Right":
            nseg = int(max(
                abs(n1[0]-n2[0]), abs(n1[1]-n2[1]), abs(n1[2]-n2[2]), abs(n1[3]-n2[3]),
                abs(n1[4]-n2[4]), abs(n1[5]-n2[5]), abs(n1[6]-n2[6])) / min_degree)
            if nseg == 0: nseg += 1
            # print("nseg: " + str(nseg))

            for i in range(0, nseg+1):
                interm_j0 = n1[0] + (n2[0]-n1[0]) / nseg * i
                interm_j1 = n1[1] + (n2[1]-n1[1]) / nseg * i
                interm_j2 = n1[2] + (n2[2]-n1[2]) / nseg * i
                interm_j3 = n1[3] + (n2[3]-n1[3]) / nseg * i
                interm_j4 = n1[4] + (n2[4]-n1[4]) / nseg * i
                interm_j5 = n1[5] + (n2[5]-n1[5]) / nseg * i
                interm_j6 = n1[6] + (n2[6]-n1[6]) / nseg * i
                intermNode = [interm_j0, interm_j1, interm_j2, interm_j3, interm_j4, interm_j5, interm_j6]
                ########## move the robot to that configuration and then check ##########
                self.setRobotToConfig(intermNode, robot, armType)
                ### check if there is collision
                isConfigValid, FLAG = self.checkConfig_AllCollisions(robot, workspace, armType)
                if not isConfigValid:
                    isEdgeValid = False
                    return isEdgeValid, FLAG
                ##########################################################################
            ### Reach here because the edge pass the collision check
            isEdgeValid = True
            FLAG = 0
            return isEdgeValid, FLAG

        if armType == "Left_torso" or armType == "Right_torso":
            nseg = int(max(
                abs(n1[0]-n2[0]), abs(n1[1]-n2[1]), abs(n1[2]-n2[2]), abs(n1[3]-n2[3]),
                abs(n1[4]-n2[4]), abs(n1[5]-n2[5]), abs(n1[6]-n2[6]), abs(n1[7]-n2[7])) / min_degree)
            if nseg == 0: nseg += 1
            # print("nseg: " + str(nseg))            

            for i in range(0, nseg+1):
                # print(i)
                interm_j0 = n1[0] + (n2[0]-n1[0]) / nseg * i
                interm_j1 = n1[1] + (n2[1]-n1[1]) / nseg * i
                interm_j2 = n1[2] + (n2[2]-n1[2]) / nseg * i
                interm_j3 = n1[3] + (n2[3]-n1[3]) / nseg * i
                interm_j4 = n1[4] + (n2[4]-n1[4]) / nseg * i
                interm_j5 = n1[5] + (n2[5]-n1[5]) / nseg * i
                interm_j6 = n1[6] + (n2[6]-n1[6]) / nseg * i
                interm_j7 = n1[7] + (n2[7]-n1[7]) / nseg * i
                intermNode = [interm_j0, interm_j1, interm_j2, interm_j3, interm_j4, interm_j5, interm_j6, interm_j7]
                ########## move the robot to that configuration and then check ##########
                self.setRobotToConfig(intermNode, robot, armType)
                ### check if there is collision
                isConfigValid, FLAG = self.checkConfig_AllCollisions(robot, workspace, armType)
                if not isConfigValid:
                    isEdgeValid = False
                    return isEdgeValid, FLAG
                ##########################################################################
            ### Reach here because the edge pass the collision check
            isEdgeValid = True
            FLAG = 0
            return isEdgeValid, FLAG

    def checkEdgeValidity_knownGEO(self, n1, n2, robot, workspace, armType):
        ### Input: n1, n2: node (a list of 7 or 8 joint values)
        ### Output: bool value indicates whether the transition from n1 to n2 is valid
        ###         and a flag which indicates the failure explanation

        # nseg = 5
        # min_degree = math.pi / 90
        min_degree = math.pi / 180 * 5 ### make it sparsely interpolated to speed up collision check

        if armType == "Left" or armType == "Right":
            nseg = int(max(
                abs(n1[0]-n2[0]), abs(n1[1]-n2[1]), abs(n1[2]-n2[2]), abs(n1[3]-n2[3]),
                abs(n1[4]-n2[4]), abs(n1[5]-n2[5]), abs(n1[6]-n2[6])) / min_degree)
            if nseg == 0: nseg += 1
            # print("nseg: " + str(nseg))

            for i in range(0, nseg+1):
                interm_j0 = n1[0] + (n2[0]-n1[0]) / nseg * i
                interm_j1 = n1[1] + (n2[1]-n1[1]) / nseg * i
                interm_j2 = n1[2] + (n2[2]-n1[2]) / nseg * i
                interm_j3 = n1[3] + (n2[3]-n1[3]) / nseg * i
                interm_j4 = n1[4] + (n2[4]-n1[4]) / nseg * i
                interm_j5 = n1[5] + (n2[5]-n1[5]) / nseg * i
                interm_j6 = n1[6] + (n2[6]-n1[6]) / nseg * i
                intermNode = [interm_j0, interm_j1, interm_j2, interm_j3, interm_j4, interm_j5, interm_j6]
                ########## move the robot to that configuration and then check ##########
                self.setRobotToConfig(intermNode, robot, armType)
                ### check if there is collision
                isConfigValid, FLAG = self.checkConfig_CollisionWithRobotAndKnownGEO(robot, workspace)
                if not isConfigValid:
                    isEdgeValid = False
                    return isEdgeValid, FLAG
                ##########################################################################
            ### Reach here because the edge pass the collision check
            isEdgeValid = True
            FLAG = 0
            return isEdgeValid, FLAG

        if armType == "Left_torso" or armType == "Right_torso":
            nseg = int(max(
                abs(n1[0]-n2[0]), abs(n1[1]-n2[1]), abs(n1[2]-n2[2]), abs(n1[3]-n2[3]),
                abs(n1[4]-n2[4]), abs(n1[5]-n2[5]), abs(n1[6]-n2[6]), abs(n1[7]-n2[7])) / min_degree)
            if nseg == 0: nseg += 1

            for i in range(0, nseg+1):
                interm_j0 = n1[0] + (n2[0]-n1[0]) / nseg * i
                interm_j1 = n1[1] + (n2[1]-n1[1]) / nseg * i
                interm_j2 = n1[2] + (n2[2]-n1[2]) / nseg * i
                interm_j3 = n1[3] + (n2[3]-n1[3]) / nseg * i
                interm_j4 = n1[4] + (n2[4]-n1[4]) / nseg * i
                interm_j5 = n1[5] + (n2[5]-n1[5]) / nseg * i
                interm_j6 = n1[6] + (n2[6]-n1[6]) / nseg * i
                interm_j7 = n1[7] + (n2[7]-n1[7]) / nseg * i
                intermNode = [interm_j0, interm_j1, interm_j2, interm_j3, interm_j4, interm_j5, interm_j6, interm_j7]
                ########## move the robot to that configuration and then check ##########
                self.setRobotToConfig(intermNode, robot, armType)
                ### check if there is collision
                isConfigValid, FLAG = self.checkConfig_CollisionWithRobotAndKnownGEO(robot, workspace)
                if not isConfigValid:
                    isEdgeValid = False
                    return isEdgeValid, FLAG
                ##########################################################################
            ### Reach here because the edge pass the collision check
            isEdgeValid = True
            FLAG = 0
            return isEdgeValid, FLAG


    def generateTrajectory_DirectConfigPath(self, n1, n2, robot, armType, workspace):
        ### This function generates a trajectory based on two configs (which has been proved to be valid transition)
        ### Input: n1, n2: node (a list of 7 or 8 joint values)
        ### output: an edge trajectory (config_edge_traj) which includes the endtail but not the head
        ###         format: a list of list(7 or 8 joint values)

        config_edge_traj = []
        # nseg = 5
        min_degree = math.pi / 180 * 1 ### want more waypoint to move more naturally

        if armType == "Left" or armType == "Right":
            nseg = int(max(
                abs(n1[0]-n2[0]), abs(n1[1]-n2[1]), abs(n1[2]-n2[2]), abs(n1[3]-n2[3]),
                abs(n1[4]-n2[4]), abs(n1[5]-n2[5]), abs(n1[6]-n2[6])) / min_degree)
            if nseg == 0: nseg += 1
            # print("nseg: " + str(nseg))
            ### we don't include the head (i=0)
            for i in range(1, nseg+1):
                interm_j0 = n1[0] + (n2[0]-n1[0]) / nseg * i
                interm_j1 = n1[1] + (n2[1]-n1[1]) / nseg * i
                interm_j2 = n1[2] + (n2[2]-n1[2]) / nseg * i
                interm_j3 = n1[3] + (n2[3]-n1[3]) / nseg * i
                interm_j4 = n1[4] + (n2[4]-n1[4]) / nseg * i
                interm_j5 = n1[5] + (n2[5]-n1[5]) / nseg * i
                interm_j6 = n1[6] + (n2[6]-n1[6]) / nseg * i
                intermNode = [interm_j0, interm_j1, interm_j2, interm_j3, interm_j4, interm_j5, interm_j6]
                robot.setSingleArmToConfig(intermNode, armType)
                ########## temporarily add here for visualization ##########
                if (self.isObjectInLeftHand and (armType == "Left" or armType == "Left_torso")) or \
                            (self.isObjectInRightHand and (armType == "Right" or armType == "Right_torso")):
                    self.updateMeshBasedonLocalPose(robot, workspace, armType)
                ############################################################
                # time.sleep(0.1)
                config_edge_traj.append(intermNode)
        
        if armType == "Left_torso" or armType == "Right_torso":
            nseg = int(max(
                abs(n1[0]-n2[0]), abs(n1[1]-n2[1]), abs(n1[2]-n2[2]), abs(n1[3]-n2[3]),
                abs(n1[4]-n2[4]), abs(n1[5]-n2[5]), abs(n1[6]-n2[6]), abs(n1[7]-n2[7])) / min_degree)
            if nseg == 0: nseg += 1
            # print("nseg: " + str(nseg))            
            for i in range(1, nseg+1):
                interm_j0 = n1[0] + (n2[0]-n1[0]) / nseg * i
                interm_j1 = n1[1] + (n2[1]-n1[1]) / nseg * i
                interm_j2 = n1[2] + (n2[2]-n1[2]) / nseg * i
                interm_j3 = n1[3] + (n2[3]-n1[3]) / nseg * i
                interm_j4 = n1[4] + (n2[4]-n1[4]) / nseg * i
                interm_j5 = n1[5] + (n2[5]-n1[5]) / nseg * i
                interm_j6 = n1[6] + (n2[6]-n1[6]) / nseg * i
                interm_j7 = n1[7] + (n2[7]-n1[7]) / nseg * i
                intermNode = [interm_j0, interm_j1, interm_j2, interm_j3, interm_j4, interm_j5, interm_j6, interm_j7]
                robot.setSingleArmToConfig_torso(intermNode[1:8], intermNode[0], armType)
                ########## temporarily add here for visualization ##########
                if (self.isObjectInLeftHand and (armType == "Left" or armType == "Left_torso")) or \
                            (self.isObjectInRightHand and (armType == "Right" or armType == "Right_torso")):
                    self.updateMeshBasedonLocalPose(robot, workspace, armType)
                ############################################################
                # time.sleep(0.1)
                config_edge_traj.append(intermNode)

        return config_edge_traj
    
    def AstarPathFinding(self, initialConfig, targetConfig,
                start_neighbors_idx, start_neighbors_cost, 
                goal_neighbors_idx, goal_neighbors_cost,
                robot, workspace, armType, isLabeledRoadmapUsed):
        if isLabeledRoadmapUsed:
            traj = self.AstarPathFinding_labeledVersion(initialConfig, targetConfig,
                start_neighbors_idx, start_neighbors_cost, 
                goal_neighbors_idx, goal_neighbors_cost,
                robot, workspace, armType)
        else:
            traj = self.AstarPathFinding_nonLabeledVersion(initialConfig, targetConfig,
                start_neighbors_idx, start_neighbors_cost, 
                goal_neighbors_idx, goal_neighbors_cost,
                robot, workspace, armType)
        return traj

    def serviceCall_astarPathFinding_nonLabeledVersion(self, 
            violated_edges, initialConfig, targetConfig, 
            start_neighbors_idx, goal_neighbors_idx, start_neighbors_cost, goal_neighbors_cost,
            robot, workspace, armType):
        ### violated_edges: [Edge(), Edge(), ...]
        ### prepare the AstarPathFindingNonLabeledRequest
        rospy.wait_for_service("astar_path_finding_nonlabeled")
        request = AstarPathFindingNonLabeledRequest()
        request.query_idx = self.query_idx
        request.start_idx = self.nsamples
        request.goal_idx = self.nsamples + 1
        request.start_config = initialConfig
        request.goal_config = targetConfig
        request.violated_edges = violated_edges
        request.armType = armType
        request.start_neighbors_idx = start_neighbors_idx
        request.goal_neighbors_idx = goal_neighbors_idx
        request.start_neighbors_cost = start_neighbors_cost
        request.goal_neighbors_cost = goal_neighbors_cost

        try:
            astarSearchNonLabeled = rospy.ServiceProxy("astar_path_finding_nonlabeled", AstarPathFindingNonLabeled)
            response = astarSearchNonLabeled(request.query_idx, 
                request.start_idx, request.goal_idx,
                request.start_config, request.goal_config,
                request.start_neighbors_idx, request.goal_neighbors_idx,
                request.start_neighbors_cost, request.goal_neighbors_cost,
                request.violated_edges, request.armType)
            return response.searchSuccess, list(response.path)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


    def AstarPathFinding_nonLabeledVersion(self, initialConfig, targetConfig,
                start_neighbors_idx, start_neighbors_cost, 
                goal_neighbors_idx, goal_neighbors_cost,
                robot, workspace, armType):
        ### Input: initialConfig, targetConfig [q1, q2, ..., q7]
        ###        neighbors_idx as well as neighbors_cost should not be empty when entering in this function
        ### Output: traj (format: [joint_state1, joint_state2, ...])
        ###         and joint_state is a list of joint values

        result_traj = [] ### the output we want to construct
        isPathValid = False
        print("current planning query: ", self.query_idx)
        violated_edges = [] ### initially there are no violated edges

        counter = 0
        while (isPathValid == False):
            counter += 1
            ### trigger new call within the same query idx
            # start_time = time.time()
            searchSuccess, path =  self.serviceCall_astarPathFinding_nonLabeledVersion(
                    violated_edges, initialConfig, targetConfig, 
                    start_neighbors_idx, goal_neighbors_idx,
                    start_neighbors_cost, goal_neighbors_cost,
                    robot, workspace, armType)
            # print("Time for service call for astarPathFinding_nonLabeledVersion: {}".format(time.time() - start_time))
            if searchSuccess == False:
                print("the plan fails at the " + str(counter) + "th trial...")
                ### the plan fails, could not find a solution
                self.query_idx += 1
                return result_traj ### an empty trajectory
            ### otherwise, we need collision check and smoothing (len(path) >= 3)
            # start_time = time.time()
            smoothed_path, isPathValid, violated_edges = self.smoothPath(
                    path, initialConfig, targetConfig, robot, workspace, armType)
            # print("Time for smooth the path: {}".format(time.time() - start_time))            

        ### congrats, the path is valid and finally smoothed, let's generate trajectory
        print("smoothed path: ", smoothed_path)
        print("\n")
        ### directly generate trajectory based on the new path
        for i in range(0, len(smoothed_path)-1):
            if i == 0:
                config1 = initialConfig
            else:
                config1 = self.nodes[armType][smoothed_path[i]]
            if i == (len(smoothed_path)-2):
                config2 = targetConfig
            else:
                config2 = self.nodes[armType][smoothed_path[i+1]]
            ### get edge trajectory
            config_edge_traj = self.generateTrajectory_DirectConfigPath(config1, config2, robot, armType, workspace)
            # result_traj.append(config_edge_traj)
            result_traj += config_edge_traj

        ### before you claim the victory of this query, increment the planning query
        ### so as to tell people this query is over, next time is a new query
        self.query_idx += 1

        return result_traj


    def smoothPath(self, path, initialConfig, targetConfig, robot, workspace, armType):
        ### This function tries to smooth the given path
        ### output: a smooth path [a list of indexes] and whether the path is valid or not
        smoothed_path = []
        violated_edges = []
        start_idx = 0
        startNode_idx = path[start_idx] ### start
        smoothed_path.append(startNode_idx)
        curr_idx = start_idx + 1
        while (curr_idx < len(path)):
            currNode_idx = path[curr_idx]
            ### check edge validity between start_idx and curr_idx
            if start_idx == 0:
                config1 = initialConfig
            else:
                config1 = self.nodes[armType][startNode_idx]
            if curr_idx == (len(path)-1):
                config2 = targetConfig
            else:
                config2 = self.nodes[armType][currNode_idx]

            if (start_idx == 0 and curr_idx == 1) or (start_idx == len(path)-2 and curr_idx == len(path)-1):
                ### no need to check the edge validity between neighboring nodes
                ### which has either start or goal
                isEdgeValid = True
            else:
                ### check the edge
                isEdgeValid, FLAG = self.checkEdgeValidity_AllCollisions(
                                config1, config2, robot, workspace, armType)
            if isEdgeValid:
                validFromStart_idx = curr_idx
                validNodeFromStart_idx = currNode_idx
                ### move on to the next node
                curr_idx += 1
                continue
            else:
                if (curr_idx - start_idx == 1):
                    print("Edge invalid, we need call A* again with the change of edge information")
                    print("FLAG: " + str(FLAG))
                    print(str(startNode_idx) + "," + str(currNode_idx) + "\n")
                    edge = Edge()
                    edge.idx1 = startNode_idx
                    edge.idx2 = currNode_idx
                    violated_edges.append(edge)
                    smoothed_path = [] ### turn it back to empty path
                    return smoothed_path, False, violated_edges
                ### the edge is not valid
                ### add validNodeFromStart_idx to the smoothed_path
                smoothed_path.append(validNodeFromStart_idx)
                ### set validNodeFromStart_idx as the new start
                ### and the curr_idx does not change
                start_idx = validFromStart_idx
                startNode_idx = validNodeFromStart_idx
        smoothed_path.append(validNodeFromStart_idx)

        return smoothed_path, True, violated_edges


    def connectToNeighbors(self, config, robot, workspace, armType):
        ### This function makes connections 
        ### between the specified config to neighboring nodes in the roadmap
        ### It returns (1) whether the config can be connected to any neighbors (isConfigValid)
        ###            (2) the neighbors it connects with
        neighbors_idx = []
        neighbors_cost = []
        connectSuccess = False

        dist = [utils.calculateNorm2(config, neighborConfig) for neighborConfig in self.nodes[armType]]
        neighborIndex, neighborDist = zip(*sorted(enumerate(dist), key=itemgetter(1)))
        neighborIndex = list(neighborIndex)
        neighborDist = list(neighborDist)

        # max_neighbors = self.num_neighbors
        max_neighbors = 5
        max_candiates_to_consider = self.num_neighbors

        ####### now connect potential neighbors for the specified config #######
        neighbors_connected = 0
        for j in range(max_candiates_to_consider):
            ### first check if the query node has already connected to enough neighbors
            if neighbors_connected >= max_neighbors:
                break
            ### otherwise, find the neighbor
            neighbor = self.nodes[armType][neighborIndex[j]]
            ### check the edge validity
            isEdgeValid, FLAG = self.checkEdgeValidity_AllCollisions(
                                config, neighbor, robot, workspace, armType)
            if isEdgeValid:
                neighbors_idx.append(neighborIndex[j])
                neighbors_cost.append(neighborDist[j])
                neighbors_connected += 1
            else:
                pass
                # print("Neighbor connection not success, raise FLAG: " + str(FLAG))
        
        ### check if the number of neighboring connections is zero
        print("Number of neighbors for current node: " + str(neighbors_connected))
        if neighbors_connected != 0: connectSuccess = True
        return connectSuccess, neighbors_idx, neighbors_cost


    def findNeighborsForStartAndGoal(self,
                    initialConfig, targetConfig, robot, workspace, armType): 
        ### return four things
        ### (1) start_neighbors_idx (a list of integer)
        ### (2) goal_neighbors_idx (a list of integer)
        ### (3) start_neighbors_cost (a list of float)
        ### (4) goal_neighbors_cost (a list of float)
        start_neighbors_idx = []
        goal_neighbors_idx = []
        start_neighbors_cost = []
        goal_neighbors_cost = []

        dist_to_start = [
            utils.calculateNorm2(initialConfig, neighborConfig) for neighborConfig in self.nodes[armType]]

        neighborIndex_to_start, neighborDist_to_start = zip(
                                    *sorted(enumerate(dist_to_start), key=itemgetter(1)))        
        neighborIndex_to_start = list(neighborIndex_to_start)
        neighborDist_to_start = list(neighborDist_to_start)
        dist_to_goal = [
            utils.calculateNorm2(targetConfig, neighborConfig) for neighborConfig in self.nodes[armType]]
        neighborIndex_to_goal, neighborDist_to_goal = zip(
                                    *sorted(enumerate(dist_to_goal), key=itemgetter(1)))
        neighborIndex_to_goal = list(neighborIndex_to_goal) 
        neighborDist_to_goal = list(neighborDist_to_goal)        

        # max_neighbors = self.num_neighbors
        max_neighbors = 5
        max_candiates_to_consider = self.num_neighbors

        ####### now connect potential neighbors for the start and the goal #######
        ### for start
        # print("for start")
        neighbors_connected_start = 0
        for j in range(max_candiates_to_consider):
            ### first check if the query node has already connected to enough neighbors
            if neighbors_connected_start >= max_neighbors:
                break
            ### otherwise, find the neighbor
            neighbor = self.nodes[armType][neighborIndex_to_start[j]]
            ### check the edge validity
            isEdgeValid, FLAG = self.checkEdgeValidity_AllCollisions(
                initialConfig, neighbor, robot, workspace, armType)
            # print("FLAG: " + str(FLAG))
            if isEdgeValid:
                start_neighbors_idx.append(neighborIndex_to_start[j])
                start_neighbors_cost.append(neighborDist_to_start[j])
                neighbors_connected_start += 1
        print("Number of neighbors for start node: " + str(neighbors_connected_start))

        ### for goal
        # print("for goal")
        neighbors_connected_goal = 0
        for j in range(max_candiates_to_consider):
            ### first check if the query node has already connected to enough neighbors
            if neighbors_connected_goal >= max_neighbors:
                break
            ### otherwise, find the neighbor
            neighbor = self.nodes[armType][neighborIndex_to_goal[j]]
            ### check the edge validity
            isEdgeValid, FLAG = self.checkEdgeValidity_AllCollisions(
                            targetConfig, neighbor, robot, workspace, armType)
            # print("FLAG: " + str(FLAG))
            if isEdgeValid:
                goal_neighbors_idx.append(neighborIndex_to_goal[j])
                goal_neighbors_cost.append(neighborDist_to_goal[j])
                neighbors_connected_goal += 1
        print("Number of neighbors for goal node: " + str(neighbors_connected_goal))

        return start_neighbors_idx, goal_neighbors_idx, start_neighbors_cost, goal_neighbors_cost


    def attachObject(self, object_idx, workspace, robot, armType):
        if armType == "Left" or armType == "Left_torso":
            self.isObjectInLeftHand = True
            self.objectInLeftHand = workspace.object_geometries[object_idx].geo ### mesh
            self.objectInLeftHand_idx = object_idx
        if armType == "Right" or armType == "Right_torso":
            self.isObjectInRightHand = True
            self.objectInRightHand = workspace.object_geometries[object_idx].geo ### mesh
            self.objectInRightHand_idx = object_idx
        ### get the object pose relative to the frame of the end effector 
        ### since once the object is attached to the end effector, 
        ### it will remain constant (may add some noice)
        if armType == "Left" or armType == "Left_torso":
            curr_ee_pose = robot.left_ee_pose
            ls = p.getBasePositionAndOrientation(
                    self.objectInLeftHand, physicsClientId=self.planningServer)
            curr_object_global_pose = [list(ls[0]), list(ls[1])]
        if armType == "Right" or armType == "Right_torso":
            curr_ee_pose = robot.right_ee_pose
            ls = p.getBasePositionAndOrientation(
                    self.objectInRightHand, physicsClientId=self.planningServer)
            curr_object_global_pose = [list(ls[0]), list(ls[1])]
        
        inverse_ee_global = p.invertTransform(curr_ee_pose[0], curr_ee_pose[1])
        temp_localPose = p.multiplyTransforms(
            list(inverse_ee_global[0]), list(inverse_ee_global[1]),
            curr_object_global_pose[0], curr_object_global_pose[1])
        temp_localPose = [list(temp_localPose[0]), list(temp_localPose[1])]

        if armType == "Left" or armType == "Left_torso":
            self.leftLocalPose = temp_localPose
        if armType == "Right" or armType == "Right_torso":
            self.rightLocalPose = temp_localPose


    def detachObject(self, workspace, robot, armType):
        if armType == "Left" or armType == "Left_torso":
            self.isObjectInLeftHand = False
            self.objectInLeftHand = None
            self.objectInLeftHand_idx = -1
        if armType == "Right" or armType == "Right_torso":
            self.isObjectInRightHand = False
            self.objectInRightHand = None
            self.objectInRightHand_idx = -1

    
    def addObjectInHand_labeledRoadmap(self, workspace, robot, armType):
        if armType == "Right" or armType == "Right_torso":
            self.isObjectInRightHand = True
            ### specify the pre-computed local pose
            self.rightLocalPose = [[0.09468472003936768, 0.0007766783237457275, -0.0014880895614624023], \
                        [0.0924038216471672, -0.700919508934021, -0.09246741980314255, 0.7011584639549255]]
            ####### use that local pose to generate the sample_object #######
            ### get the global pose of the object
            sample_object_global_pose = self.getObjectGlobalPose(self.rightLocalPose, robot.right_ee_pose)
            sample_object_c = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=workspace.cylinder_radius, 
                        height=workspace.cylinder_height, physicsClientId=self.planningServer)
            sample_object_v = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                radius=workspace.cylinder_radius, length=workspace.cylinder_height, 
                rgbaColor=[1.0, 0.0, 0.0, 0.8], physicsClientId=self.planningServer)
            sample_objectM = p.createMultiBody(
                baseCollisionShapeIndex=sample_object_c, baseVisualShapeIndex=sample_object_v,
                basePosition=sample_object_global_pose[0], baseOrientation=sample_object_global_pose[1], 
                physicsClientId=self.planningServer)
            self.objectInRightHand = sample_objectM
            ### as this object is temporarily created. Does not belong to object_geometries
            self.objectInRightHand_idx = -1 
    
    def deleteObjectInHand_labeledRoadmap(self, armType):
        if armType == "Right" or armType == "Right_torso":
            self.isObjectInRightHand = False
            p.removeBody(self.objectInRightHand)
            self.objectInRightHand = None
            self.objectInRightHand_idx = -1

    def samplesConnect_labeledRoadmap(self, robot, workspace, armType):
        connectivity = np.zeros((self.nsamples, self.nsamples))
        tree = spatial.KDTree(self.nodes[armType]) ### use KD tree to arrange neighbors assignment
        connectionsFile = self.roadmapFolder + "/connections_" + str(armType) + ".txt"
        f_connection = open(connectionsFile, "w")
        ### for each node
        for node_idx in range(len(self.nodes[armType])):
            queryNode = self.nodes[armType][node_idx]
            knn = tree.query(queryNode, k=self.num_neighbors, p=2)

            neighbors_connected = 0
            ### for each potential neighbor
            for j in range(len(knn[1])):
                ### first check if this query node has already connected to enough neighbors
                if neighbors_connected >= self.num_neighbors:
                    break
                if knn[1][j] == node_idx:
                    ### if the neighbor is the query node itself
                    continue
                if connectivity[node_idx][knn[1][j]] == 1:
                    ### the connectivity has been checked before
                    neighbors_connected += 1
                    continue
                ### Otherwise, check the edge validity between the query node and the current neighbor
                neighbor = self.nodes[armType][knn[1][j]]
                isEdgeValid, FLAG, objectCollided_total, inhandValidity_eventual, objectCollided_inHand_total = \
                    self.checkEdgeValidity_AllCollisions_labeledRoadmap(queryNode, neighbor, robot, workspace, armType)
                if not isEdgeValid:
                    ### FLAG must be 2 (robot self-collision) or 3 (robot-knownGEO)
                    ### the edge is not valid, it incurs robot self-collision or robot-knownGEO
                    continue
                else:
                    # print("PRINT BEFORE WRITE IN THE FILE")
                    # print(str(node_idx) + " " + str(knn[1][j]))
                    # print(objectCollided_total)
                    # print(inhandValidity_eventual)
                    # print(objectCollided_inHand_total)
                    # input("============================\n")
                    ### the edge is at least valid
                    ### write edge information: node1, node2, cost, objectCollided, inHandValidity, objectCollided_inHand
                    ### (1) node1, node2, cost
                    f_connection.write(str(node_idx) + " " + str(knn[1][j]) + " " + str(knn[0][j]))
                    ### (2) objectCollided_total
                    if objectCollided_total != []:
                        for obj_idx in objectCollided_total:
                            f_connection.write(" " + str(obj_idx))
                    ### (3) inhandValidity_eventual
                    if inhandValidity_eventual == True:
                        f_connection.write(" " + str(-1))
                    else:
                        f_connection.write(" " + str(-2))
                    ### (4) objectCollided_inHand_total
                    if objectCollided_inHand_total != []:
                        for obj_idx in objectCollided_inHand_total:
                            f_connection.write(" " + str(obj_idx))
                    f_connection.write("\n")
                    connectivity[node_idx][knn[1][j]] = 1
                    connectivity[knn[1][j]][node_idx] = 1
                    neighbors_connected += 1
            print("Number of neighbors for current node " + str(node_idx) + ": " + str(neighbors_connected))
        f_connection.close()

    def checkEdgeValidity_AllCollisions_labeledRoadmap(self, n1, n2, robot, workspace, armType):
        ### Input: n1, n2: node (a list of 7 or 8 joint values)
        ### Output: isEdgeValid (bool), FLAG (2,3,4,5,6,7), 
        ### objectCollided (a list of obj_idx(labels)), inHandValidity (bool)
        ### objectCollided_inHand (a list of obj_idx(labels))
        
        # nseg = 5
        # min_degree = math.pi / 90
        min_degree = math.pi / 180 * 5 ### make it sparsely interpolated to speed up collision check

        ### initialization
        objectCollided_total = set()
        inhandValidity_eventual = True
        objectCollided_inHand_total = set()

        if armType == "Left" or armType == "Right":
            nseg = int(max(
                abs(n1[0]-n2[0]), abs(n1[1]-n2[1]), abs(n1[2]-n2[2]), abs(n1[3]-n2[3]),
                abs(n1[4]-n2[4]), abs(n1[5]-n2[5]), abs(n1[6]-n2[6])) / min_degree)
            if nseg == 0: nseg += 1
            # print("nseg: " + str(nseg))

            for i in range(0, nseg+1):
                interm_j0 = n1[0] + (n2[0]-n1[0]) / nseg * i
                interm_j1 = n1[1] + (n2[1]-n1[1]) / nseg * i
                interm_j2 = n1[2] + (n2[2]-n1[2]) / nseg * i
                interm_j3 = n1[3] + (n2[3]-n1[3]) / nseg * i
                interm_j4 = n1[4] + (n2[4]-n1[4]) / nseg * i
                interm_j5 = n1[5] + (n2[5]-n1[5]) / nseg * i
                interm_j6 = n1[6] + (n2[6]-n1[6]) / nseg * i
                intermNode = [interm_j0, interm_j1, interm_j2, interm_j3, interm_j4, interm_j5, interm_j6]
                ########## move the robot to that configuration and then check ##########
                self.setRobotToConfig(intermNode, robot, armType)
                ### check if there is collision
                isConfigValid, FLAG, objectCollided, inHandValidity, objectCollided_inHand = \
                                self.checkConfig_AllCollisions_labeledRoadmap(robot, workspace, armType, inhandValidity_eventual)
                if not isConfigValid:
                    ### FLAG must be 2 (robot self-collision) or 3 (robot-knownGEO)
                    isEdgeValid = False
                    return isEdgeValid, FLAG, [], False, []
                else:
                    ### it's a valid config
                    objectCollided_total = set(list(objectCollided_total) + objectCollided)
                    objectCollided_inHand_total = set(list(objectCollided_inHand_total) + objectCollided_inHand)
                    if not inHandValidity: inhandValidity_eventual = False                    
                ##########################################################################
            ### Reach here because the edge pass the collision check
            isEdgeValid = True
            FLAG = 0
            return isEdgeValid, FLAG, objectCollided_total, inhandValidity_eventual, objectCollided_inHand_total

        if armType == "Left_torso" or armType == "Right_torso":
            nseg = int(max(
                abs(n1[0]-n2[0]), abs(n1[1]-n2[1]), abs(n1[2]-n2[2]), abs(n1[3]-n2[3]),
                abs(n1[4]-n2[4]), abs(n1[5]-n2[5]), abs(n1[6]-n2[6]), abs(n1[7]-n2[7])) / min_degree)
            if nseg == 0: nseg += 1
            # print("nseg: " + str(nseg))            

            for i in range(0, nseg+1):
                # print(i)
                interm_j0 = n1[0] + (n2[0]-n1[0]) / nseg * i
                interm_j1 = n1[1] + (n2[1]-n1[1]) / nseg * i
                interm_j2 = n1[2] + (n2[2]-n1[2]) / nseg * i
                interm_j3 = n1[3] + (n2[3]-n1[3]) / nseg * i
                interm_j4 = n1[4] + (n2[4]-n1[4]) / nseg * i
                interm_j5 = n1[5] + (n2[5]-n1[5]) / nseg * i
                interm_j6 = n1[6] + (n2[6]-n1[6]) / nseg * i
                interm_j7 = n1[7] + (n2[7]-n1[7]) / nseg * i
                intermNode = [interm_j0, interm_j1, interm_j2, interm_j3, interm_j4, interm_j5, interm_j6, interm_j7]
                ########## move the robot to that configuration and then check ##########
                self.setRobotToConfig(intermNode, robot, armType)
                ### check if there is collision
                isConfigValid, FLAG, objectCollided, inHandValidity, objectCollided_inHand = \
                                self.checkConfig_AllCollisions_labeledRoadmap(robot, workspace, armType, inhandValidity_eventual)
                # print("\n")
                # print("isConfigValid: " + str(isConfigValid))
                # print("FLAG: " + str(FLAG))
                # print("objectCollided: " + str(objectCollided))
                # print("inHandValidity: " + str(inHandValidity))
                # print("objectCollided_inHand: " + str(objectCollided_inHand))
                if not isConfigValid:
                    ### FLAG must be 2 (robot self-collision) or 3 (robot-knownGEO)
                    # input("ah oh, not valid, move on to the next edge")
                    isEdgeValid = False
                    return isEdgeValid, FLAG, [], False, []
                else:
                    ### it's a valid config
                    # print("it's a valid waypoint")
                    objectCollided_total = set(list(objectCollided_total) + objectCollided)
                    objectCollided_inHand_total = set(list(objectCollided_inHand_total) + objectCollided_inHand)
                    if not inHandValidity: inhandValidity_eventual = False
                    # print("objectCollided_total: " + str(objectCollided_total))
                    # print("inHandValidity_eventual: " + str(inhandValidity_eventual))
                    # print("objectCollided_inHand_total: " + str(objectCollided_inHand_total))
                    # print("move on to the next waypoint")
                ##########################################################################
            ### Reach here because the edge pass the collision check
            isEdgeValid = True
            FLAG = 0
            # print("congrats on finishing this valid edge!!")
            # print("objectCollided_total: " + str(objectCollided_total))
            # print("inHandValidity_eventual: " + str(inhandValidity_eventual))
            # print("objectCollided_inHand_total: " + str(objectCollided_inHand_total))
            # print("move on to the next edge")
            return isEdgeValid, FLAG, objectCollided_total, inhandValidity_eventual, objectCollided_inHand_total

    def checkConfig_AllCollisions_labeledRoadmap(self, robot, workspace, armType, inHandValidity):
        ######## before calling this function, don't forget to call API: setRobotToConfig ########
        ### This function checks all collisions
        ### and will label collisions for (1) robot-objects (2) moving object-static objects
        isConfigMeetHardRequirement = True
        objectCollided = []
        objectCollided_inHand = []

        ############  first check potential collision between robot and knownGEO & robot ############
        ### FLAG: (2,3) or 0
        isConfigValid, FLAG = self.checkConfig_CollisionWithRobotAndKnownGEO(robot, workspace)
        if isConfigValid == False: 
            ### Early return: this config is not valid 
            ### as the flag is 2 (robt self-collision) or 3 (robot-knownGEO)
            isConfigMeetHardRequirement = False
            return isConfigMeetHardRequirement, FLAG, objectCollided, inHandValidity, objectCollided_inHand
        #############################################################################################
        
        ################ then check potential collision with objects not in hand ####################
        ### first get all the objects which are not in hand (here objects are really candidates)
        static_object_geometries = { obj_info.position_idx : obj_info.geo \
            for obj_info in workspace.candidate_geometries.values() \
            if (obj_info.position_idx != self.objectInLeftHand_idx) and \
                (obj_info.position_idx != self.objectInRightHand_idx) }
        ### FLAG: 4
        isConfigValid, FLAG, objectCollided = \
            self.checkConfig_CollisionBetweenRobotAndStaticObjects_labeled(robot, static_object_geometries)
        #############################################################################################

        ############ then check potential collisions for in-hand manipulation #######################
        ### (i) add an object in the specified hand
        self.addObjectInHand_labeledRoadmap(workspace, robot, armType)
        if armType == "Left" or armType == "Left_torso":
            manipulation_objectGEO = self.objectInLeftHand
        if armType == "Right" or armType == "Right_torso":
            manipulation_objectGEO = self.objectInRightHand
        ### only check inHandValidity with robot and knownGEO if currently inHandValidity is still true
        if inHandValidity:
            ### (ii) check the potential collision between the robot and the moving object
            ### FLAG: 5
            isConfigValid, FLAG = self.checkConfig_CollisionBetweenRobotAndMovingObject(
                                                        robot, manipulation_objectGEO, armType)
            if isConfigValid:
                ### (iii) check the potential collision between the moving object and known geometries
                ### FLAG: 6
                isConfigValid, FLAG = self.checkConfig_CollisionMovingObjectAndKnownGEO(
                                                                manipulation_objectGEO, workspace)
            inHandValidity = True if isConfigValid else False
        ### (iv) check the potential collision between the moving object and other static objects
        ### FLAG: 7
        isConfigValid, FLAG, objectCollided_inHand = \
            self.checkConfig_CollisionMovingObjectAndStaticObjects_labeled(
                                            manipulation_objectGEO, static_object_geometries)
        #############################################################################################
        ### reach here since you finish all the check
        ### (v) delete an object out of the specified hand
        self.deleteObjectInHand_labeledRoadmap(armType)
        return isConfigMeetHardRequirement, FLAG, objectCollided, inHandValidity, objectCollided_inHand
        
    
    def checkConfig_CollisionBetweenRobotAndStaticObjects_labeled(self, robot, object_geometries):
        ######## before calling this function, don't forget to call API: setRobotToConfig ########
        ### check if there is collision between the robot and the static objects (objects in hand excluded)
        ### object_geometries is a dictionary {obj_idx : obj_geo}
        isCollision, objectCollided = self.collisionAgent_p.collisionCheck_robot_staticObjectGEOs_labeled(
                                                                        robot.motomanGEO, object_geometries)
        if isCollision == True:
            # print("collision between robot and the static objects")
            ### raise the flag
            isConfigValid = False
            FLAG = 4
            return isConfigValid, FLAG, objectCollided
        else:
            ### if you reach here, the configuration passes current collision check
            ### objectCollided should be an empty list [ ]
            isConfigValid = True
            FLAG = 0
            return isConfigValid, FLAG, objectCollided

    def checkConfig_CollisionMovingObjectAndStaticObjects_labeled(self, movingObjectGEO, object_geometries):
        ######## before calling this function, don't forget to call API: setRobotToConfig ########
        ### check if there is collision between the moving object and the static objects
        ### object_geometries is a dictionary {obj_idx : obj_geo}
        isCollision, objectCollided_inHand = self.collisionAgent_p.collisionCheck_object_objectGEO_labeled(
                                                                            movingObjectGEO, object_geometries)
        if isCollision == True:
            # print("moving object collide with static objects")
            ### raise the flag
            isConfigValid = False
            FLAG = 7
            return isConfigValid, FLAG, objectCollided_inHand
        else:
            ### if you reach here, the configuration passes current collision check
            ### objectCollided_inHand should be an empty list [ ]
            isConfigValid = True
            FLAG = 0
            return isConfigValid, FLAG, objectCollided_inHand


    def generateConfigBasedOnSamplePose(self, pose, rest_config, robot, workspace, armType):
        ### This function checks the validity of a pose by checking its corresponding config (IK)
        ### Input: pose: [[x,y,z],[x,y,z,w]]
        ###        armType: "Left" or "Right"
        ### Output: isIKValid (bool), singleArmConfig_IK (list)), FLAG
        if armType == "Left" or armType == "Left_torso":
            ee_idx = robot.left_ee_idx
            first_joint_index = 1
        if armType == "Right" or armType == "Right_torso":
            ee_idx = robot.right_ee_idx
            first_joint_index = 8
        
        if armType == "Left":
            rest_pose = robot.torsoCurrConfiguration + rest_config + robot.rightArmCurrConfiguration + robot.rightHandCurrConfiguration
        if armType == "Right":
            rest_pose = robot.torsoCurrConfiguration + robot.leftArmCurrConfiguration + rest_config + robot.rightHandCurrConfiguration
        if armType == "Left_torso":
            rest_pose = [rest_config[0]] + rest_config[1:8] + robot.rightArmCurrConfiguration + robot.rightHandCurrConfiguration
        if armType == "Right_torso":
            rest_pose = [rest_config[0]] + robot.leftArmCurrConfiguration + rest_config[1:8] + robot.rightHandCurrConfiguration

        ### we add rest pose in the IK solver to get as high-quality IK as possible
        config_IK = p.calculateInverseKinematics(bodyUniqueId=robot.motomanGEO,
                                endEffectorLinkIndex=ee_idx,
                                targetPosition=pose[0],
                                targetOrientation=pose[1],
                                lowerLimits=robot.ll, upperLimits=robot.ul, 
                                jointRanges=robot.jr, restPoses=rest_pose,
                                maxNumIterations=2000, residualThreshold=0.0000001,
                                physicsClientId=robot.server)
        if armType == "Left" or armType == "Right":
            singleArmConfig_IK = list(config_IK[first_joint_index:first_joint_index+7])
        if armType == "Left_torso" or armType == "Right_torso":
            singleArmConfig_IK = [config_IK[0]] + list(config_IK[first_joint_index:first_joint_index+7])
        ########## move the robot to that configuration and then check ##########
        self.setRobotToConfig(singleArmConfig_IK, robot, armType)
        isIKValid, FLAG = self.checkSamplePoseIK(pose, robot, workspace, armType)
        ##########################################################################
        
        trials = 0
        while (not isIKValid) and (trials < 1):
            ### try another IK (not specify rest pose)
            ### try another IK given curernt rest pose with random noise
            rest_pose = self.randomRestPose(robot, armType)
            config_IK = p.calculateInverseKinematics(bodyUniqueId=robot.motomanGEO,
                                    endEffectorLinkIndex=ee_idx,
                                    targetPosition=pose[0],
                                    targetOrientation=pose[1],
                                    lowerLimits=robot.ll, upperLimits=robot.ul, 
                                    jointRanges=robot.jr, restPoses=rest_pose,
                                    maxNumIterations=2000, residualThreshold=0.0000001,
                                    physicsClientId=robot.server)
            if armType == "Left" or armType == "Right":
                singleArmConfig_IK = list(config_IK[first_joint_index:first_joint_index+7])
            if armType == "Left_torso" or armType == "Right_torso":
                singleArmConfig_IK = [config_IK[0]] + list(config_IK[first_joint_index:first_joint_index+7])
            ########## move the robot to that configuration and then check ##########
            self.setRobotToConfig(singleArmConfig_IK, robot, armType)
            isIKValid, FLAG = self.checkSamplePoseIK(pose, robot, workspace, armType)
            if isIKValid: break
            ### otherwise
            trials += 1
            ##########################################################################
        ### you need to return both the statement whether the pose is valid
        ### and the valid configuration the pose corresponds to
        return isIKValid, FLAG, singleArmConfig_IK

    def checkSamplePoseIK(self, desired_ee_pose, robot, workspace, armType):
        ######## before calling this function, don't forget to call API: setRobotToConfig ########
        ### This function checks if an IK solution is valid in terms of
        ### (1) small error from the desired pose (position error + quaternion error)
        ### (2) no collision occurred for knownGEO + robot

        ### Input: desired_ee_pose: [[x,y,z], [x,y,z,w]]
        ###        armType: "Left" or "Right"
        ### Output: isConfigValid (bool) indicating whether the IK is valid
        ###         FLAG (int): indicating the validity/invalidity reason
        ###                     0: no any issues (pass all IK check)
        #                       1: reachability
        #                       2: robot self-collision
        #                       3: collision between robot and static geometry
        
        isConfigValid, FLAG = self.checkConfig_reachability(desired_ee_pose, robot, armType)
        if not isConfigValid: 
            return isConfigValid, FLAG
        ### Congrats! The IK successfully passed reachability checker. 
        ### Then check if there is collision (robot self-collision + robot-knownGEO)
        isConfigValid, FLAG = self.checkConfig_CollisionWithRobotAndKnownGEO(robot, workspace)
        return isConfigValid, FLAG

    def AstarPathFinding_labeledVersion(self, initialConfig, targetConfig,
                start_neighbors_idx, start_neighbors_cost, 
                goal_neighbors_idx, goal_neighbors_cost,
                robot, workspace, armType):
        ### Input: initialConfig, targetConfig [q1, q2, ..., q7]
        ###        neighbors_idx as well as neighbors_cost should not be empty when entering in this function
        ### Output: traj (format: [joint_state1, joint_state2, ...])
        ###         and joint_state is a list of joint values

        result_traj = [] ### the output we want to construct
        isPathValid = False
        print("current planning query: ", self.query_idx)
        violated_edges = [] ### initially there are no violated edges

        ### for path finding on a labeled roadmap, we need to know what labels
        ### are being occupied by the current arrangement
        occupied_labels = []
        if armType == "Right_torso":
            for obj_idx, obj_info in workspace.object_geometries.items():
                if (obj_idx != self.objectInRightHand_idx):
                    occupied_labels.append(obj_info.collision_position_idx)
            if self.objectInRightHand_idx == -1:
                isInHandManipulation = False
            else:
                isInHandManipulation = True

        counter = 0
        while (isPathValid == False):
            counter += 1
            ### trigger new call within the same query idx
            # start_time = time.time()
            searchSuccess, path =  self.serviceCall_astarPathFinding_labeledVersion(
                    violated_edges, initialConfig, targetConfig, 
                    start_neighbors_idx, goal_neighbors_idx,
                    start_neighbors_cost, goal_neighbors_cost,
                    occupied_labels, isInHandManipulation, 
                    robot, workspace, armType)
            # print("Time for service call for astarPathFinding_labeledVersion: {}".format(time.time() - start_time))
            if searchSuccess == False:
                print("the plan fails at the " + str(counter) + "th trial...")
                ### the plan fails, could not find a solution
                self.query_idx += 1
                return result_traj ### an empty trajectory
            ### otherwise, we need collision check and smoothing (len(path) >= 3)
            # start_time = time.time()
            smoothed_path, isPathValid, violated_edges = self.smoothPath(
                    path, initialConfig, targetConfig, robot, workspace, armType)
            # print("Time for smooth the path: {}".format(time.time() - start_time))            

        ### congrats, the path is valid and finally smoothed, let's generate trajectory
        print("smoothed path: ", smoothed_path)
        print("\n")
        ### directly generate trajectory based on the new path
        for i in range(0, len(smoothed_path)-1):
            if i == 0:
                config1 = initialConfig
            else:
                config1 = self.nodes[armType][smoothed_path[i]]
            if i == (len(smoothed_path)-2):
                config2 = targetConfig
            else:
                config2 = self.nodes[armType][smoothed_path[i+1]]
            ### get edge trajectory
            config_edge_traj = self.generateTrajectory_DirectConfigPath(config1, config2, robot, armType, workspace)
            # result_traj.append(config_edge_traj)
            result_traj += config_edge_traj

        ### before you claim the victory of this query, increment the planning query
        ### so as to tell people this query is over, next time is a new query
        self.query_idx += 1

        return result_traj

    def serviceCall_astarPathFinding_labeledVersion(self, 
            violated_edges, initialConfig, targetConfig, 
            start_neighbors_idx, goal_neighbors_idx, start_neighbors_cost, goal_neighbors_cost,
            occupied_labels, isInHandManipulation, 
            robot, workspace, armType):
        ### violated_edges: [Edge(), Edge(), ...]
        ### prepare the AstarPathFindingLabeledRequest
        rospy.wait_for_service("astar_path_finding_labeled")
        request = AstarPathFindingLabeledRequest()
        request.query_idx = self.query_idx
        request.start_idx = self.nsamples
        request.goal_idx = self.nsamples + 1
        request.start_config = initialConfig
        request.goal_config = targetConfig
        request.violated_edges = violated_edges
        request.armType = armType
        request.start_neighbors_idx = start_neighbors_idx
        request.goal_neighbors_idx = goal_neighbors_idx
        request.start_neighbors_cost = start_neighbors_cost
        request.goal_neighbors_cost = goal_neighbors_cost
        request.occupied_labels = occupied_labels
        request.isInHandManipulation = isInHandManipulation

        try:
            astarSearchLabeled = rospy.ServiceProxy("astar_path_finding_labeled", AstarPathFindingLabeled)
            response = astarSearchLabeled(request.query_idx, 
                request.start_idx, request.goal_idx,
                request.start_config, request.goal_config,
                request.start_neighbors_idx, request.goal_neighbors_idx,
                request.start_neighbors_cost, request.goal_neighbors_cost,
                request.occupied_labels, request.isInHandManipulation, 
                request.violated_edges, request.armType)
            return response.searchSuccess, list(response.path)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    ###################################################################################################################
    ###################################################################################################################
    def generateConfigBasedOnPose_candidates(self, pose, robot, workspace, armType, cylinder_positions_geometries):
        ### This function generates IK for a pose and check the IK
        ### in term of reachablity, essential collisions, as well as labels
        ### Input: pose: [[x,y,z],[x,y,z,w]]
        ###        armType: "Left(torso)" or "Right(torso)"
        ### Output: approaching_config, grasping_config, approaching_label, grasping_label, total_label

        approaching_config_candidates = [] ### a list of approaching configs
        grasping_config_candidates = [] ### a list of grasping configs
        approaching_labels = [] ### a list of labels per approaching config
        grasping_labels = [] ### a list of labels per grasping config
        total_labels = [] ### a list of labels per (approaching + grasping)

        if armType == "Left" or armType == "Left_torso":
            ee_idx = robot.left_ee_idx
            first_joint_index = 1
        if armType == "Right" or armType == "Right_torso":
            ee_idx = robot.right_ee_idx
            first_joint_index = 8

        tryAgain = True
        while tryAgain == True:
            ############################### check grasping pose #############################
            rest_pose = self.randomRestPose(robot, armType)
            ### we add rest pose in the IK solver to get as high-quality IK as possible
            config_IK = p.calculateInverseKinematics(bodyUniqueId=robot.motomanGEO,
                                    endEffectorLinkIndex=ee_idx,
                                    targetPosition=pose[0],
                                    targetOrientation=pose[1],
                                    lowerLimits=robot.ll, upperLimits=robot.ul, 
                                    jointRanges=robot.jr, restPoses=rest_pose,
                                    maxNumIterations=2000, residualThreshold=0.0000001,
                                    physicsClientId=robot.server)
            if armType == "Left" or armType == "Right":
                singleArmConfig_IK_grasping = list(config_IK[first_joint_index:first_joint_index+7])
            if armType == "Left_torso" or armType == "Right_torso":
                singleArmConfig_IK_grasping = [config_IK[0]] + list(config_IK[first_joint_index:first_joint_index+7])
            self.setRobotToConfig(singleArmConfig_IK_grasping, robot, armType)
            print("grasp IK: ")
            print(singleArmConfig_IK_grasping)
            isIKValid, FLAG = self.checkSamplePoseIK(pose, robot, workspace, armType)
            if not isIKValid:
                tryAgain = True if input("Try another IK for the grasping pose (y/n)?") == 'y' else False
                continue 
            ### check labels
            isConfigValid, FLAG, objectCollided_grasping = \
                self.checkConfig_CollisionBetweenRobotAndStaticObjects_labeled(robot, cylinder_positions_geometries)
            print("Labels for grasping pose: ", objectCollided_grasping)

            ############################### check approaching pose #############################
            temp_rot_matrix = p.getMatrixFromQuaternion(pose[1])
            ### local z-axis of the end effector
            temp_approaching_direction = [temp_rot_matrix[2], temp_rot_matrix[5], temp_rot_matrix[8]]
            temp_pos = list(np.array(pose[0]) - 0.05*np.array(temp_approaching_direction))
            new_pose = [temp_pos, pose[1]] ### the quaternion remains the same as input pose
            q_newPoseIK = p.calculateInverseKinematics(bodyUniqueId=robot.motomanGEO, 
                                    endEffectorLinkIndex=ee_idx, 
                                    targetPosition=new_pose[0], 
                                    targetOrientation=new_pose[1], 
                                    lowerLimits=robot.ll, upperLimits=robot.ul, 
                                    jointRanges=robot.jr, restPoses=rest_pose,
                                    maxNumIterations=2000, residualThreshold=0.0000001,
                                    physicsClientId=robot.server)
            if armType == "Left" or armType == "Right":
                singleArmConfig_IK_approaching = list(q_newPoseIK[first_joint_index:first_joint_index+7])
            if armType == "Left_torso" or armType == "Right_torso":
                singleArmConfig_IK_approaching = [q_newPoseIK[0]] + list(q_newPoseIK[first_joint_index:first_joint_index+7])
            self.setRobotToConfig(singleArmConfig_IK_approaching, robot, armType)
            print("approach IK: ")
            print(singleArmConfig_IK_approaching)
            isIKValid, FLAG = self.checkSamplePoseIK(new_pose, robot, workspace, armType)
            if not isIKValid:
                tryAgain = True if input("Try another IK for the grasping pose (y/n)?") == 'y' else False
                continue
            ### check labels
            isConfigValid, FLAG, objectCollided_approaching = \
                self.checkConfig_CollisionBetweenRobotAndStaticObjects_labeled(robot, cylinder_positions_geometries)
            print("Labels for approaching pose: ", objectCollided_approaching)

            ############### congrats! temporarily save this pose ###############
            print("approaching config: ", singleArmConfig_IK_approaching)
            print("grasping config: ", singleArmConfig_IK_grasping)
            print("approaching labels: ", set(objectCollided_approaching))
            print("grasping labels: ", set(objectCollided_grasping))
            print("total labels: ", set(objectCollided_approaching + objectCollided_grasping))
            approaching_config_candidates.append(singleArmConfig_IK_approaching)
            grasping_config_candidates.append(singleArmConfig_IK_grasping)
            approaching_labels.append(set(objectCollided_approaching))
            grasping_labels.append(set(objectCollided_grasping))
            total_labels.append(set(objectCollided_approaching + objectCollided_grasping))

            tryAgain = True if input("Try another IK for the grasping pose (y/n)?") == 'y' else False
        
        ### out of the loop
        print("out of the loop")
        if total_labels == []:
            return [], [], set(), set(), set()
        else:
            print("approaching_config_candidates: ", approaching_config_candidates)
            print("grasping_config_candidates: ", grasping_config_candidates)
            print("approaching labels: ", approaching_labels)
            print("grasping labels: ", grasping_labels)
            print("total labels: ", total_labels)
            pose_id = int(input("choose your favorite pose"))
            if pose_id >= len(total_labels):
                print("no pose is satisfying. return nothing")
                return [], [], set(), set(), set()
            else:
                return approaching_config_candidates[pose_id], grasping_config_candidates[pose_id], \
                    approaching_labels[pose_id], grasping_labels[pose_id], total_labels[pose_id]
    ###################################################################################################################
    
    ###################################################################################################################
    def generateConfigBasedOnPose_initialPositions(self, pose, robot, workspace, armType, cylinder_positions_geometries):
        ### This function generates IK for a pose and check the IK
        ### in term of reachablity, essential collisions, as well as labels
        ### Input: pose: [[x,y,z],[x,y,z,w]]
        ###        armType: "Left(torso)" or "Right(torso)"
        ### Output: approaching_config, grasping_config, approaching_label, grasping_label, total_label

        if armType == "Left" or armType == "Left_torso":
            ee_idx = robot.left_ee_idx
            first_joint_index = 1
        if armType == "Right" or armType == "Right_torso":
            ee_idx = robot.right_ee_idx
            first_joint_index = 8

        max_trials = 10
        num_trials = 0
        isIKValid = False
        while (not isIKValid) and (num_trials < max_trials):
            ### initialization
            ############################### check grasping pose #############################
            rest_pose = self.randomRestPose(robot, armType)
            ### we add rest pose in the IK solver to get as high-quality IK as possible
            config_IK = p.calculateInverseKinematics(bodyUniqueId=robot.motomanGEO,
                                    endEffectorLinkIndex=ee_idx,
                                    targetPosition=pose[0],
                                    targetOrientation=pose[1],
                                    lowerLimits=robot.ll, upperLimits=robot.ul, 
                                    jointRanges=robot.jr, restPoses=rest_pose,
                                    maxNumIterations=2000, residualThreshold=0.0000001,
                                    physicsClientId=robot.server)
            if armType == "Left" or armType == "Right":
                singleArmConfig_IK_grasping = list(config_IK[first_joint_index:first_joint_index+7])
            if armType == "Left_torso" or armType == "Right_torso":
                singleArmConfig_IK_grasping = [config_IK[0]] + list(config_IK[first_joint_index:first_joint_index+7])
            self.setRobotToConfig(singleArmConfig_IK_grasping, robot, armType)
            # print("grasp IK: ")
            # print(singleArmConfig_IK_grasping)
            isIKValid, FLAG = self.checkSamplePoseIK(pose, robot, workspace, armType)
            if not isIKValid:
                # print("grasping pose not valid, FLAG: " + str(FLAG))
                num_trials += 1
                continue
            ### check labels
            isConfigValid, FLAG, objectCollided_grasping = \
                self.checkConfig_CollisionBetweenRobotAndStaticObjects_labeled(robot, cylinder_positions_geometries)
            # print("Labels for grasping pose: ", objectCollided_grasping)

            ############################### check approaching pose #############################
            temp_rot_matrix = p.getMatrixFromQuaternion(pose[1])
            ### local z-axis of the end effector
            temp_approaching_direction = [temp_rot_matrix[2], temp_rot_matrix[5], temp_rot_matrix[8]]
            temp_pos = list(np.array(pose[0]) - 0.05*np.array(temp_approaching_direction))
            new_pose = [temp_pos, pose[1]] ### the quaternion remains the same as input pose
            q_newPoseIK = p.calculateInverseKinematics(bodyUniqueId=robot.motomanGEO, 
                                    endEffectorLinkIndex=ee_idx, 
                                    targetPosition=new_pose[0], 
                                    targetOrientation=new_pose[1], 
                                    lowerLimits=robot.ll, upperLimits=robot.ul, 
                                    jointRanges=robot.jr, restPoses=rest_pose,
                                    maxNumIterations=2000, residualThreshold=0.0000001,
                                    physicsClientId=robot.server)
            if armType == "Left" or armType == "Right":
                singleArmConfig_IK_approaching = list(q_newPoseIK[first_joint_index:first_joint_index+7])
            if armType == "Left_torso" or armType == "Right_torso":
                singleArmConfig_IK_approaching = [q_newPoseIK[0]] + list(q_newPoseIK[first_joint_index:first_joint_index+7])
            self.setRobotToConfig(singleArmConfig_IK_approaching, robot, armType)
            # print("approach IK: ")
            # print(singleArmConfig_IK_approaching)
            isIKValid, FLAG = self.checkSamplePoseIK(new_pose, robot, workspace, armType)
            if not isIKValid:
                # print("approaching pose not valid, FLAG: " + str(FLAG))
                num_trials += 1
                continue
            ### check labels
            isConfigValid, FLAG, objectCollided_approaching = \
                self.checkConfig_CollisionBetweenRobotAndStaticObjects_labeled(robot, cylinder_positions_geometries)
            # print("Labels for approaching pose: ", objectCollided_approaching)

        ### out of the loop
        if not isIKValid:
            return [],[],set(), set(), set()
        else:
            return singleArmConfig_IK_approaching, singleArmConfig_IK_grasping, \
                set(objectCollided_approaching), set(objectCollided_grasping), set(objectCollided_approaching+objectCollided_grasping)
    ###################################################################################################################

    ###################################################################################################################
    def generateOrientations(self, 
            default_orientation=np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            possible_angles=[0, -15, -45, -30, 15, 45, 30]):
        ###### this function generates all orientation given default and angle options ######
        ### defalut_orientation: np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
        orientations = []
        for angle in possible_angles:
            angle *= math.pi / 180.0
            rotation = np.array([[math.cos(angle), -math.sin(angle), 0], 
                                 [math.sin(angle), math.cos(angle), 0], 
                                 [0, 0, 1]])
            orientation = np.dot(default_orientation, rotation)
            temp_quat = utils.getQuaternionFromRotationMatrix(orientation)
            orientations.append(temp_quat)
        return orientations    
    ###################################################################################################################

    ###################################################################################################################
    def generate_pose_candidates(self, object_position, cylinder_height):
        '''This function generate pose candidates for the given object_position'''
        ### object_position: [x,y,z]
        pose_candidates = []
        orientations = self.generateOrientations()
        ### for each orientation
        for orientation in orientations:
            targetPose = [[object_position[0], object_position[1], \
                           object_position[2] + cylinder_height/2 - 0.03], orientation]
            pose_candidates.append(targetPose)

        return pose_candidates
    ###################################################################################################################

    ###################################################################################################################
    def generateAllConfigPoses_startPositions(self, robot, workspace, armType):
        self.object_initial_configPoses = OrderedDict()
        cylinder_positions_geometries = {
            candidate.position_idx : candidate.geo for candidate in workspace.candidate_geometries.values()}
        for obj_idx, obj_initial_info in workspace.object_initial_infos.items():
            ### for each object
            self.object_initial_configPoses[obj_idx] = PositionCandidateConfigs(
                                                            workspace.object_initial_infos[obj_idx].position_idx)
            ### first generate graspingPose candidates with different orientations
            graspingPose_candidates = self.generate_pose_candidates(obj_initial_info.pos, workspace.cylinder_height)
            for pose_id, graspingPose in enumerate(graspingPose_candidates):
                approaching_config, grasping_config, approaching_label, grasping_label, total_label = \
                    self.generateConfigBasedOnPose_initialPositions(
                        graspingPose, robot, workspace, armType, cylinder_positions_geometries)
                if approaching_config != []:
                    self.object_initial_configPoses[obj_idx].approaching_configs.append(approaching_config)
                    self.object_initial_configPoses[obj_idx].grasping_configs.append(grasping_config)
                    self.object_initial_configPoses[obj_idx].approaching_labels.append(approaching_label)
                    self.object_initial_configPoses[obj_idx].grasping_labels.append(grasping_label)
                    self.object_initial_configPoses[obj_idx].total_labels.append(total_label)
        print("========= finish generate all object_initial_configPoses =========")
        ### put the robot back to home configuration please
        robot.resetRobotToHomeConfiguration()
        ### uncomment below printing utility if you need debug ###
        # for obj_idx, object_initial_configs in self.planner_p.object_initial_configPoses.items():
        #     print(obj_idx)
        #     print("approaching_configs: " + str(object_initial_configs.approaching_configs))
        #     print("grasping_configs: " + str(object_initial_configs.grasping_configs))
        #     print("approaching_labels: " + str(object_initial_configs.approaching_labels))
        #     print("grasping_labels: " + str(object_initial_configs.grasping_labels))
        #     print("total_labels: " + str(object_initial_configs.total_labels))
        #     print("\n")        
    ###################################################################################################################

    #########################################################################################
    def generatePosesForAllCandidates(self, robot, workspace, armType):
        self.position_candidates_configPoses = OrderedDict()
        cylinder_positions_geometries = {
            candidate.position_idx : candidate.geo for candidate in workspace.candidate_geometries.values()}
        for candidate_idx, cylinder_candidate in workspace.candidate_geometries.items():
            print("++++++++++++++CANDIDATE_IDX: " + str(candidate_idx) + "++++++++++++++")
            self.position_candidates_configPoses[candidate_idx] = PositionCandidateConfigs(candidate_idx)
            ### first generate graspingPose_candidates with different orientations
            graspingPose_candidates = self.generate_pose_candidates(cylinder_candidate.pos, workspace.cylinder_height)
            for pose_id, graspingPose in enumerate(graspingPose_candidates):
                approaching_config, grasping_config, approaching_label, grasping_label, total_label = \
                    self.generateConfigBasedOnPose_candidates(
                        graspingPose, robot, workspace, armType, cylinder_positions_geometries)
                if approaching_config != []:
                    self.position_candidates_configPoses[candidate_idx].approaching_configs.append(approaching_config)
                    self.position_candidates_configPoses[candidate_idx].grasping_configs.append(grasping_config)
                    self.position_candidates_configPoses[candidate_idx].approaching_labels.append(approaching_label)
                    self.position_candidates_configPoses[candidate_idx].grasping_labels.append(grasping_label)
                    self.position_candidates_configPoses[candidate_idx].total_labels.append(total_label)

            print("=============================================================================")
            print("candidate " + str(candidate_idx) + " approaching_configs: ", \
                self.position_candidates_configPoses[candidate_idx].approaching_configs)
            print("candidate " + str(candidate_idx) + " grasping_configs: ", 
                self.position_candidates_configPoses[candidate_idx].grasping_configs)
            print("candidate " + str(candidate_idx) + " approaching_labels: ", \
                self.position_candidates_configPoses[candidate_idx].approaching_labels)
            print("candidate " + str(candidate_idx) + " grasping_labels: ", \
                self.position_candidates_configPoses[candidate_idx].grasping_labels)
            print("candidate " + str(candidate_idx) + " total_labels: ", \
                self.position_candidates_configPoses[candidate_idx].total_labels)
            input("ENTER to next candidate")
        
        ### wooo!!! finished!
        ### save the whole position_candidates_configPoses
        self.serializeCandidatesConfigPoses()
    #########################################################################################

    #########################################################################################
    def generatePoses_IKdataSet(self, robot, workspace, armType):
        cylinder_positions_geometries = {
            candidate.position_idx : candidate.geo for candidate in workspace.candidate_geometries.values()}
        generateMore = True
        while(generateMore):
            candidate_idx = int(input('which candidate_idx are you interested?'))
            cylinder_candidate = workspace.candidate_geometries[candidate_idx]
            print("++++++++++++++CANDIDATE_IDX: " + str(candidate_idx) + "++++++++++++++")
            ### first generate graspingPose_candidates with different orientations
            graspingPose_candidates = self.generate_pose_candidates(cylinder_candidate.pos, workspace.cylinder_height)
            for pose_id, graspingPose in enumerate(graspingPose_candidates):
                approaching_config, grasping_config, approaching_label, grasping_label, total_label = \
                    self.generateConfigBasedOnPose_candidates(
                        graspingPose, robot, workspace, armType, cylinder_positions_geometries)

            generateMore = True if input('generate more? (y/n)') == 'y' else False
    #########################################################################################

    def serializeCandidatesConfigPoses(self):
        f_candidate_geometries = open(self.roadmapFolder+"/CandidatesConfigPoses.obj", 'wb')
        pickle.dump(self.position_candidates_configPoses, f_candidate_geometries)

    def deserializeCandidatesConfigPoses(self):
        f_candidate_geometries = open(self.roadmapFolder+"/CandidatesConfigPoses.obj", 'rb')
        self.position_candidates_configPoses = pickle.load(f_candidate_geometries)
        for position, position_candidate_config in self.position_candidates_configPoses.items():
            position_candidate_config.approaching_labels_all = []
            position_candidate_config.grasping_labels_all = []
            position_candidate_config.total_labels_all = []

    def reset_CandidatesConfigPoses(self):
        for position, position_candidate_config in self.position_candidates_configPoses.items():
            position_candidate_config.approaching_labels_all = []
            position_candidate_config.grasping_labels_all = []
            position_candidate_config.total_labels_all = []
        for obj_idx, object_initial_config in self.object_initial_configPoses.items():
            object_initial_config.approaching_labels_all = []
            object_initial_config.grasping_labels_all = []
            object_initial_config.total_labels_all = []

    
    #########################################################################################
    ### ATTENTION: THE FUNCTION BELOW IS NOT USED AND THUS NOT BEING MAINTAINED AT THIS POINT
    def calculateReachabilityMap(self, robot, workspace, orientation, placeholder_shape="cylinder"):
        ### calculate the reachability map for a particular orientation
        ### mark with the shape of different color specified
        ### Input: orientation is a format of quaternion (x,y,z,w)
        ### set up the two dictionaries as the mapping
        flag_color = {
            0: [0,1,0,1],
            1: [1,0,0,1],
            2: [0,0,1,1],
            3: [1,1,0,1]
        }
        obj_idx = 0
        for object_pos in workspace.all_goal_positions.values():
            print("current obj_idx: {}".format(obj_idx))
            temp_pos = copy.deepcopy(object_pos)
            temp_pos[2] = object_pos[2] + workspace.cylinder_height/2 - 0.03
            target_pose = [temp_pos, orientation]
            ### first check the pose
            isPoseValid, FLAG, configToPickingPose = self.generateConfigBasedOnPose(
                        target_pose, obj_idx, robot, workspace, "Right_torso")
            if not isPoseValid:
                print("the pose is not valid")
                if placeholder_shape == "cylinder":
                    temp_placeholder_v = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                        radius=workspace.cylinder_radius, length=workspace.cylinder_height, 
                        rgbaColor=flag_color[FLAG], physicsClientId=self.planningServer)
                if placeholder_shape == "sphere":
                    temp_placeholder_v = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                        radius=0.005, rgbaColor=flag_color[FLAG], physicsClientId=self.planningServer)
                temp_placeholderM = p.createMultiBody(
                    baseVisualShapeIndex=temp_placeholder_v, basePosition=object_pos, physicsClientId=self.planningServer)
            else:
                ### then check the pre-grasp pose
                isPoseValid, FLAG, prePickingPose, configToPrePickingPose = \
                    self.generatePrePickingPose(
                        target_pose, obj_idx, robot, workspace, "Right_torso")
                if not isPoseValid:
                    print("the pre-picking pose is not valid")
                    if placeholder_shape == "cylinder":
                        temp_placeholder_v = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                            radius=workspace.cylinder_radius, length=workspace.cylinder_height, 
                            rgbaColor=flag_color[FLAG], physicsClientId=self.planningServer)
                    if placeholder_shape == "sphere":
                        temp_placeholder_v = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                            radius=0.005, rgbaColor=flag_color[FLAG], physicsClientId=self.planningServer)
                    temp_placeholderM = p.createMultiBody(
                        baseVisualShapeIndex=temp_placeholder_v, basePosition=object_pos, physicsClientId=self.planningServer)
            if isPoseValid:
                print("both picking and pre-picking poses are valid")
                if placeholder_shape == "cylinder":
                    temp_placeholder_v = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                        radius=workspace.cylinder_radius, length=workspace.cylinder_height, 
                        rgbaColor=flag_color[FLAG], physicsClientId=self.planningServer)
                if placeholder_shape == "sphere":
                    temp_placeholder_v = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                        radius=0.005, rgbaColor=flag_color[FLAG], physicsClientId=self.planningServer)
                temp_placeholderM = p.createMultiBody(
                    baseVisualShapeIndex=temp_placeholder_v, basePosition=object_pos, physicsClientId=self.planningServer)
            print("finish the object {}".format(obj_idx))
            obj_idx += 1
            input("press to continue")
        ### out of the loop
        robot.resetArmConfig_torso(
            robot.leftArmHomeConfiguration+robot.rightArmHomeConfiguration, robot.torsoHomeConfiguration)
        time.sleep(10000)
    #########################################################################################


class PositionCandidateConfigs(object):
    def __init__(self, position_idx):
        self.position_idx = position_idx
        self.approaching_configs = [] ### a list of configs(list) with different orientations
        self.grasping_configs = [] ### a list of configs(list) with different orientations
        self.approaching_labels = [] ### a list of approaching labels(set) with different orientations
        self.grasping_labels = [] ### a list of grasping labels(set) with different orientations
        self.total_labels = [] ### a list of total labels(set) with different orientations
    ### More functions to be added later (if necessary) ###
        self.approaching_labels_all = [] ### a list of approaching labels(set) with different orientations (including start labels)
        self.grasping_labels_all = [] ### a list of grasping labels(set) with different orientations (including start labels)
        self.total_labels_all = [] ### a list of total labels(set) with different orientations (including start labels)
