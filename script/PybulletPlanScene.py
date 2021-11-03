#!/usr/bin/env python
from __future__ import division

import pybullet as p
import pybullet_data

import time
import sys
import os
import copy
import math
import numpy as np
import random
from collections import OrderedDict

import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point

from MotomanRobot import MotomanRobot
from WorkspaceTable import WorkspaceTable
from Planner import Planner
from Planner import PositionCandidateConfigs
import utils

from confined_space_rearrangement.msg import ArmTrajectory
from confined_space_rearrangement.msg import ObjectRearrangePath
from confined_space_rearrangement.msg import ArrState
from confined_space_rearrangement.msg import ObjArrStates
from confined_space_rearrangement.srv import ReproduceInstanceCylinder, ReproduceInstanceCylinderResponse
from confined_space_rearrangement.srv import GenerateConfigsForStartPositions, GenerateConfigsForStartPositionsResponse
from confined_space_rearrangement.srv import DetectInvalidArrStates, DetectInvalidArrStatesResponse
from confined_space_rearrangement.srv import RearrangeCylinderObject, RearrangeCylinderObjectResponse
from confined_space_rearrangement.srv import GetCurrRobotConfig, GetCurrRobotConfigResponse
from confined_space_rearrangement.srv import UpdateCertainObjectPose, UpdateCertainObjectPoseResponse
from confined_space_rearrangement.srv import ResetRobotCurrConfig, ResetRobotCurrConfigResponse
from confined_space_rearrangement.srv import UpdateManipulationStatus, UpdateManipulationStatusResponse
from confined_space_rearrangement.srv import SetSceneBasedOnArrangement, SetSceneBasedOnArrangementResponse
from confined_space_rearrangement.srv import SelectObjectAndBuffer, SelectObjectAndBufferResponse
from confined_space_rearrangement.srv import ResetPlanningInstance, ResetPlanningInstanceResponse
from confined_space_rearrangement.srv import ClearPlanningInstance, ClearPlanningInstanceResponse
from confined_space_rearrangement.srv import ResetRobotHome, ResetRobotHomeResponse

################################## description #####################################
### This class defines a PybulletPlanScene class which
### entertain an planning scene that
### (1) reproduces the instance from the execution scene based on perception output
### (2) performs planning for manipulation (transit + transfer)
### (3) communicates with grasp_pose node to get the grasp pose for grasping objects
####################################################################################


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class PybulletPlanScene(object):

    def __init__(self, args):
        ### read in relevant ros parameters for plan scene
        basePosition, baseOrientation, urdfFile, \
        leftArmHomeConfiguration, rightArmHomeConfiguration, torsoHomeConfiguration, \
        standingBase_dim, table_dim, table_offset_x, \
        cylinder_radius, cylinder_height, \
        discretization_x, discretization_y, \
        object_interval_x, object_interval_y, \
        side_clearance_x, side_clearance_y, \
        ceiling_height, thickness_flank, \
        object_mesh_path = self.readROSParam()
        
        ### set the rospkg path
        rospack = rospkg.RosPack()
        self.rosPackagePath = rospack.get_path("confined_space_rearrangement")
	
        ### set the server for the pybullet plan scene
        # self.planningClientID = p.connect(p.DIRECT)
        self.planningClientID = p.connect(p.GUI)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.egl_plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        # print("plugin=", self.egl_plugin)

        ### configure the robot
        self.configureMotomanRobot(urdfFile, basePosition, baseOrientation, \
            leftArmHomeConfiguration, rightArmHomeConfiguration, torsoHomeConfiguration, False)
        ### setup the workspace
        self.setupWorkspace(standingBase_dim, table_dim, table_offset_x, object_mesh_path, False)
        self.workspace_p.addConstrainedArea(ceiling_height, thickness_flank)
        self.workspace_p.setDeploymentParam(
                cylinder_radius, cylinder_height, side_clearance_x, side_clearance_y, \
                discretization_x, discretization_y, object_interval_x, object_interval_y)
        self.workspace_p.deployAllPositionCandidates(generateMesh=True)

        ### create a planner assistant
        self.planner_p = Planner(
            self.rosPackagePath, self.planningClientID,
            isObjectInLeftHand=False, isObjectInRightHand=False,
            objectInLeftHand=None, objectInRightHand=None)


    def configureMotomanRobot(self, 
            urdfFile, basePosition, baseOrientation,
            leftArmHomeConfiguration, rightArmHomeConfiguration, torsoHomeConfiguration, isPhysicsTurnOn):
        ### This function configures the robot in the real scene
        self.robot_p = MotomanRobot(
            os.path.join(self.rosPackagePath, urdfFile), 
            basePosition, baseOrientation, 
            leftArmHomeConfiguration, rightArmHomeConfiguration, torsoHomeConfiguration,
            isPhysicsTurnOn, self.planningClientID)

    def setupWorkspace(self,
            standingBase_dim, table_dim, table_offset_x,
            object_mesh_path, isPhysicsTurnOn):
        ### This function sets up the workspace
        self.workspace_p = WorkspaceTable(self.rosPackagePath, self.robot_p.basePosition,
            standingBase_dim, table_dim, table_offset_x, 
            os.path.join(self.rosPackagePath, object_mesh_path),
            isPhysicsTurnOn, self.planningClientID)


    def rosInit(self):
        ### This function specifies the role of a node instance for this class
        ### and initialize a ros node
        self.reproduce_instance_cylinder_server = rospy.Service(
            "reproduce_instance_cylinder", ReproduceInstanceCylinder,
            self.reproduce_instance_cylinder_callback)

        self.rearrange_cylinder_object_server = rospy.Service(
            "rearrange_cylinder_object", RearrangeCylinderObject,
            self.rearrange_cylinder_object_callback)

        self.generate_configs_for_start_positions_server = rospy.Service(
            "generate_configs_for_start_positions", GenerateConfigsForStartPositions,
            self.generate_configs_for_start_positions_callback)

        self.detect_invalid_arr_states_server = rospy.Service(
            "detect_invalid_arr_states", DetectInvalidArrStates,
            self.detect_invalid_arr_states_callback)

        self.get_curr_robot_config_server = rospy.Service(
            "get_curr_robot_config", GetCurrRobotConfig,
            self.get_curr_robot_config_callback)

        self.update_certain_object_pose_server = rospy.Service(
            "update_certain_object_pose", UpdateCertainObjectPose,
            self.update_certain_object_pose_callback)

        self.reset_robot_curr_config_server = rospy.Service(
            "reset_robot_curr_config", ResetRobotCurrConfig,
            self.reset_robot_curr_config_callback)

        self.update_manipulation_status_server = rospy.Service(
            "update_manipulation_status", UpdateManipulationStatus,
            self.update_manipulation_status_callback)

        self.set_scene_basedOn_arrangement_server = rospy.Service(
            "set_scene_based_on_arrangement", SetSceneBasedOnArrangement,
            self.set_scene_basedOn_arrangement_callback)

        self.select_object_and_buffer_server = rospy.Service(
            "select_object_and_buffer", SelectObjectAndBuffer,
            self.select_object_and_buffer_callback)

        self.reset_planning_instance_server = rospy.Service(
            "reset_planning_instance", ResetPlanningInstance,
            self.reset_planning_instance_callback)

        self.clear_planning_instance_server = rospy.Service(
            "clear_planning_instance", ClearPlanningInstance,
            self.clear_planning_instance_callback)

        self.reset_robot_home_server = rospy.Service(
            "reset_robot_home", ResetRobotHome,
            self.reset_robot_home_callback)

        rospy.init_node("pybullet_plan_scene", anonymous=True)


    def reproduce_instance_cylinder_callback(self, req):
        ### given the estimated cylinder objects
        rospy.logwarn("REPRODUCE REARRANGEMENT INSTANCE")
        initial_arrangement, final_arrangement, success = \
            self.workspace_p.reproduceInstance_cylinders(req.cylinder_objects)
        if success == True:
            print("successfully reproduce an instance")
        else:
            print("fail to reproduce an instance")
        return ReproduceInstanceCylinderResponse(initial_arrangement, final_arrangement, success)

    def generate_configs_for_start_positions_callback(self, req):
        rospy.logwarn("GENERATE CONFIGS FOR START POSITIONS OF ALL OBJECTS")
        self.planner_p.generateAllConfigPoses_startPositions(self.robot_p, self.workspace_p, req.armType)
        return GenerateConfigsForStartPositionsResponse(True)

    def detect_invalid_arr_states_callback(self, req):
        rospy.logwarn("DETECT INVALID ARR STATES")
        ### data initialization
        self.planner_p.invalid_arr_states_per_obj = OrderedDict()
        for obj_idx in range(len(req.start_arrangement)):
            self.planner_p.invalid_arr_states_per_obj[obj_idx] = []
        ### reason about each object to be manipulated
        all_objects = [i for i in range(len(req.start_arrangement)) \
            if req.start_arrangement[i] != req.target_arrangement[i]]
        for obj_idx in all_objects:
            print("obj_idx: " + str(obj_idx))
            #################################################################################
            ### get the object's all pre-picking + picking configPoses
            curr_object_configPoses = \
                self.planner_p.obtainCurrObjectConfigPoses(self.workspace_p, obj_idx)
            ### get picking_configPoses_constraints (a list of list of objects) for this object
            picking_configPoses_constraints = self.planner_p.getConstraintsFromLabels(
                curr_object_configPoses, obj_idx, req.target_arrangement, "picking")
            # print("picking_configPoses_constraints: ")
            # print(picking_configPoses_constraints)
            self.planner_p.addInvalidArrStates(picking_configPoses_constraints, obj_idx)
            ##################################################################################
            ##################################################################################
            ### get the object's all placing configPoses
            target_object_configPoses = \
                self.planner_p.position_candidates_configPoses[req.target_arrangement[obj_idx]]
            ### get placing_configPoses_constraints (a list of list of objects) for this object
            placing_configPoses_constraints = self.planner_p.getConstraintsFromLabels(
                target_object_configPoses, obj_idx, req.target_arrangement, "placing")
            # print("placing_configPoses_constraints: ")
            # print(placing_configPoses_constraints)
            self.planner_p.addInvalidArrStates(placing_configPoses_constraints, obj_idx)
            ##################################################################################
        # print("invalid_arr_states_per_obj: ")
        # print(self.planner_p.invalid_arr_states_per_obj)
        # input("Check invalid arr states per object, to get an overall idea why this instance is non-monotone...")
        ### prepare the response
        detect_invalid_arr_states_response = DetectInvalidArrStatesResponse()
        for obj_idx, obj_arr_states in self.planner_p.invalid_arr_states_per_obj.items():
            obj_arr_states_msg = ObjArrStates()
            #################################################
            obj_arr_states_msg.obj_idx = obj_idx
            for arr_state in obj_arr_states:
                ### each arr_state is a dict, construct it as ArrState msg
                arr_state_msg = ArrState()
                for obj, isAtTarget in arr_state.items():
                    arr_state_msg.obj_indices.append(obj)
                    arr_state_msg.isAtTarget.append(isAtTarget)
                obj_arr_states_msg.invalid_arr_states.append(arr_state_msg)
            #################################################
            detect_invalid_arr_states_response.all_obj_invalid_arr_states.append(obj_arr_states_msg)
        # print("check ros messages")
        # for obj_arr_states_msg in detect_invalid_arr_states_response.all_obj_invalid_arr_states:
        #     print("obj_idx: " + str(obj_arr_states_msg.obj_idx))
        #     for arr_state_msg in obj_arr_states_msg.invalid_arr_states:
        #         print("obj_indices: ")
        #         print(arr_state_msg.obj_indices)
        #         print("isAtTarget: ")
        #         print(arr_state_msg.isAtTarget)
        #     print("========================")
        return detect_invalid_arr_states_response


    def get_curr_robot_config_callback(self, req):
        ### get the current robot config
        joint_state = JointState()
        joint_state.position = self.robot_p.getRobotCurrConfig()
        # print("successfully get the current robot configuration")
        return GetCurrRobotConfigResponse(joint_state)

    def update_certain_object_pose_callback(self, req):
        ### update the geometry mesh of a certain object to the target pose
        self.workspace_p.updateObjectMesh(req.object_idx, req.object_position_idx)
        # print("successfully update certain object geometry to the target pose" + str(position))
        return UpdateCertainObjectPoseResponse(True)

    def reset_robot_curr_config_callback(self, req):
        ### reset the robot to the specified configuration
        joint_positions = list(req.robot_config.position)
        self.robot_p.resetArmConfig_torso(joint_positions[1:15], joint_positions[0])
        ### update the hand as well (I doubt it is a redundant step at this point)
        self.robot_p.resetRightHandConfig(joint_positions[15:21])
        # print("successfully reset the robot current configuration")
        return ResetRobotCurrConfigResponse(True)

    def update_manipulation_status_callback(self, req):
        ### disable any relationship between the robot and any of the object
        self.planner_p.detachObject(self.workspace_p, self.robot_p, req.armType)
        # print("successfully update the manipulation status")
        return UpdateManipulationStatusResponse(True)

    def set_scene_basedOn_arrangement_callback(self, req):
        ### (i) put all the objects to the position specified in the arrangement
        self.workspace_p.set_scene_for_objects(req.arrangement)
        ### (ii) reset the robot to the specified configuration
        joint_positions = list(req.robot_config.position)
        # self.robot_p.setSingleArmToConfig_torso(joint_positions[8:15], joint_positions[0], req.armType)
        self.robot_p.resetArmConfig_torso(joint_positions[1:15], joint_positions[0])
        ### update the hand as well (I doubt it is a redundant step at this point)
        self.robot_p.resetRightHandConfig(joint_positions[15:21])
        ### (iii) disable any relationship between the robot and any of the object
        self.planner_p.detachObject(self.workspace_p, self.robot_p, req.armType)
        return SetSceneBasedOnArrangementResponse(True)

    def select_object_and_buffer_callback(self, req):
        ############################## first select an object ##############################
        object_path = ObjectRearrangePath()
        if req.heuristic_level == 0:
            object_idx = random.choice(list(req.objects_to_move))
        if req.heuristic_level == 1:
            object_ranking = self.workspace_p.getObjectConstraintRanking(req.objects_to_move, req.final_arrangement)
            object_idx = object_ranking[0]
        ### Once select the object, check if it can be reached based on current arrangement
        transit_success, transit_traj, pickingPose_neighbors_idx, pickingPose_neighbors_cost = \
                                                    self.transit_cylinder_object(object_idx, req)
        if transit_success == False:
            ### the selected object is not reachable
            print("The selected object is not reachable, let alone selecting buffer")
            return SelectObjectAndBufferResponse(transit_success, -1, -1, object_path)
        ####################################################################################

        ### before move on to selecting and putting on a buffer, do some book keeping
        obj_position_idx_transit = self.workspace_p.object_geometries[object_idx].curr_position_idx
        robot_config_transit = self.robot_p.getRobotCurrConfig()

        ############################### then select a buffer ###############################
        ### 3 chances are given for selecting a buffer
        max_trials = 3
        current_trials = 1
        buffer_success = False
        while (current_trials < max_trials) and (buffer_success == False):
            ### select a buffer, which 
            ### (1) is not the current/target position of the selected object
            ### (2) should not collide with any objects other than itself
            buffer_select_success, buffer_idx = self.workspace_p.selectNoCollisionBuffer(
                                            object_idx, req.final_arrangement[object_idx])
            if buffer_select_success:
                ### check the transfer path to the buffer location for that object
                transfer_success, transfer_traj, finish_traj = \
                    self.transfer_cylinder_object(
                        object_idx, buffer_idx, pickingPose_neighbors_idx, pickingPose_neighbors_cost, req)
                if transfer_success == True:
                    ### congrats! you successfully transfer the object to the buffer
                    buffer_success = True
                    break
                else:
                    ### it does not work, put it back to the end of transit stage
                    ### (i) put the object back to the end of transit
                    self.workspace_p.updateObjectMesh(object_idx, obj_position_idx_transit)
                    ### (ii) put the arm back to the end of transit
                    self.robot_p.resetArmConfig_torso(robot_config_transit[1:15], robot_config_transit[0])
                    ### update the hand as well (I doubt it is a redundant step at this point)
                    self.robot_p.resetRightHandConfig(robot_config_transit[15:21])
                    ### (iii) update manipulation status
                    self.planner_p.detachObject(self.workspace_p, self.robot_p, req.armType)
                    current_trials += 1
                    continue
            else:
                current_trials += 1
                continue
        ### reach here either success or not
        if buffer_success == True:
            object_path.transit_trajectory = self.generateArmTrajectory(
                                transit_traj, req.armType, self.robot_p.motomanRJointNames)
            object_path.transfer_trajectory = self.generateArmTrajectory(
                                transfer_traj, req.armType, self.robot_p.motomanRJointNames)
            object_path.finish_trajectory = self.generateArmTrajectory(
                                finish_traj, req.armType, self.robot_p.motomanRJointNames)
            object_path.object_idx = object_idx
            return SelectObjectAndBufferResponse(buffer_success, object_idx, buffer_idx, object_path)
        else:
            return SelectObjectAndBufferResponse(buffer_success, -1, -1, object_path)
        ####################################################################################


    def reset_planning_instance_callback(self, req):
        ### reset the instance in the planning scene, which involves
        ### (i) reset all object meshes (current collision bodies) in the workspace
        self.workspace_p.reset_planning_instance()
        ### (ii) reset some planner parameters
        self.planner_p.resetPlannerParams()
        ### (iii) reset the robot back to the home configuration
        self.robot_p.resetRobotToHomeConfiguration()
        return ResetPlanningInstanceResponse(True)

    def clear_planning_instance_callback(self, req):
        ### clear the instance in the planning scene, which involves
        ### (i) delete all object meshes (current collision bodies/goal visualization) in the workspace
        self.workspace_p.clear_planning_instance()
        ### (ii) reset some planner parameters
        self.planner_p.resetPlannerParams()
        ### (iii) reset the robot back to the home configuration
        self.robot_p.resetRobotToHomeConfiguration()
        return ClearPlanningInstanceResponse(True)

    def reset_robot_home_callback(self, req):
        resetHome_trajectory = ArmTrajectory()
        ### reset the robot to home configuration
        ######################## check currConfig's neighboring connectivity ########################
        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        connectSuccess, currConfig_neighbors_idx, currConfig_neighbors_cost = self.planner_p.connectToNeighbors(
                                    currConfig, self.robot_p, self.workspace_p, req.armType)
        if not connectSuccess:
            print("There exists connection problem for current config. Not be able to reset the robot home")
            return ResetRobotHomeResponse(False, resetHome_trajectory)
        ##############################################################################################

        ######################## check homeConfig's neighboring connectivity ########################
        if req.armType == "Right_torso":
            homeConfig = [self.robot_p.torsoHomeConfiguration] + self.robot_p.rightArmHomeConfiguration
        connectSuccess, homeConfig_neighbors_idx, homeConfig_neighbors_cost = self.planner_p.connectToNeighbors(
                                    homeConfig, self.robot_p, self.workspace_p, req.armType)
        if not connectSuccess:
            print("home config has problem of connecting to neighbors, which should not happen. Not be able to reset the robot home")
            return ResetRobotHomeResponse(False, resetHome_trajectory)
        ##############################################################################################

        ####################### motion planning from currConfig to homeConfig ########################
        resetHome_traj = self.planner_p.AstarPathFinding(currConfig, homeConfig,
                        currConfig_neighbors_idx, currConfig_neighbors_cost, 
                        homeConfig_neighbors_idx, homeConfig_neighbors_cost,
                        self.robot_p, self.workspace_p, req.armType, req.isLabeledRoadmapUsed)
        if resetHome_traj != []:
            print("The reset_home path for %s arm is successfully found" % req.armType)
            ### generate ArmTrajectory from resetHome_traj
            resetHome_trajectory = self.generateArmTrajectory(resetHome_traj, req.armType, self.robot_p.motomanRJointNames)
            return ResetRobotHomeResponse(True, resetHome_trajectory)
        else:
            print("The reset_home path for %s arm is not successfully found" % req.armType)
            print("Not be able to reset the robot home")
            return ResetRobotHomeResponse(False, resetHome_trajectory)
        ##############################################################################################
        

    def rearrange_cylinder_object_callback(self, req):
        rearrange_success, object_manipulation_path = self.rearrange_cylinder_object(req)
        return RearrangeCylinderObjectResponse(rearrange_success, object_manipulation_path)

    def rearrange_cylinder_object(self, req):
        ### given the specified cylinder object and the armType
        rospy.logwarn("PLANNING TO REARRANGE THE OBJECT %s", str(req.object_idx))
        object_path = ObjectRearrangePath()
        transit_traj = []
        transfer_traj = []
        finish_traj = []
        blockPrint()

        curr_object_configPoses = self.planner_p.obtainCurrObjectConfigPoses(self.workspace_p, req.object_idx)

        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        ############################# select the right picking pose until it works #############################
        transit_success = False
        for config_id in range(len(curr_object_configPoses.grasping_configs)):
            configToPickingPose = curr_object_configPoses.grasping_configs[config_id]
            ############## check the collision of the selected configToPickingPose ##############
            self.planner_p.setRobotToConfig(configToPickingPose, self.robot_p, req.armType)
            # isConfigValid, FLAG = self.planner_p.checkConfig_AllCollisions(self.robot_p, self.workspace_p, req.armType)
            isConfigValid, FLAG, objectCollided = self.planner_p.checkConfig_labelCollisions(self.robot_p, self.workspace_p, req.armType)
            if not isConfigValid:
                print("This picking pose is not even valid.")
                print("FLAG: {}, objectCollided: {}".format(FLAG, objectCollided))
                print("Move on to next candidate.")
                continue
            else:
                ### check the connection with neighbors in the roadmap
                print("The picking pose works. Check its neighboring connections.")
                ### when to check the connection of the picking pose, you have to attach the object
                temp_object_curr_pos = self.workspace_p.object_geometries[req.object_idx].curr_pos
                self.planner_p.attachObject(req.object_idx, self.workspace_p, self.robot_p, req.armType)
                connectSuccess, pickingPose_neighbors_idx, pickingPose_neighbors_cost = self.planner_p.connectToNeighbors(
                                            configToPickingPose, self.robot_p, self.workspace_p, req.armType)
                ############## after check, disattach the object and put the object back ##############
                self.planner_p.detachObject(self.workspace_p, self.robot_p, req.armType)
                p.resetBasePositionAndOrientation(
                    self.workspace_p.object_geometries[req.object_idx].geo, 
                    temp_object_curr_pos, [0, 0, 0, 1.0], physicsClientId=self.planningClientID)
                self.workspace_p.object_geometries[req.object_idx].curr_pos = temp_object_curr_pos
                #######################################################################################

                if not connectSuccess:
                    print("This picking pose is not valid, due to no neighboring connections.")
                    print("Move on to next candidate.")
                    continue
                else:
                    print("The picking pose is valid, generate pre-picking")
                    configToPrePickingPose = curr_object_configPoses.approaching_configs[config_id]
                    ############## check the collision of the selected configToPrePickingPose ##############
                    self.planner_p.setRobotToConfig(configToPrePickingPose, self.robot_p, req.armType)
                    # isConfigValid, FLAG = self.planner_p.checkConfig_AllCollisions(self.robot_p, self.workspace_p, req.armType)
                    isConfigValid, FLAG, objectCollided = self.planner_p.checkConfig_labelCollisions(self.robot_p, self.workspace_p, req.armType)
                    if not isConfigValid:
                        print("This pre-picking pose is not even valid. ")
                        print("FLAG: {}, objectCollided: {}".format(FLAG, objectCollided))
                        print("Move on to next candidate.")
                        continue
                    else:
                        ### check the connection with neighbors in the roadmap
                        print("The pre-picking pose works. Check its neighboring connections.")
                        connectSuccess, prePickingPose_neighbors_idx, prePickingPose_neighbors_cost = self.planner_p.connectToNeighbors(
                                    configToPrePickingPose, self.robot_p, self.workspace_p, req.armType)
                        if not connectSuccess:
                            print("This pre-picking pose is not valid, due to no neighboring connections.")
                            print("Move on to next candidate.")
                            continue
                        print("Both picking pose and pre-picking pose are legitimate. Proceed to planning for pre-picking.")
            ###########################################################################################

            ################### plan the path to pre-picking configuration ############################
            connectSuccess, currConfig_neighbors_idx, currConfig_neighbors_cost = self.planner_p.connectToNeighbors(
                        currConfig, self.robot_p, self.workspace_p, req.armType)
            prePicking_traj = self.planner_p.AstarPathFinding(currConfig, configToPrePickingPose, 
                                currConfig_neighbors_idx, currConfig_neighbors_cost, 
                                prePickingPose_neighbors_idx, prePickingPose_neighbors_cost, 
                                self.robot_p, self.workspace_p, req.armType, req.isLabeledRoadmapUsed)   
            ### the planning has been finished, either success or failure
            if prePicking_traj != []:
                print("The transit (pre-picking) path for %s arm is successfully found" % req.armType)
                transit_traj += prePicking_traj
                ################# cartesian path from pre-picking to picking configuration #####################
                currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
                ### you are reaching here since pre-picking has been reached, 
                ### now get the path from pre-picking to picking
                prePickToPickTraj = self.planner_p.generateTrajectory_DirectConfigPath(
                    currConfig, configToPickingPose, self.robot_p, req.armType, self.workspace_p)
                transit_traj += prePickToPickTraj
                #################################################################################################
                transit_success = True
                break
            else:
                print("The transit (pre-picking) path for %s arm is not successfully found" % req.armType)
                print("Move on to next candidate")
                continue
            ###########################################################################################

        if not transit_success:
            print("No picking pose is qualified, either failed (1) picking pose (2) pre-picking pose (3) planning to pre-picking")
            return False, object_path
        
        ### Otherwise, congrats! Transit is successful!
        ######################################### attach the object ###########################################
        ### Now we need to attach the object in hand before transferring the object
        self.planner_p.attachObject(req.object_idx, self.workspace_p, self.robot_p, req.armType)
        #######################################################################################################        

        target_object_configPoses = self.planner_p.position_candidates_configPoses[req.target_position_idx]
        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        ############################# select the right placing pose until it works #############################
        transfer_success = False
        for config_id in range(len(target_object_configPoses.grasping_configs)):
            configToPlacingPose = target_object_configPoses.grasping_configs[config_id]
            ############## check the collision of the selected configToPlacingPose ##############
            self.planner_p.setRobotToConfig(configToPlacingPose, self.robot_p, req.armType)
            # isConfigValid, FLAG = self.planner_p.checkConfig_AllCollisions(self.robot_p, self.workspace_p, req.armType)
            isConfigValid, FLAG, objectCollided = self.planner_p.checkConfig_labelCollisions(self.robot_p, self.workspace_p, req.armType)
            if not isConfigValid:
                print("This placing pose is not even valid.")
                print("FLAG: {}, objectCollided: {}".format(FLAG, objectCollided))
                print("Move on to next candidate.")
                continue
            else:
                ### check the connection with neighbors in the roadmap
                print("The placing pose works. Check its neighboring connections.")
                connectSuccess, placingPose_neighbors_idx, placingPose_neighbors_cost = self.planner_p.connectToNeighbors(
                            configToPlacingPose, self.robot_p, self.workspace_p, req.armType)
                if not connectSuccess:
                    print("This placing pose is not valid, due to no neighboring connections.")
                    print("Move on to next candidate.")
                    continue
                print("The placing pose is legitimate. Proceed to planning for placing.")
            
            ################### plan the path to placing configuration ###################
            placing_traj = self.planner_p.AstarPathFinding(currConfig, configToPlacingPose, 
                            pickingPose_neighbors_idx, pickingPose_neighbors_cost, 
                            placingPose_neighbors_idx, placingPose_neighbors_cost,
                            self.robot_p, self.workspace_p, req.armType, req.isLabeledRoadmapUsed)
            ### the planning has been finished, either success or failure
            if placing_traj != []:
                print("The transfer placing path for %s arm is successfully found" % req.armType)
                transfer_traj += placing_traj
                transfer_success = True
                ### after transferring the object, 
                ### update the object's current position_idx and collision_position_idx
                self.workspace_p.object_geometries[req.object_idx].setCurrPosition(
                    req.target_position_idx, req.target_position_idx)
                break
            else:
                print("The transfer placing path for %s arm is not successfully found" % req.armType)
                print("Move on to next candidate")
                continue
            ############################################################################################
        
        if not transfer_success:
            print("No placing pose is qualified, either failed (1) placing pose (2) planning to placing")
            return False, object_path
        
        ### Otherwise, congrats! Transfer is successful!
        ######################################### detach the object ###########################################
        ### Now we need to detach the object in hand before retracting the object (post-placing)
        self.planner_p.detachObject(self.workspace_p, self.robot_p, req.armType)
        #######################################################################################################
        ############# generate post-placing pose + cartesian move from placing to post-placing ################
        ### The arm leaves the object from ABOVE
        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        placingPose = self.robot_p.getCurrentEEPose(req.armType)
        postPlacingPose = copy.deepcopy(placingPose)
        postPlacingPose[0][2] += 0.05
        isPoseValid, FLAG, configToPostPlacingPose = self.planner_p.generateConfigBasedOnPose(
            postPlacingPose, currConfig, self.robot_p, self.workspace_p, req.armType)
        placeToPostPlaceTraj = self.planner_p.generateTrajectory_DirectConfigPath(
                currConfig, configToPostPlacingPose, self.robot_p, req.armType, self.workspace_p)
        finish_traj += placeToPostPlaceTraj
        ########################################################################################################

        ################################# prepare the path for the object ######################################
        ### get the current state
        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        ### congrat! No problem of rearranging the current object
        ### prepare the object path
        object_path.transit_trajectory = self.generateArmTrajectory(
                                            transit_traj, req.armType, self.robot_p.motomanRJointNames)
        object_path.transfer_trajectory = self.generateArmTrajectory(
                                            transfer_traj, req.armType, self.robot_p.motomanRJointNames)
        object_path.finish_trajectory = self.generateArmTrajectory(
                                            finish_traj, req.armType, self.robot_p.motomanRJointNames)
        object_path.object_idx = req.object_idx
        enablePrint()
        return True, object_path
        ########################################################################################################

    def transit_cylinder_object(self, object_idx, req):
        '''This function plans a transit to a specified object given the current arrangement'''
        rospy.logwarn("PLANNING TO TRANSIT TO THE OBJECT %s", str(object_idx))
        transit_traj = []
        blockPrint()

        curr_object_configPoses = self.planner_p.obtainCurrObjectConfigPoses(self.workspace_p, object_idx)

        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        ############################# select the right picking pose until it works #############################
        transit_success = False
        for config_id in range(len(curr_object_configPoses.grasping_configs)):
            configToPickingPose = curr_object_configPoses.grasping_configs[config_id]
            ############## check the collision of the selected configToPickingPose ##############
            self.planner_p.setRobotToConfig(configToPickingPose, self.robot_p, req.armType)
            # isConfigValid, FLAG = self.planner_p.checkConfig_AllCollisions(self.robot_p, self.workspace_p, req.armType)
            isConfigValid, FLAG, objectCollided = self.planner_p.checkConfig_labelCollisions(self.robot_p, self.workspace_p, req.armType)
            if not isConfigValid:
                print("This picking pose is not even valid.")
                print("FLAG: {}, objectCollided: {}".format(FLAG, objectCollided))
                print("Move on to next candidate.")
                continue
            else:
                ### check the connection with neighbors in the roadmap
                print("The picking pose works. Check its neighboring connections.")
                ### when to check the connection of the picking pose, you have to attach the object
                temp_object_curr_pos = self.workspace_p.object_geometries[object_idx].curr_pos
                self.planner_p.attachObject(object_idx, self.workspace_p, self.robot_p, req.armType)
                connectSuccess, pickingPose_neighbors_idx, pickingPose_neighbors_cost = self.planner_p.connectToNeighbors(
                                            configToPickingPose, self.robot_p, self.workspace_p, req.armType)
                ############## after check, disattach the object and put the object back ##############
                self.planner_p.detachObject(self.workspace_p, self.robot_p, req.armType)
                p.resetBasePositionAndOrientation(
                    self.workspace_p.object_geometries[object_idx].geo, 
                    temp_object_curr_pos, [0, 0, 0, 1.0], physicsClientId=self.planningClientID)
                self.workspace_p.object_geometries[object_idx].curr_pos = temp_object_curr_pos
                #######################################################################################

                if not connectSuccess:
                    print("This picking pose is not valid, due to no neighboring connections.")
                    print("Move on to next candidate.")
                    continue
                else:
                    print("The picking pose is valid, generate pre-picking")
                    configToPrePickingPose = curr_object_configPoses.approaching_configs[config_id]
                    ############## check the collision of the selected configToPrePickingPose ##############
                    self.planner_p.setRobotToConfig(configToPrePickingPose, self.robot_p, req.armType)
                    # isConfigValid, FLAG = self.planner_p.checkConfig_AllCollisions(self.robot_p, self.workspace_p, req.armType)
                    isConfigValid, FLAG, objectCollided = self.planner_p.checkConfig_labelCollisions(self.robot_p, self.workspace_p, req.armType)
                    if not isConfigValid:
                        print("This pre-picking pose is not even valid. ")
                        print("FLAG: {}, objectCollided: {}".format(FLAG, objectCollided))
                        print("Move on to next candidate.")
                        continue
                    else:
                        ### check the connection with neighbors in the roadmap
                        print("The pre-picking pose works. Check its neighboring connections.")
                        connectSuccess, prePickingPose_neighbors_idx, prePickingPose_neighbors_cost = self.planner_p.connectToNeighbors(
                                    configToPrePickingPose, self.robot_p, self.workspace_p, req.armType)
                        if not connectSuccess:
                            print("This pre-picking pose is not valid, due to no neighboring connections.")
                            print("Move on to next candidate.")
                            continue
                        print("Both picking pose and pre-picking pose are legitimate. Proceed to planning for pre-picking.")
            ###########################################################################################
            
            ################### plan the path to pre-picking configuration ############################
            connectSuccess, currConfig_neighbors_idx, currConfig_neighbors_cost = self.planner_p.connectToNeighbors(
                        currConfig, self.robot_p, self.workspace_p, req.armType)
            prePicking_traj = self.planner_p.AstarPathFinding(currConfig, configToPrePickingPose, 
                                currConfig_neighbors_idx, currConfig_neighbors_cost, 
                                prePickingPose_neighbors_idx, prePickingPose_neighbors_cost, 
                                self.robot_p, self.workspace_p, req.armType, req.isLabeledRoadmapUsed)
            ### the planning has been finished, either success or failure
            if prePicking_traj != []:
                print("The transit (pre-picking) path for %s arm is successfully found" % req.armType)
                transit_traj += prePicking_traj
                ################# cartesian path from pre-picking to picking configuration #####################
                currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
                ### you are reaching here since pre-picking has been reached, 
                ### now get the path from pre-picking to picking
                prePickToPickTraj = self.planner_p.generateTrajectory_DirectConfigPath(
                    currConfig, configToPickingPose, self.robot_p, req.armType, self.workspace_p)
                transit_traj += prePickToPickTraj
                #################################################################################################
                transit_success = True
                break
            else:
                print("The transit (pre-picking) path for %s arm is not successfully found" % req.armType)
                print("Move on to next candidate")
                continue
            ###########################################################################################
        
        ### reach here either transit_success == True or None of the option works
        if not transit_success:
            print("No picking pose is qualified, either failed (1) picking pose (2) pre-picking pose (3) planning to pre-picking")
            return transit_success, transit_traj, [], []
        else:
            return transit_success, transit_traj, pickingPose_neighbors_idx, pickingPose_neighbors_cost

    
    def transfer_cylinder_object(self, object_idx, buffer_idx, pickingPose_neighbors_idx, pickingPose_neighbors_cost, req):
        '''This function plans an object transfer to a specified buffer'''
        rospy.logwarn("PLANNING TO TRANSFER OBJECT %s TO A BUFFER", str(object_idx))
        rospy.logwarn("BUFFER: %s", str(buffer_idx))
        transfer_traj = []
        finish_traj = []
        blockPrint()

        ######################################### attach the object ###########################################
        ### Now we need to attach the object in hand before transferring the object
        self.planner_p.attachObject(object_idx, self.workspace_p, self.robot_p, req.armType)
        #######################################################################################################

        target_object_configPoses = self.planner_p.position_candidates_configPoses[buffer_idx]
        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        ############################# select the right placing pose until it works #############################
        transfer_success = False
        for config_id in range(len(target_object_configPoses.grasping_configs)):
            configToPlacingPose = target_object_configPoses.grasping_configs[config_id]
            ############## check the collision of the selected configToPlacingPose ##############
            self.planner_p.setRobotToConfig(configToPlacingPose, self.robot_p, req.armType)
            # isConfigValid, FLAG = self.planner_p.checkConfig_AllCollisions(self.robot_p, self.workspace_p, req.armType)
            isConfigValid, FLAG, objectCollided = self.planner_p.checkConfig_labelCollisions(self.robot_p, self.workspace_p, req.armType)
            if not isConfigValid:
                print("This placing pose is not even valid.")
                print("FLAG: {}, objectCollided: {}".format(FLAG, objectCollided))
                print("Move on to next candidate.")
                continue
            else:
                ### check the connection with neighbors in the roadmap
                print("The placing pose works. Check its neighboring connections.")
                connectSuccess, placingPose_neighbors_idx, placingPose_neighbors_cost = self.planner_p.connectToNeighbors(
                            configToPlacingPose, self.robot_p, self.workspace_p, req.armType)
                if not connectSuccess:
                    print("This placing pose is not valid, due to no neighboring connections.")
                    print("Move on to next candidate.")
                    continue
                print("The placing pose is legitimate. Proceed to planning for placing.")

            ################### plan the path to placing configuration ###################
            placing_traj = self.planner_p.AstarPathFinding(currConfig, configToPlacingPose, 
                            pickingPose_neighbors_idx, pickingPose_neighbors_cost, 
                            placingPose_neighbors_idx, placingPose_neighbors_cost,
                            self.robot_p, self.workspace_p, req.armType, req.isLabeledRoadmapUsed)
            ### the planning has been finished, either success or failure
            if placing_traj != []:
                print("The transfer placing path for %s arm is successfully found" % req.armType)
                transfer_traj += placing_traj
                transfer_success = True
                ### after transferring the object, 
                ### update the object's current position_idx and collision_position_idx
                self.workspace_p.object_geometries[object_idx].setCurrPosition(buffer_idx, buffer_idx)
                break
            else:
                print("The transfer placing path for %s arm is not successfully found" % req.armType)
                print("Move on to next candidate")
                continue
            ############################################################################################
        
        if not transfer_success:
            print("No placing pose is qualified, either failed (1) placing pose (2) planning to placing")
            return transfer_success, transfer_traj, finish_traj
        
        ### Otherwise, congrats! Transfer is successful!
        ######################################### detach the object ###########################################
        ### Now we need to detach the object in hand before retracting the object (post-placing)
        self.planner_p.detachObject(self.workspace_p, self.robot_p, req.armType)
        #######################################################################################################
        ############# generate post-placing pose + cartesian move from placing to post-placing ################
        ### The arm leaves the object from ABOVE
        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        placingPose = self.robot_p.getCurrentEEPose(req.armType)
        postPlacingPose = copy.deepcopy(placingPose)
        postPlacingPose[0][2] += 0.05
        isPoseValid, FLAG, configToPostPlacingPose = self.planner_p.generateConfigBasedOnPose(
            postPlacingPose, currConfig, self.robot_p, self.workspace_p, req.armType)
        placeToPostPlaceTraj = self.planner_p.generateTrajectory_DirectConfigPath(
                currConfig, configToPostPlacingPose, self.robot_p, req.armType, self.workspace_p)
        finish_traj += placeToPostPlaceTraj
        ########################################################################################################
        return transfer_success, transfer_traj, finish_traj


    def generateArmTrajectory(self, traj, armType, motomanRJointNames):
        '''generate arm trajectory (a list of JointState)
        inputs
        ======
            traj (a list of list): a list of joint states [q1, q2, ..., qn]
            armType (string): the arm type (e.g., "Left", "Right_torso)
            motomanRJointNames (a list of strings): the names for controllable joints
        outputs
        =======
            result_traj (ArmTrajectory()): the resulting trajectory (ArmTrajectory object)
        '''
        result_traj = ArmTrajectory()
        result_traj.armType = armType
        if armType == "Left" or armType == "Left_torso":
            first_joint_index = 1
        if armType == "Right" or armType == "Right_torso":
            first_joint_index = 8
        if armType == "Left_torso" or armType == "Right_torso":
            jointNames = [motomanRJointNames[0]] + motomanRJointNames[first_joint_index:first_joint_index+7]
        if armType == "Left" or armType == "Right":
            jointNames = motomanRJointNames[first_joint_index:first_joint_index+7]

        for config in traj:
            joint_state = JointState()
            joint_state.name = jointNames
            joint_state.position = config
            result_traj.trajectory.append(joint_state)
        
        return result_traj

    #########################################################################################
    #########################################################################################

    def rearrange_cylinder_object_legend(self, req):
        ### given the specified cylinder object and the armType
        rospy.logwarn("PLANNING TO REARRANGE THE OBJECT %s", str(req.object_idx))
        object_path = ObjectRearrangePath()
        transit_traj = []
        transfer_traj = []
        finish_traj = []
        blockPrint()

        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        ########################## generate picking pose candidates ###################################
        pickingPose_candidates = self.planner_p.generate_pose_candidates(
            self.workspace_p.object_geometries[req.object_idx].curr_pos, self.workspace_p.cylinder_height)
        ###############################################################################################

        #################### select the right picking pose until it works #############################
        transit_success = False
        for pose_id, pickingPose in enumerate(pickingPose_candidates):
            ######################## check both picking and pre-picking pose ##########################
            isPoseValid, FLAG, configToPickingPose = self.planner_p.generateConfigBasedOnPose(
                                pickingPose, currConfig, self.robot_p, self.workspace_p, req.armType)
            if not isPoseValid:
                print("This picking pose is not even valid. Move on to next candidate.")
                continue
            else:
                ### check the connection with neighbors in the roadmap
                print("The picking pose works. Check its neighboring connections.")
                ### when to check the connection of the picking pose, you have to attach the object
                temp_object_curr_pos = self.workspace_p.object_geometries[req.object_idx].curr_pos
                self.planner_p.attachObject(req.object_idx, self.workspace_p, self.robot_p, req.armType)
                connectSuccess, pickingPose_neighbors_idx, pickingPose_neighbors_cost = self.planner_p.connectToNeighbors(
                                            configToPickingPose, self.robot_p, self.workspace_p, req.armType)
                ############## after check, disattach the object and put the object back ##############
                self.planner_p.detachObject(self.workspace_p, self.robot_p, req.armType)
                p.resetBasePositionAndOrientation(
                    self.workspace_p.object_geometries[req.object_idx].geo, 
                    temp_object_curr_pos, [0, 0, 0, 1.0], physicsClientId=self.planningClientID)
                self.workspace_p.object_geometries[req.object_idx].curr_pos = temp_object_curr_pos
                #######################################################################################

                if not connectSuccess:
                    print("This picking pose is not valid, due to no neighboring connections.")
                    print("Move on to next candidate.")
                    continue
                else:                    
                    print("The picking pose is valid, generate pre-picking")
                    isPoseValid, FLAG, prePickingPose, configToPrePickingPose = \
                        self.planner_p.generatePrePickingPose(
                            pickingPose, currConfig, self.robot_p, self.workspace_p, req.armType)
                    if not isPoseValid:
                        print("The pre-picking pose is not valid. Move on to next candidate.")
                        continue
                    else:
                        ### check the connection with neighbors in the roadmap
                        print("The pre-picking pose works. Check its neighboring connections.")
                        connectSuccess, prePickingPose_neighbors_idx, prePickingPose_neighbors_cost = self.planner_p.connectToNeighbors(
                                    configToPrePickingPose, self.robot_p, self.workspace_p, req.armType)
                        if not connectSuccess:
                            print("This pre-picking pose is not valid, due to no neighboring connections.")
                            print("Move on to next candidate.")
                            continue
                        print("Both picking pose and pre-picking pose are legitimate. Proceed to planning for pre-picking.")
            ###########################################################################################

            ################### plan the path to pre-picking configuration ############################
            connectSuccess, currConfig_neighbors_idx, currConfig_neighbors_cost = self.planner_p.connectToNeighbors(
                        currConfig, self.robot_p, self.workspace_p, req.armType)
            prePicking_traj = self.planner_p.AstarPathFinding(currConfig, configToPrePickingPose, 
                                currConfig_neighbors_idx, currConfig_neighbors_cost, 
                                prePickingPose_neighbors_idx, prePickingPose_neighbors_cost, 
                                self.robot_p, self.workspace_p, req.armType, req.isLabeledRoadmapUsed)
            ### the planning has been finished, either success or failure
            if prePicking_traj != []:
                print("The transit (pre-picking) path for %s arm is successfully found" % req.armType)
                transit_traj += prePicking_traj
                ################# cartesian path from pre-picking to picking configuration #####################
                currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
                ### you are reaching here since pre-picking has been reached, 
                ### now get the path from pre-picking to picking
                prePickToPickTraj = self.planner_p.generateTrajectory_DirectConfigPath(
                    currConfig, configToPickingPose, self.robot_p, req.armType, self.workspace_p)
                transit_traj += prePickToPickTraj
                #################################################################################################
                transit_success = True
                break
            else:
                print("The transit (pre-picking) path for %s arm is not successfully found" % req.armType)
                print("Move on to next candidate")
                continue
            ###########################################################################################
        
        if not transit_success:
            print("No picking pose is qualified, either failed (1) picking pose (2) pre-picking pose (3) planning to pre-picking")
            return False, object_path
        
        ### Otherwise, congrats! Transit is successful!
        ######################################### attach the object ###########################################
        ### Now we need to attach the object in hand before transferring the object
        self.planner_p.attachObject(req.object_idx, self.workspace_p, self.robot_p, req.armType)
        #######################################################################################################

        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        ########################## generate placing pose candidates ###################################
        placingPose_candidates = self.planner_p.generate_pose_candidates(
            self.workspace_p.candidate_geometries[req.target_position_idx].pos, self.workspace_p.cylinder_height)
        ###############################################################################################

        #################### select the right placing pose until it works #############################
        transfer_success = False
        for pose_id, placingPose in enumerate(placingPose_candidates):
            ####################### check the placing pose ###########################
            isPoseValid, FLAG, configToPlacingPose = self.planner_p.generateConfigBasedOnPose(
                    placingPose, currConfig, self.robot_p, self.workspace_p, req.armType)
            if not isPoseValid:
                print("This placing pose is not valid. Move on to next candidate.")
                continue
            else:
                ### check the connection with neighbors in the roadmap
                print("The placing pose works. Check its neighboring connections.")
                connectSuccess, placingPose_neighbors_idx, placingPose_neighbors_cost = self.planner_p.connectToNeighbors(
                            configToPlacingPose, self.robot_p, self.workspace_p, req.armType)
                if not connectSuccess:
                    print("This placing pose is not valid, due to no neighboring connections.")
                    print("Move on to next candidate.")
                    continue
                print("The placing pose is legitimate. Proceed to planning for placing.")
        
            ################### plan the path to placing configuration ###################
            placing_traj = self.planner_p.AstarPathFinding(currConfig, configToPlacingPose, 
                            pickingPose_neighbors_idx, pickingPose_neighbors_cost, 
                            placingPose_neighbors_idx, placingPose_neighbors_cost,
                            self.robot_p, self.workspace_p, req.armType, req.isLabeledRoadmapUsed)
            ### the planning has been finished, either success or failure
            if placing_traj != []:
                print("The transfer placing path for %s arm is successfully found" % req.armType)
                transfer_traj += placing_traj
                transfer_success = True
                ### after transferring the object, 
                ### update the object's current position_idx and collision_position_idx
                self.workspace_p.object_geometries[req.object_idx].setCurrPosition(
                    req.target_position_idx, req.target_position_idx)
                break
            else:
                print("The transfer placing path for %s arm is not successfully found" % req.armType)
                print("Move on to next candidate")
                continue
            ############################################################################################
        
        if not transfer_success:
            print("No placing pose is qualified, either failed (1) placing pose (2) planning to placing")
            return False, object_path

        ### Otherwise, congrats! Transfer is successful!
        ######################################### detach the object ###########################################
        ### Now we need to detach the object in hand before retracting the object (post-placing)
        self.planner_p.detachObject(self.workspace_p, self.robot_p, req.armType)
        #######################################################################################################
        ############# generate post-placing pose + cartesian move from placing to post-placing ################
        ### The arm leaves the object from ABOVE
        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        postPlacingPose = copy.deepcopy(placingPose)
        postPlacingPose[0][2] += 0.05
        isPoseValid, FLAG, configToPostPlacingPose = self.planner_p.generateConfigBasedOnPose(
            postPlacingPose, currConfig, self.robot_p, self.workspace_p, req.armType)
        placeToPostPlaceTraj = self.planner_p.generateTrajectory_DirectConfigPath(
                currConfig, configToPostPlacingPose, self.robot_p, req.armType, self.workspace_p)
        finish_traj += placeToPostPlaceTraj
        ########################################################################################################
        
        ################################# prepare the path for the object ######################################
        ### get the current state
        currConfig = self.robot_p.getRobotCurrSingleArmConfig(req.armType)
        ### congrat! No problem of rearranging the current object
        ### prepare the object path
        object_path.transit_trajectory = self.generateArmTrajectory(
                                            transit_traj, req.armType, self.robot_p.motomanRJointNames)
        object_path.transfer_trajectory = self.generateArmTrajectory(
                                            transfer_traj, req.armType, self.robot_p.motomanRJointNames)
        object_path.finish_trajectory = self.generateArmTrajectory(
                                            finish_traj, req.armType, self.robot_p.motomanRJointNames)
        object_path.object_idx = req.object_idx
        enablePrint()
        return True, object_path
        ########################################################################################################

    def readROSParam(self):
        ### This functions read in needed ROS parameters
        while not rospy.has_param('/motoman_robot/basePosition'):
            rospy.sleep(0.2)
        basePosition = rospy.get_param('/motoman_robot/basePosition')

        while not rospy.has_param('/motoman_robot/baseOrientation'):
            rospy.sleep(0.2)
        baseOrientation = rospy.get_param('/motoman_robot/baseOrientation')

        while not rospy.has_param('/motoman_robot/urdfFile'):
            rospy.sleep(0.2)
        urdfFile = rospy.get_param('/motoman_robot/urdfFile')

        while not rospy.has_param('/motoman_robot/leftArmHomeConfiguration'):
            rospy.sleep(0.2)
        leftArmHomeConfiguration = rospy.get_param('/motoman_robot/leftArmHomeConfiguration')

        while not rospy.has_param('/motoman_robot/rightArmHomeConfiguration'):
            rospy.sleep(0.2)
        rightArmHomeConfiguration = rospy.get_param('/motoman_robot/rightArmHomeConfiguration')

        while not rospy.has_param('/motoman_robot/torsoHomeConfiguration'):
            rospy.sleep(0.2)
        torsoHomeConfiguration = rospy.get_param('/motoman_robot/torsoHomeConfiguration')

        while not rospy.has_param('/workspace_table/standingBase_dim'):
            rospy.sleep(0.2)
        standingBase_dim = rospy.get_param('/workspace_table/standingBase_dim')

        while not rospy.has_param('/workspace_table/table_dim'):
            rospy.sleep(0.2)
        table_dim = rospy.get_param('/workspace_table/table_dim')

        while not rospy.has_param('/workspace_table/table_offset_x'):
            rospy.sleep(0.2)
        table_offset_x = rospy.get_param('/workspace_table/table_offset_x')

        while not rospy.has_param('/uniform_cylinder_object/radius'):
            rospy.sleep(0.2)
        cylinder_radius = rospy.get_param('/uniform_cylinder_object/radius')

        while not rospy.has_param('/uniform_cylinder_object/height'):
            rospy.sleep(0.2)
        cylinder_height = rospy.get_param('/uniform_cylinder_object/height')

        while not rospy.has_param('/object_goal_deployment/discretization_x'):
            rospy.sleep(0.2)
        discretization_x = rospy.get_param('/object_goal_deployment/discretization_x')

        while not rospy.has_param('/object_goal_deployment/discretization_y'):
            rospy.sleep(0.2)
        discretization_y = rospy.get_param('/object_goal_deployment/discretization_y')

        while not rospy.has_param('/object_goal_deployment/object_interval_x'):
            rospy.sleep(0.2)
        object_interval_x = rospy.get_param('/object_goal_deployment/object_interval_x')

        while not rospy.has_param('/object_goal_deployment/object_interval_y'):
            rospy.sleep(0.2)
        object_interval_y = rospy.get_param('/object_goal_deployment/object_interval_y')

        while not rospy.has_param('/object_goal_deployment/side_clearance_x'):
            rospy.sleep(0.2)
        side_clearance_x = rospy.get_param('/object_goal_deployment/side_clearance_x')

        while not rospy.has_param('/object_goal_deployment/side_clearance_y'):
            rospy.sleep(0.2)
        side_clearance_y = rospy.get_param('/object_goal_deployment/side_clearance_y')

        while not rospy.has_param('/constrained_area/ceiling_height'):
            rospy.sleep(0.2)
        ceiling_height = rospy.get_param('/constrained_area/ceiling_height')

        while not rospy.has_param('/constrained_area/thickness_flank'):
            rospy.sleep(0.2)
        thickness_flank = rospy.get_param('/constrained_area/thickness_flank')

        while not rospy.has_param('/object_mesh_to_drop_in_real_scene/object_mesh_path'):
            rospy.sleep(0.2)
        object_mesh_path = rospy.get_param('/object_mesh_to_drop_in_real_scene/object_mesh_path')

        return basePosition, baseOrientation, urdfFile, \
            leftArmHomeConfiguration, rightArmHomeConfiguration, torsoHomeConfiguration, \
            standingBase_dim, table_dim, table_offset_x, \
            cylinder_radius, cylinder_height, \
            discretization_x, discretization_y, \
            object_interval_x, object_interval_y, \
            side_clearance_x, side_clearance_y, \
            ceiling_height, thickness_flank, \
            object_mesh_path


def main(args):
    pybullet_plan_scene = PybulletPlanScene(args)
    pybullet_plan_scene.planner_p.loadSamples()

    pybullet_plan_scene.rosInit()
    rate = rospy.Rate(10) ### 10hz

    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)