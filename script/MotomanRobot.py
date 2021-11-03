#!/usr/bin/env python
from __future__ import division

import pybullet as p
import pybullet_data

import time
import copy
import math
from collections import OrderedDict
import IPython

### This file defines the motoman robot of type sda10f ###

class MotomanRobot(object):

    def __init__(self, 
        urdf_filepath, 
        basePosition, baseOrientation, 
        leftArmHomeConfiguration, rightArmHomeConfiguration, torsoHomeConfiguration,
        isPhysicsTurnOn, server):
        ### get the server
        self.server = server
        ### get the urdf file
        self.urdf_filepath = urdf_filepath

        ### reset the base of motoman
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation

        ### load the urdf of the robot
        if isPhysicsTurnOn == False:
            self.motomanGEO = p.loadURDF(
                    fileName=self.urdf_filepath, 
                    basePosition=self.basePosition, baseOrientation=self.baseOrientation, 
                    useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION, physicsClientId=self.server)
        else:
            self.motomanGEO = p.loadURDF(
                    fileName=self.urdf_filepath, 
                    basePosition=self.basePosition, baseOrientation=self.baseOrientation, 
                    useFixedBase=True, physicsClientId=self.server)

        ### set the robot to the home configuration (both left, right arm and the torso)
        self.leftArmHomeConfiguration = leftArmHomeConfiguration
        self.rightArmHomeConfiguration = rightArmHomeConfiguration
        self.torsoHomeConfiguration = torsoHomeConfiguration
        self.rightHandHomeConfiguration = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        ################### intrinsic value of the motoman_sda10f ###################
        ### joint and end effector information
        ### end-effector index
        self.left_ee_idx = 10 ### left hand ee
        self.right_ee_idx = 20 ### right hand ee
        ### There is a torso joint which connects the lower and upper body (-2.957 ~ 2.957)
        ### But so far we decide to make that torso joint fixed
        ### For each arm, there are 10 joints and 7 of them are revolute joints
        ### There are total 14 revolute joints for each arm
        ### lower limits for null space
        self.ll = [-1.58, \
            -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13, \
            -3.13, -1.90, -2.95, -2.36, -3.13, -1.90, -3.13] +  \
            [0.0, -0.8757, 0.0, 0.0, -0.8757, 0.0]
        ### upper limits for null space
        self.ul = [1.58, \
            3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13, \
            3.13, 1.90, 2.95, 2.36, 3.13, 1.90, 3.13] + \
            [0.8, 0.0, 0.8757, 0.81, 0.0, 0.8757]
        ### joint ranges for null space
        self.jr = [3.16, \
            6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26, \
            6.26, 3.80, 5.90, 4.72, 6.26, 3.80, 6.26] + \
            [0.8, 0.8757, 0.8757, 0.81, 0.8757, 0.8757]
        ### restposes for null space
        # self.rp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.setRestPoses(self.torsoHomeConfiguration, self.leftArmHomeConfiguration, 
                        self.rightArmHomeConfiguration, self.rightHandHomeConfiguration)

        ### get all controllable joints (for motoman, they are all revolute joints)
        self.getRobotJointInfo()

        ### initialize the robot's configuration at its home configuration
        self.updateSingleArmConfig(self.leftArmHomeConfiguration, "Left")
        self.updateSingleArmConfig(self.rightArmHomeConfiguration, "Right")
        self.updateTorsoConfig(self.torsoHomeConfiguration)
        self.updateRightHandConfig(self.rightHandHomeConfiguration)
        self.resetArmConfig(self.leftArmCurrConfiguration + self.rightArmCurrConfiguration)

    def setRestPoses(self, torsoConfiguration, leftArmConfiguration, rightArmConfiguration, rightHandConfiguration):
        self.rp = [torsoConfiguration] + list(leftArmConfiguration) + list(rightArmConfiguration) + list(rightHandConfiguration)

    def updateSingleArmConfig(self, currSingleArmConfig, armType):
        if armType == "Left" or armType == "Left_torso":
            self.leftArmCurrConfiguration = currSingleArmConfig
        if armType == "Right" or armType == "Right_torso":
            self.rightArmCurrConfiguration = currSingleArmConfig
    
    def updateTorsoConfig(self, torsoConfig):
        self.torsoCurrConfiguration = torsoConfig

    def updateRightHandConfig(self, currRightHandConfig):
        self.rightHandCurrConfiguration = currRightHandConfig

    def keepTorsoConfig(self):
        joint_index = 0
        p.resetJointState(self.motomanGEO, joint_index, self.torsoCurrConfiguration, physicsClientId=self.server)

    def keepCurrRightHandConfig(self):
        joint_idx = 21 ### left_out_knuckle_joint (finger_joint)
        p.resetJointState(self.motomanGEO, joint_idx, self.rightHandCurrConfiguration[0], physicsClientId=self.server)
        joint_idx = 23 ### left_inner_finger_joint
        p.resetJointState(self.motomanGEO, joint_idx, self.rightHandCurrConfiguration[1], physicsClientId=self.server)
        joint_idx = 25 ### left_inner_knuckle_joint
        p.resetJointState(self.motomanGEO, joint_idx, self.rightHandCurrConfiguration[2], physicsClientId=self.server)
        joint_idx = 26 ### right_out_knuckle_joint
        p.resetJointState(self.motomanGEO, joint_idx, self.rightHandCurrConfiguration[3], physicsClientId=self.server)
        joint_idx = 28 ### right_inner_finger_joint
        p.resetJointState(self.motomanGEO, joint_idx, self.rightHandCurrConfiguration[4], physicsClientId=self.server)
        joint_idx = 30 ### left_inner_knuckle_joint
        p.resetJointState(self.motomanGEO, joint_idx, self.rightHandCurrConfiguration[5], physicsClientId=self.server)

    def resetRightHandConfig(self, resetHandConfig):
        joint_idx = 21 ### left_out_knuckle_joint (finger_joint)
        p.resetJointState(self.motomanGEO, joint_idx, resetHandConfig[0], physicsClientId=self.server)
        joint_idx = 23 ### left_inner_finger_joint
        p.resetJointState(self.motomanGEO, joint_idx, resetHandConfig[1], physicsClientId=self.server)
        joint_idx = 25 ### left_inner_knuckle_joint
        p.resetJointState(self.motomanGEO, joint_idx, resetHandConfig[2], physicsClientId=self.server)
        joint_idx = 26 ### right_out_knuckle_joint
        p.resetJointState(self.motomanGEO, joint_idx, resetHandConfig[3], physicsClientId=self.server)
        joint_idx = 28 ### right_inner_finger_joint
        p.resetJointState(self.motomanGEO, joint_idx, resetHandConfig[4], physicsClientId=self.server)
        joint_idx = 30 ### left_inner_knuckle_joint
        p.resetJointState(self.motomanGEO, joint_idx, resetHandConfig[5], physicsClientId=self.server)

        p.stepSimulation(physicsClientId=self.server)
        self.updateRightHandConfig(resetHandConfig)


    def resetArmConfig(self, resetArmConfiguration):
        for j in range(1, 8):
            p.resetJointState(self.motomanGEO, j, resetArmConfiguration[j-1], physicsClientId=self.server)
        for j in range(11, 18):
            p.resetJointState(self.motomanGEO, j, resetArmConfiguration[j-4], physicsClientId=self.server)
        self.updateSingleArmConfig(resetArmConfiguration[0:7], "Left")
        self.updateSingleArmConfig(resetArmConfiguration[7:14], "Right")
        ### keep the right hand/gripper and torso unchanged
        self.keepCurrRightHandConfig()
        self.keepTorsoConfig()
        p.stepSimulation(physicsClientId=self.server)
        left_ee_pos_quat = p.getLinkState(self.motomanGEO, self.left_ee_idx, physicsClientId=self.server)
        self.left_ee_pose = [list(left_ee_pos_quat[0]), list(left_ee_pos_quat[1])]
        right_ee_pos_quat = p.getLinkState(self.motomanGEO, self.right_ee_idx, physicsClientId=self.server)
        self.right_ee_pose = [list(right_ee_pos_quat[0]), list(right_ee_pos_quat[1])]
        self.setRestPoses(
            self.torsoCurrConfiguration, resetArmConfiguration[0:7], resetArmConfiguration[7:14], self.rightHandCurrConfiguration)

    def resetArmConfig_torso(self, resetArmConfiguration, resetTorsoConfiguration):
        p.resetJointState(self.motomanGEO, 0, resetTorsoConfiguration, physicsClientId=self.server)
        self.updateTorsoConfig(resetTorsoConfiguration)
        for j in range(1, 8):
            p.resetJointState(self.motomanGEO, j, resetArmConfiguration[j-1], physicsClientId=self.server)
        for j in range(11, 18):
            p.resetJointState(self.motomanGEO, j, resetArmConfiguration[j-4], physicsClientId=self.server)
        self.updateSingleArmConfig(resetArmConfiguration[0:7], "Left")
        self.updateSingleArmConfig(resetArmConfiguration[7:14], "Right")
        ### keep the right hand unchanged
        self.keepCurrRightHandConfig()
        p.stepSimulation(physicsClientId=self.server)
        left_ee_pos_quat = p.getLinkState(self.motomanGEO, self.left_ee_idx, physicsClientId=self.server)
        self.left_ee_pose = [list(left_ee_pos_quat[0]), list(left_ee_pos_quat[1])]
        right_ee_pos_quat = p.getLinkState(self.motomanGEO, self.right_ee_idx, physicsClientId=self.server)
        self.right_ee_pose = [list(right_ee_pos_quat[0]), list(right_ee_pos_quat[1])]
        self.setRestPoses(
            resetTorsoConfiguration, resetArmConfiguration[0:7], resetArmConfiguration[7:14], self.rightHandCurrConfiguration)

    def setSingleArmToConfig(self, singleArmConfig, armType):
        ### the planning version of moveSingArm
        if armType == "Left":
            for j in range(1, 8):
                p.resetJointState(self.motomanGEO, j, singleArmConfig[j-1], physicsClientId=self.server)
            self.updateSingleArmConfig(singleArmConfig, armType)
            for j in range(11, 18):
                p.resetJointState(self.motomanGEO, j, self.rightArmCurrConfiguration[j-11], physicsClientId=self.server)
        if armType == "Right":
            for j in range(1, 8):
                p.resetJointState(self.motomanGEO, j, self.leftArmCurrConfiguration[j-1], physicsClientId=self.server)
            for j in range(11, 18):
                p.resetJointState(self.motomanGEO, j, singleArmConfig[j-11], physicsClientId=self.server)
            self.updateSingleArmConfig(singleArmConfig, armType)
        ### keep the right hand/gripper and torso unchanged
        self.keepCurrRightHandConfig()
        self.keepTorsoConfig()
        p.stepSimulation(physicsClientId=self.server)
        left_ee_pos_quat = p.getLinkState(self.motomanGEO, self.left_ee_idx, physicsClientId=self.server)
        self.left_ee_pose = [list(left_ee_pos_quat[0]), list(left_ee_pos_quat[1])]
        right_ee_pos_quat = p.getLinkState(self.motomanGEO, self.right_ee_idx, physicsClientId=self.server)
        self.right_ee_pose = [list(right_ee_pos_quat[0]), list(right_ee_pos_quat[1])]
        if armType == "Left":
            self.setRestPoses(
                self.torsoCurrConfiguration, singleArmConfig, self.rightArmCurrConfiguration, self.rightHandCurrConfiguration)
        if armType == "Right":
            self.setRestPoses(
                self.torsoCurrConfiguration, self.leftArmCurrConfiguration, singleArmConfig, self.rightHandCurrConfiguration)

    def setSingleArmToConfig_torso(self, singleArmConfig, torsoConfig, armType):
        ### the planning version of moveSingArm
        p.resetJointState(self.motomanGEO, 0, torsoConfig, physicsClientId=self.server)
        self.updateTorsoConfig(torsoConfig)
        if armType == "Left_torso":
            for j in range(1, 8):
                p.resetJointState(self.motomanGEO, j, singleArmConfig[j-1], physicsClientId=self.server)
            self.updateSingleArmConfig(singleArmConfig, "Left")
            for j in range(11, 18):
                p.resetJointState(self.motomanGEO, j, self.rightArmCurrConfiguration[j-11], physicsClientId=self.server)
        if armType == "Right_torso":
            for j in range(1, 8):
                p.resetJointState(self.motomanGEO, j, self.leftArmCurrConfiguration[j-1], physicsClientId=self.server)
            for j in range(11, 18):
                p.resetJointState(self.motomanGEO, j, singleArmConfig[j-11], physicsClientId=self.server)
            self.updateSingleArmConfig(singleArmConfig, "Right")
        ### keep the right hand/gripper unchanged
        self.keepCurrRightHandConfig()
        p.stepSimulation(physicsClientId=self.server)
        left_ee_pos_quat = p.getLinkState(self.motomanGEO, self.left_ee_idx, physicsClientId=self.server)
        self.left_ee_pose = [list(left_ee_pos_quat[0]), list(left_ee_pos_quat[1])]
        right_ee_pos_quat = p.getLinkState(self.motomanGEO, self.right_ee_idx, physicsClientId=self.server)
        self.right_ee_pose = [list(right_ee_pos_quat[0]), list(right_ee_pos_quat[1])]
        if armType == "Left_torso":
            self.setRestPoses(
                torsoConfig, singleArmConfig, self.rightArmCurrConfiguration, self.rightHandCurrConfiguration)
        if armType == "Right_torso":
            self.setRestPoses(
                torsoConfig, self.leftArmCurrConfiguration, singleArmConfig, self.rightHandCurrConfiguration)

    def resetRobotToHomeConfiguration(self):
        ### this function reset the robot to home configuration (torso, arms, hand)
        self.resetArmConfig_torso(self.leftArmHomeConfiguration+self.rightArmHomeConfiguration, self.torsoHomeConfiguration)
        self.resetRightHandConfig(self.rightHandHomeConfiguration)


    def moveSingleArm(self, singleArmConfig, armType):
        ### move single arm by resetJointState() function
        if armType == "Left":
            for j in range(1, 8):
                p.resetJointState(self.motomanGEO, j, singleArmConfig[j-1], physicsClientId=self.server)
            self.updateSingleArmConfig(singleArmConfig, armType)
            for j in range(11, 18):
                p.resetJointState(self.motomanGEO, j, self.rightArmCurrConfiguration[j-11], physicsClientId=self.server)
        if armType == "Right":
            for j in range(1, 8):
                p.resetJointState(self.motomanGEO, j, self.leftArmCurrConfiguration[j-1], physicsClientId=self.server)
            for j in range(11, 18):
                p.resetJointState(self.motomanGEO, j, singleArmConfig[j-11], physicsClientId=self.server)
            self.updateSingleArmConfig(singleArmConfig, armType)
        ### keep the right hand/gripper and torso unchanged
        self.keepCurrRightHandConfig()
        self.keepTorsoConfig()
        ###### This function is in maintenance ###### 
        # p.stepSimulation(physicsClientId=self.server)
        left_ee_pos_quat = p.getLinkState(self.motomanGEO, self.left_ee_idx, physicsClientId=self.server)
        self.left_ee_pose = [list(left_ee_pos_quat[0]), list(left_ee_pos_quat[1])]
        right_ee_pos_quat = p.getLinkState(self.motomanGEO, self.right_ee_idx, physicsClientId=self.server)
        self.right_ee_pose = [list(right_ee_pos_quat[0]), list(right_ee_pos_quat[1])]


    def moveSingleArm_torso(self, singleArmConfig, torsoConfig, armType):
        ### move single arm by resetJointState() function
        p.resetJointState(self.motomanGEO, 0, torsoConfig, physicsClientId=self.server)
        self.updateTorsoConfig(torsoConfig)
        if armType == "Left_torso":
            for j in range(1, 8):
                p.resetJointState(self.motomanGEO, j, singleArmConfiguration[j-1], physicsClientId=self.server)
            self.updateSingleArmConfig(singleArmConfiguration, "Left")
            for j in range(11, 18):
                p.resetJointState(self.motomanGEO, j, self.rightArmCurrConfiguration[j-11], physicsClientId=self.server)
        if armType == "Right_torso":
            for j in range(1, 8):
                p.resetJointState(self.motomanGEO, j, self.leftArmCurrConfiguration[j-1], physicsClientId=self.server)
            for j in range(11, 18):
                p.resetJointState(self.motomanGEO, j, singleArmConfig[j-11], physicsClientId=self.server)
            self.updateSingleArmConfig(singleArmConfig, "Right")
        ### keep the right hand/gripper unchanged
        self.keepCurrRightHandConfig()
        ###### This function is in maintenance ###### 
        # p.stepSimulation(physicsClientId=self.server)
        left_ee_pos_quat = p.getLinkState(self.motomanGEO, self.left_ee_idx, physicsClientId=self.server)
        self.left_ee_pose = [list(left_ee_pos_quat[0]), list(left_ee_pos_quat[1])]
        right_ee_pos_quat = p.getLinkState(self.motomanGEO, self.right_ee_idx, physicsClientId=self.server)
        self.right_ee_pose = [list(right_ee_pos_quat[0]), list(right_ee_pos_quat[1])]

    def getRobotCurrConfig(self):
        ### this function return the current full configuration of the robot 
        ### joints [1(torso) + 7(leftArm) + 7(rightArm) + 6(rightHand)]
        return [self.torsoCurrConfiguration] + self.leftArmCurrConfiguration + \
                    self.rightArmCurrConfiguration + self.rightHandCurrConfiguration
    
    def getRobotCurrSingleArmConfig(self, armType):
        ### this functions return the current single arm configuration of the robot
        ### (i/ii) (torso)+left arm (iii/iv) (torso)+right arm
        if armType == "Left":
            currSingleArmConfig = copy.deepcopy(self.leftArmCurrConfiguration)
        if armType == "Left_torso":
            currSingleArmConfig = copy.deepcopy([self.torsoCurrConfiguration] + self.leftArmCurrConfiguration)
        if armType == "Right":
            currSingleArmConfig = copy.deepcopy(self.rightArmCurrConfiguration)
        if armType == "Right_torso":
            currSingleArmConfig = copy.deepcopy([self.torsoCurrConfiguration] + self.rightArmCurrConfiguration)
        return currSingleArmConfig

    def getCurrentEEPose(self, armType):
        if armType == "Left" or armType == "Left_torso":
            ee_pose = copy.deepcopy(self.left_ee_pose)
        if armType == "Right" or armType == "Right_torso":
            ee_pose = copy.deepcopy(self.right_ee_pose)
        return ee_pose

    def getJointState(self):
        torso_joint_state = p.getJointStates(
                bodyUniqueId=self.motomanGEO, jointIndices=range(0, 1), physicsClientId=self.server)
        left_joint_state = p.getJointStates(
                bodyUniqueId=self.motomanGEO, jointIndices=range(1, 8), physicsClientId=self.server)
        right_joint_state = p.getJointStates(
                bodyUniqueId=self.motomanGEO, jointIndices=range(11, 18), physicsClientId=self.server)
        temp_left_arm_config = []
        temp_right_arm_config = []
        for joint in left_joint_state:
            temp_left_arm_config.append(joint[0])
        for joint in right_joint_state:
            temp_right_arm_config.append(joint[0])

        self.updateSingleArmConfig(temp_left_arm_config, "Left")
        self.updateSingleArmConfig(temp_right_arm_config, "Right")
        self.updateTorsoConfig(torso_joint_state[0][0])
        dualArmCurrConfiguration = list(self.leftArmCurrConfiguration) + list(self.rightArmCurrConfiguration)

        return self.motomanRJointNames, dualArmCurrConfiguration


    def getRobotJointInfo(self):
        ################# information related to Motoman arm (joint info) #################
        ### output: self.motomanRJointNames [torso joint, 7 left joints, 7 right joints, hand joints]        
        self.motomanRJointNames = []
        num_joints = p.getNumJoints(self.motomanGEO, self.server)
        for i in range(num_joints):
            jointInfo = p.getJointInfo(self.motomanGEO, i, self.server)
            # print(jointInfo)
            if jointInfo[2] == 0:
                ### only get revolute joint
                self.motomanRJointNames.append(jointInfo[1].decode())
        # self.printRJointNames()


    def printRJointNames(self):
        for name in self.motomanRJointNames:
            print(name)




