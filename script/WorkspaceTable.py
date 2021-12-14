#!/usr/bin/env python
from __future__ import division

import pybullet as p
import pybullet_data

from collections import OrderedDict
import os
import math
import numpy as np
import time
import IPython
import random

from CollisionChecker import CollisionChecker
import utils

from confined_space_rearrangement.msg import CylinderObj

### This file defines workspace for the table ###

class WorkspaceTable(object):
    def __init__(self, 
        rosPackagePath, robotBasePosition,
        standingBase_dim, table_dim, table_offset_x, 
        mesh_path, isPhysicsTurnOn, server):
        ### get the server
        self.server = server
        self.rosPackagePath = rosPackagePath
        self.mesh_path = mesh_path
        self.known_geometries = []
        self.createRobotStandingBase(robotBasePosition, standingBase_dim, isPhysicsTurnOn)
        self.createTableScene(robotBasePosition, table_dim, table_offset_x, isPhysicsTurnOn)
        self.table_offset_x = table_offset_x
        self.collisionAgent = CollisionChecker(self.server)

        ### RED, GREEN, BLUE, YELLOW, MAGENTA,
        ### ORANGE, BROWN, BLACK, GREY,
        ### CYAN, PURPLE, LIME,
        ### MAROON, OLIVE, TEAL, NAVY,
        ### PINK, APRICOT, BEIGE,
        ### MINT, LAVENDER
        ### (21 colors up to 21 objects)

        # self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
        #                 [0.96, 0.51, 0.19], [0.604, 0.388, 0.141], [0, 0, 0], [0.66, 0.66, 0.66],
        #                 [0.259, 0.831, 0.957], [0.567, 0.118, 0.706], [0.749, 0.937, 0.271],
        #                 [0.502, 0, 0], [0.502, 0.502, 0], [0.275, 0.6, 0.565], [0, 0, 0.459],
        #                 [0.98, 0.745, 0.831], [1.0, 0.847, 0.694], [1.0, 0.98, 0.784],
        #                 [0.667, 1.0, 0.765], [0.863, 0.745, 1.0]]

        # self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19], 
        #                     [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19],
        #                     [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19]]

    def createRobotStandingBase(self, robotBasePosition, standingBase_dim, isPhysicsTurnOn):
        ################ create the known geometries - standingBase  ####################
        self.standingBase_dim = np.array(standingBase_dim)
        self.standingBasePosition = [
            robotBasePosition[0], robotBasePosition[1], robotBasePosition[2]-self.standingBase_dim[2]/2-0.005]
        self.standingBase_c = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                    halfExtents=self.standingBase_dim/2, physicsClientId=self.server)
        self.standingBase_v = p.createVisualShape(shapeType=p.GEOM_BOX,
                                    halfExtents=self.standingBase_dim/2, physicsClientId=self.server)
        if isPhysicsTurnOn == True:
            self.standingBaseM = p.createMultiBody(
                baseCollisionShapeIndex=self.standingBase_c, baseVisualShapeIndex=self.standingBase_v,
                basePosition=self.standingBasePosition, physicsClientId=self.server)
        else:
            self.standingBaseM = p.createMultiBody(
                baseCollisionShapeIndex=self.standingBase_c, baseVisualShapeIndex=self.standingBase_v,
                basePosition=self.standingBasePosition, physicsClientId=self.server)            
        print("standing base: " + str(self.standingBaseM))
        self.known_geometries.append(self.standingBaseM)
        #################################################################################

    def createTableScene(self, 
        robotBasePosition, table_dim, table_offset_x, isPhysicsTurnOn):

        ################ create the known geometries - table  ###########################
        print("---------Enter to table scene!----------")
        # self.table_dim = np.array([table_dim[0], table_dim[1], table_dim[2]+self.standingBase_dim[2]+0.005])
        self.table_dim = np.array([table_dim[0], table_dim[1], table_dim[2]])
        self.tablePosition = [
            table_offset_x+self.table_dim[0]/2, robotBasePosition[1], robotBasePosition[2]+(self.table_dim[2]/2-self.standingBase_dim[2]-0.005)]

        self.table_c = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                halfExtents=self.table_dim/2, physicsClientId=self.server)
        self.table_v = p.createVisualShape(shapeType=p.GEOM_BOX,
                                halfExtents=self.table_dim/2, physicsClientId=self.server)
        self.tableM = p.createMultiBody(baseCollisionShapeIndex=self.table_c, baseVisualShapeIndex=self.table_v,
                                            basePosition=self.tablePosition, physicsClientId=self.server)
        print("table: " + str(self.tableM))
        self.known_geometries.append(self.tableM)

        print("inner edge of the table: ")
        print(self.tablePosition[0]-table_dim[0]/2)
        print("table surface: ")
        print(self.tablePosition[2]+table_dim[2]/2)
        print("left side of the table: ")
        print(self.tablePosition[1]+table_dim[1]/2)
        print("right side of the table: ")
        print(self.tablePosition[1]-table_dim[1]/2)
        #################################################################################

    def addConstrainedArea(self, ceiling_height, thickness_flank):
        self.ceiling_height = ceiling_height
        self.thickness_flank = thickness_flank
        print("---------Add constrained area!----------")
        ################ add a constrained area - a transparent box  ####################
        self.flankColor = [111/255.0, 78/255.0, 55/255.0, 0.3]
        self.sideFlank_dim = np.array([self.table_dim[0], thickness_flank, ceiling_height])
        self.backFlank_dim = np.array([thickness_flank, self.table_dim[1]-thickness_flank*2, ceiling_height])
        self.topFlank_dim = np.array([self.table_dim[0], self.table_dim[1], thickness_flank])
        self.sideFlank_c = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                    halfExtents=self.sideFlank_dim/2, physicsClientId=self.server)
        self.sideFlank_v = p.createVisualShape(shapeType=p.GEOM_BOX,
                            halfExtents=self.sideFlank_dim/2, rgbaColor=self.flankColor, physicsClientId=self.server)
        self.leftFlankPosition = [self.tablePosition[0],
                                  self.tablePosition[1] + self.table_dim[1] / 2 - thickness_flank / 2,
                                  self.tablePosition[2] + self.table_dim[2] / 2 + self.sideFlank_dim[2] / 2]
        self.rightFlankPosition = [self.tablePosition[0],
                                  self.tablePosition[1] - self.table_dim[1] / 2 + thickness_flank / 2,
                                  self.tablePosition[2] + self.table_dim[2] / 2 + self.sideFlank_dim[2] / 2]
        self.leftFlankM = p.createMultiBody(
            baseCollisionShapeIndex=self.sideFlank_c, baseVisualShapeIndex=self.sideFlank_v,
            basePosition=self.leftFlankPosition, physicsClientId=self.server)
        self.rightFlankM = p.createMultiBody(
            baseCollisionShapeIndex=self.sideFlank_c, baseVisualShapeIndex=self.sideFlank_v,
            basePosition=self.rightFlankPosition, physicsClientId=self.server)
        self.known_geometries.append(self.leftFlankM)
        self.known_geometries.append(self.rightFlankM)

        self.backFlank_c = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                    halfExtents=self.backFlank_dim/2, physicsClientId=self.server)
        self.backFlank_v = p.createVisualShape(shapeType=p.GEOM_BOX,
                            halfExtents=self.backFlank_dim/2, rgbaColor=self.flankColor, physicsClientId=self.server)
        self.backFlankPosition = [self.tablePosition[0] + self.table_dim[0] / 2 - thickness_flank / 2,
                                  self.tablePosition[1], self.tablePosition[2] + self.table_dim[2] / 2 + self.backFlank_dim[2] / 2]
        self.backFlankM = p.createMultiBody(
            baseCollisionShapeIndex=self.backFlank_c, baseVisualShapeIndex=self.backFlank_v,
            basePosition=self.backFlankPosition, physicsClientId=self.server)
        self.known_geometries.append(self.backFlankM)

        self.topFlank_c = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                    halfExtents=self.topFlank_dim/2, physicsClientId=self.server)
        self.topFlank_v = p.createVisualShape(shapeType=p.GEOM_BOX,
                            halfExtents=self.topFlank_dim/2, rgbaColor=self.flankColor, physicsClientId=self.server)
        self.topFlankPosition = [self.tablePosition[0], self.tablePosition[1], 
                                 self.tablePosition[2] + self.table_dim[2] / 2 + ceiling_height + thickness_flank / 2]
        self.topFlankM = p.createMultiBody(
            baseCollisionShapeIndex=self.topFlank_c, baseVisualShapeIndex=self.topFlank_v,
            basePosition=self.topFlankPosition, physicsClientId=self.server)
        self.known_geometries.append(self.topFlankM)

        self.constrained_area_center = [self.tablePosition[0]-thickness_flank/2, 0.0]
        self.constrained_area_x_limit = [self.tablePosition[0]-self.table_dim[0]/2, \
                                         self.tablePosition[0]+self.table_dim[0]/2-thickness_flank]
        self.constrained_area_y_limit = [self.tablePosition[1]-self.table_dim[1]/2+thickness_flank, \
                                         self.tablePosition[1]+self.table_dim[1]/2-thickness_flank]
        self.constrained_area_dim = [self.constrained_area_x_limit[1]-self.constrained_area_x_limit[0], \
                                     self.constrained_area_y_limit[1]-self.constrained_area_y_limit[0]]

        print("left flank: " + str(self.leftFlankM))
        print("right flank: " + str(self.rightFlankM))
        print("back flank: " + str(self.backFlankM))
        print("top flank: " + str(self.topFlankM))
        # print("constrained_area_center: ", str(self.constrained_area_center))
        # print("constrained_area_x_limit", str(self.constrained_area_x_limit))
        # print("constrained_area_y_limit", str(self.constrained_area_y_limit))
        # print("constrained_area_dim", str(self.constrained_area_dim))
        ###########################################################################################

    def setDeploymentParam(self, 
            cylinder_radius, cylinder_height, side_clearance_x, side_clearance_y, 
            discretization_x=None, discretization_y=None, object_interval_x=None, object_interval_y=None):
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height
        self.side_clearance_x = side_clearance_x
        self.side_clearance_y = side_clearance_y
        self.discretization_x = discretization_x
        self.discretization_y = discretization_y
        self.object_interval_x = object_interval_x
        self.object_interval_y = object_interval_y

    def generateInstance_cylinders(self, num_objects):
        ### return: success (bool)
        self.num_objects = num_objects ### obtain the number of objects

        if self.num_objects == 6:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if self.num_objects == 8:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
        if self.num_objects == 10:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
        if self.num_objects == 12:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19],
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19]]


        self.object_geometries = OrderedDict()
        print("--------generate an instance---------")
        cylinder_c = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                radius=self.cylinder_radius, height=self.cylinder_height, 
                                                physicsClientId=self.server)
        for obj_i in range(self.num_objects):
            rgbacolor = self.color_pools[obj_i] + [1.0] ### No transparency
            cylinder_v = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                                radius=self.cylinder_radius, length=self.cylinder_height, 
                                                rgbaColor=rgbacolor, physicsClientId=self.server)
            isCollision = True
            timeout = 20
            while isCollision and timeout > 0:
                print("generate the {}th object".format(obj_i))
                timeout -= 1
                start_pos = [
                    round(random.uniform(self.constrained_area_x_limit[0] + self.cylinder_radius, \
                            self.constrained_area_x_limit[1] - self.side_clearance_x - self.cylinder_radius), 3),
                    round(random.uniform(self.constrained_area_y_limit[0] + self.side_clearance_y + self.cylinder_radius, \
                            self.constrained_area_y_limit[1] - self.side_clearance_y - self.cylinder_radius), 3),
                    round(self.tablePosition[2] + self.table_dim[2] / 2 + self.cylinder_height / 2, 3)
                ]
                cylinder_objectM = p.createMultiBody(
                            baseCollisionShapeIndex=cylinder_c, baseVisualShapeIndex=cylinder_v,
                            basePosition=start_pos, physicsClientId=self.server)
                isCollision = self.collisionAgent.collisionCheck_instance_cylinder_objects(
                    cylinder_objectM, [obj_geo.geo for obj_geo in self.object_geometries.values()], self.cylinder_radius)
                if isCollision:
                    ### remove that mesh as it is invalid (collision with existing object meshes)
                    p.removeBody(cylinder_objectM)
            if isCollision:
                ### could not fit in this object, the instance generation fails...
                return False
            else:
                ### congrats, the object's location is accepted
                self.object_geometries[obj_i] = CylinderObject(
                    obj_i, start_pos, cylinder_objectM, self.cylinder_radius, self.cylinder_height)

        ############## print to confirm ##############
        for obj_idx in range(self.num_objects):
            print(obj_idx)
            print(self.object_geometries[obj_idx].curr_pos)
        ##############################################
        return True


    def reproduceInstance_cylinders(self, cylinder_objects):
        ### input: cylinder_objects (CylinderObj[])
        ### output: success (bool)
        print("--------reproduce an instance---------")
        self.num_objects = len(cylinder_objects)

        if self.num_objects == 6:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if self.num_objects == 8:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
        if self.num_objects == 10:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
        if self.num_objects == 12:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19],
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19]]

        ### first assign goal positions
        self.object_geometries = OrderedDict()
        self.initial_object_position_meshes = OrderedDict() ### key: position_idx, value: mesh id
        self.assignGoalPositions()
        self.object_initial_infos = OrderedDict()
        self.goal_visualization_mesh = OrderedDict() ### obj_idx (key): mesh (value)

        counter = 0
        for cylinder_obj in cylinder_objects:
            obj_idx = cylinder_obj.obj_idx
            cylinder_curr_position = [cylinder_obj.curr_position.x, cylinder_obj.curr_position.y, cylinder_obj.curr_position.z]
            cylinder_radius = round(cylinder_obj.radius, 3)
            cylinder_height = round(cylinder_obj.height, 3)
            cylinder_rgbacolor_start = self.color_pools[obj_idx] + [1.0] ### concrete color

            ############################################ generate object current mesh ##################################################
            cylinder_c = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                radius=cylinder_radius, height=cylinder_height, physicsClientId=self.server)
            cylinder_v = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                radius=cylinder_radius, length=cylinder_height, rgbaColor=cylinder_rgbacolor_start, physicsClientId=self.server)
            cylinder_objectM = p.createMultiBody(baseCollisionShapeIndex=cylinder_c, baseVisualShapeIndex=cylinder_v,
                basePosition=cylinder_curr_position, physicsClientId=self.server)
            ####### create a copy of collision meshes for the start positions #######
            cylinder_c_copy = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                radius=cylinder_radius, height=cylinder_height, physicsClientId=self.server)
            cylinder_rgbacolor_start_copy = self.color_pools[obj_idx] + [0.0] ### transparent color
            cylinder_v_copy = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                radius=cylinder_radius, length=cylinder_height, rgbaColor=cylinder_rgbacolor_start_copy, physicsClientId=self.server)
            cylinder_objectM_copy = p.createMultiBody(baseCollisionShapeIndex=cylinder_c_copy, baseVisualShapeIndex=cylinder_v_copy,
                basePosition=cylinder_curr_position, physicsClientId=self.server)
            self.initial_object_position_meshes[self.num_candidates+counter] = cylinder_objectM_copy
            #########################################################################
            self.object_geometries[obj_idx] = CylinderObject(
                                    obj_idx, cylinder_curr_position, cylinder_objectM, cylinder_radius, cylinder_height)
            collision_position_idx = self.assignToNearestCandiate(cylinder_curr_position)
            self.object_geometries[obj_idx].setCurrPosition(self.num_candidates+counter, collision_position_idx)
            #############################################################################################################################

            ######################################### record object initial position ####################################################
            self.object_initial_infos[obj_idx] = objectInitialInfo(
                                        obj_idx, cylinder_curr_position, self.num_candidates+counter, collision_position_idx)
            #############################################################################################################################

            ######################################### generate goal visualization mesh ##################################################
            goal_position_idx = self.all_goal_positions[obj_idx]
            goal_position = self.candidate_geometries[goal_position_idx].pos
            self.object_geometries[obj_idx].setGoalPosition(goal_position, goal_position_idx)
            cylinder_rgbacolor_goal = self.color_pools[obj_idx] + [0.35] ### transparent color
            cylinder_v = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                radius=cylinder_radius, length=cylinder_height, rgbaColor=cylinder_rgbacolor_goal, physicsClientId=self.server)
            cylinder_objectM = p.createMultiBody(baseVisualShapeIndex=cylinder_v,
                            basePosition=goal_position, physicsClientId=self.server) ### only visualization
            self.goal_visualization_mesh[obj_idx] = cylinder_objectM
            #############################################################################################################################
            counter += 1
        
        ### get the arrangement task
        self.initial_arrangement = []
        self.final_arrangement = []
        for obj_idx, object_info in self.object_geometries.items():
            self.initial_arrangement.append(object_info.curr_position_idx)
            self.final_arrangement.append(object_info.goal_position_idx)

        ###### print test ######
        # for obj_idx, object_info in self.object_geometries.items():
        #     print(obj_idx)
        #     print("object_index: " + str(object_info.object_index))
        #     print("cylinder_radius: " + str(object_info.cylinder_radius))
        #     print("cylinder_height: " + str(object_info.cylinder_height))
        #     print("geo: " + str(object_info.geo))
        #     print("curr_pos: " + str(object_info.curr_pos))
        #     print("curr_position_idx: " + str(object_info.curr_position_idx))
        #     print("collision_position_idx: " + str(object_info.collision_position_idx))
        #     print("goal_pos: " + str(object_info.goal_pos))
        #     print("goal_position_idx: " + str(object_info.goal_position_idx))
        #     print("\n")

        return self.initial_arrangement, self.final_arrangement, True

    
    def obtainCylinderObjectsInfo(self):
        cylinder_objects = []
        for object_info in self.object_geometries.values():
            cylinder_object = CylinderObj()
            ### fill in the data
            cylinder_object.obj_idx = object_info.object_index
            cylinder_object.curr_position.x = object_info.curr_pos[0]
            cylinder_object.curr_position.y = object_info.curr_pos[1]
            cylinder_object.curr_position.z = object_info.curr_pos[2]
            cylinder_object.radius = object_info.cylinder_radius
            cylinder_object.height = object_info.cylinder_height
            cylinder_objects.append(cylinder_object)
        return cylinder_objects


    def loadInstance_cylinders(self, num_objects, instance_number):

        if num_objects == 6:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if num_objects == 8:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]
        if num_objects == 10:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
        if num_objects == 12:
            self.color_pools = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19], 
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19],
                                [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0.96, 0.51, 0.19]]

        instanceFile = os.path.join(self.rosPackagePath, "examples", str(num_objects)) + "/" + str(instance_number) + "/instance_info.txt"
        self.object_geometries = OrderedDict()
        print("--------load an instance---------")
        self.cylinder_c = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
            radius=self.cylinder_radius, height=self.cylinder_height, physicsClientId=self.server)
        self.num_objects = 0
        counter = 0
        f_instance = open(instanceFile, "r")
        for line in f_instance:
            line = line.split()
            if counter % 2 == 0:
                ### that's object index
                obj_idx = int(line[0])
            if counter % 2 == 1:
                ### that's current/start position for that object
                start_pos = [float(i) for i in line]
                ### create the visualization mesh of the object
                object_rgbacolor = self.color_pools[obj_idx] + [1.0]
                cylinder_start_v = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                            radius=self.cylinder_radius, length=self.cylinder_height, 
                            rgbaColor=object_rgbacolor, physicsClientId=self.server)
                cylinder_objectM = p.createMultiBody(
                    baseCollisionShapeIndex=self.cylinder_c, baseVisualShapeIndex=cylinder_start_v,
                    basePosition=start_pos, physicsClientId=self.server)
                self.object_geometries[obj_idx] = CylinderObject(
                    obj_idx, start_pos, cylinder_objectM, self.cylinder_radius, self.cylinder_height)
                self.num_objects += 1
            ### before the next loop
            counter += 1
        f_instance.close()

        ############## print to confirm ##############
        for obj_idx in range(self.num_objects):
            print(obj_idx)
            print(self.object_geometries[obj_idx].curr_pos)
        ##############################################
        return True


    def deployAllPositionCandidates(self, generateMesh=True, discretization_x=None, discretization_y=None):
        ################ This function calculates all possible postions (x,y) ################
        ################### given the constrained area + discretization ######################
        ############### and generates the candidate meshes if necessary ######################
        self.candidate_geometries = OrderedDict()
        cylinder_c = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
            radius=self.cylinder_radius, height=self.cylinder_height, physicsClientId=self.server)
        rgbacolor = [0.5, 0.5, 0.5, 0.15]
        cylinder_v = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
            radius=self.cylinder_radius, length=self.cylinder_height, rgbaColor=rgbacolor, physicsClientId=self.server)        

        if discretization_x == None: discretization_x = self.discretization_x
        if discretization_y == None: discretization_y = self.discretization_y
        ### formula for compute n_y and n_x (number of objects in y and x direction)
        ### 0.02 + r + d_x*(n_x - 1) + (r + sc_x) = |x|
        ### (r + sc_y) + d_y*(n_y - 1) + (r + sc_y) = |y|
        self.n_candidates_x = \
            int(np.floor((self.constrained_area_dim[0] - 0.02 - self.side_clearance_x + discretization_x - 2*self.cylinder_radius) / discretization_x))
        self.n_candidates_y = \
            int(np.floor((self.constrained_area_dim[1] - 2*self.side_clearance_y + discretization_y - 2*self.cylinder_radius) / discretization_y))

        candidate_idx = 0
        for x_i in range(self.n_candidates_x):
            for y_j in range(self.n_candidates_y):
                candidate_pos = [
                    round(self.constrained_area_x_limit[0] + 0.02 + self.cylinder_radius + discretization_x * x_i, 3),
                    round(self.constrained_area_y_limit[0] + self.side_clearance_y + self.cylinder_radius + discretization_y * y_j, 3),
                    round(self.tablePosition[2] + self.table_dim[2] / 2 + self.cylinder_height / 2, 3)                    
                ]
                if generateMesh:
                    cylinder_objectM = p.createMultiBody(
                        baseCollisionShapeIndex=cylinder_c, baseVisualShapeIndex=cylinder_v, \
                        basePosition=candidate_pos, physicsClientId=self.server)
                else:
                    cylinder_objectM = None
                self.candidate_geometries[candidate_idx] = CylinderCandidate(
                                    candidate_idx, candidate_pos, cylinder_objectM, self.cylinder_radius, self.cylinder_height)
                candidate_idx += 1
        self.num_candidates = len(self.candidate_geometries)
        self.positionCandidate_x_limit = [self.candidate_geometries[0].pos[0], \
                                          self.candidate_geometries[self.num_candidates-1].pos[0]]
        self.positionCandidate_y_limit = [self.candidate_geometries[0].pos[1], \
                                          self.candidate_geometries[self.num_candidates-1].pos[1]]
        ### printing test
        # for candidate_idx, cylinder_candidate in self.candidate_geometries.items():
        #     print(str(candidate_idx) + ": " + str(cylinder_candidate.pos))

    def assignGoalPositions(self, object_interval_x=None, object_interval_y=None):
        ### assign goal positions based on #objects
        ### as well as object_interval_x and object_interval_y
        if object_interval_x == None: object_interval_x = self.object_interval_x
        if object_interval_y == None: object_interval_y = self.object_interval_y
        self.all_goal_positions = OrderedDict()

        self.incremental_x = int(object_interval_x / self.discretization_x)
        self.incremental_y = int(object_interval_y / self.discretization_y)

        if self.num_objects == 6:
            temp_row_start_idx = 3
            candidate_idx = 3
            self.all_goal_positions[0] = candidate_idx
            self.all_goal_positions[1] = self.all_goal_positions[0] + self.incremental_y
            self.all_goal_positions[2] = self.all_goal_positions[1] + self.incremental_y
            candidate_idx = temp_row_start_idx + self.n_candidates_y * self.incremental_x
            self.all_goal_positions[3] = candidate_idx
            self.all_goal_positions[4] = self.all_goal_positions[3] + self.incremental_y
            self.all_goal_positions[5] = self.all_goal_positions[4] + self.incremental_y

        if self.num_objects == 8:
            temp_row_start_idx = 2
            candidate_idx = 2
            self.all_goal_positions[0] = candidate_idx
            self.all_goal_positions[1] = self.all_goal_positions[0] + self.incremental_y
            self.all_goal_positions[2] = self.all_goal_positions[1] + self.incremental_y
            self.all_goal_positions[3] = self.all_goal_positions[2] + self.incremental_y
            candidate_idx = temp_row_start_idx + self.n_candidates_y * self.incremental_x
            self.all_goal_positions[4] = candidate_idx
            self.all_goal_positions[5] = self.all_goal_positions[4] + self.incremental_y
            self.all_goal_positions[6] = self.all_goal_positions[5] + self.incremental_y
            self.all_goal_positions[7] = self.all_goal_positions[6] + self.incremental_y

        if self.num_objects == 10:
            temp_row_start_idx = 1
            candidate_idx = 1
            self.all_goal_positions[0] = candidate_idx
            self.all_goal_positions[1] = self.all_goal_positions[0] + self.incremental_y
            self.all_goal_positions[2] = self.all_goal_positions[1] + self.incremental_y
            self.all_goal_positions[3] = self.all_goal_positions[2] + self.incremental_y
            self.all_goal_positions[4] = self.all_goal_positions[3] + self.incremental_y
            candidate_idx = temp_row_start_idx + self.n_candidates_y * self.incremental_x
            self.all_goal_positions[5] = candidate_idx
            self.all_goal_positions[6] = self.all_goal_positions[5] + self.incremental_y
            self.all_goal_positions[7] = self.all_goal_positions[6] + self.incremental_y
            self.all_goal_positions[8] = self.all_goal_positions[7] + self.incremental_y
            self.all_goal_positions[9] = self.all_goal_positions[8] + self.incremental_y

        if self.num_objects == 12:
            temp_row_start_idx = 1
            candidate_idx = 0
            self.all_goal_positions[0] = candidate_idx
            self.all_goal_positions[1] = self.all_goal_positions[0] + self.incremental_y
            self.all_goal_positions[2] = self.all_goal_positions[1] + self.incremental_y
            self.all_goal_positions[3] = self.all_goal_positions[2] + self.incremental_y
            self.all_goal_positions[4] = self.all_goal_positions[3] + self.incremental_y
            self.all_goal_positions[5] = self.all_goal_positions[4] + self.incremental_y
            candidate_idx = temp_row_start_idx + self.n_candidates_y * self.incremental_x
            self.all_goal_positions[6] = candidate_idx - 1
            self.all_goal_positions[7] = self.all_goal_positions[6] + self.incremental_y
            self.all_goal_positions[8] = self.all_goal_positions[7] + self.incremental_y
            self.all_goal_positions[9] = self.all_goal_positions[8] + self.incremental_y
            self.all_goal_positions[10] = self.all_goal_positions[9] + self.incremental_y
            self.all_goal_positions[11] = self.all_goal_positions[10] + self.incremental_y


        # print("all_goal_positions: ")
        # for obj_idx, position_idx in self.all_goal_positions.items():
        #     print(str(obj_idx) + ": " + str(position_idx))

    def assignToNearestCandiate(self, position):
        ### given a position (x,y,z), calculate the nearest candidate to that position
        ### and return the idx of the position candidate
        potential_neighbor_indexes = []
        x_value = math.floor((position[0]-self.positionCandidate_x_limit[0])/self.discretization_x)
        if x_value < 0:
            ### very close to constraint area x lower limit
            x_indexes = [0]
        elif x_value >= self.n_candidates_x - 1:
            ### very close to constraint area x upper limit
            x_indexes = [self.n_candidates_x - 1] 
        else:
            x_indexes = [x_value, x_value+1]
        y_value = math.floor((position[1]-self.positionCandidate_y_limit[0])/self.discretization_y)
        if y_value < 0:
            ### very close to constraint area y lower limit
            y_indexes = [0]
        elif y_value >= self.n_candidates_y - 1:
            ### very close to constraint area y upper limit
            y_indexes = [self.n_candidates_y - 1] 
        else:
            y_indexes = [y_value, y_value+1]

        for y_idx in y_indexes:
            for x_idx in x_indexes:
                potential_neighbor_indexes.append(x_idx*self.n_candidates_y+y_idx)
        ### find the nearest candidate
        nearest_candidate_dist = np.inf
        for neighbor_idx in potential_neighbor_indexes:
            temp_dist = utils.computePoseDist_pos(position, self.candidate_geometries[neighbor_idx].pos)
            if temp_dist < nearest_candidate_dist:
                nearest_candidate_dist = temp_dist
                nearest_candidate_idx = neighbor_idx

        return nearest_candidate_idx


    def updateObjectMesh(self, obj_idx, position_idx, orientation=[0, 0, 0, 1]):
        ### input - target_pose: [x, y, z]
        if position_idx >= self.num_candidates:
            ### this is an object initial position
            position = self.object_initial_infos[obj_idx].pos
            collision_position_idx = self.object_initial_infos[obj_idx].collision_position_idx
        else:
            ### this is a candidate position
            position = self.candidate_geometries[position_idx].pos
            collision_position_idx = position_idx

        p.resetBasePositionAndOrientation(
            self.object_geometries[obj_idx].geo, position, orientation, physicsClientId=self.server)
        self.object_geometries[obj_idx].setCurrPosition(position_idx, collision_position_idx, position)


    def reset_planning_instance(self):
        ### this function resets the planning instance
        ### (1) resets all object meshes back to initial positions
        for obj_idx, object_initial_info in self.object_initial_infos.items():
            temp_object_initial_pos = object_initial_info.pos
            temp_position_idx = object_initial_info.position_idx
            temp_collision_position_idx = object_initial_info.collision_position_idx
            ### first update the mesh
            p.resetBasePositionAndOrientation(
                self.object_geometries[obj_idx].geo, 
                temp_object_initial_pos, [0, 0, 0, 1.0], physicsClientId=self.server)
            ### then update the statistics in object_geometries
            self.object_geometries[obj_idx].setCurrPosition(
                temp_position_idx, temp_collision_position_idx, temp_object_initial_pos)

    def clear_planning_instance(self):
        ### this function clears the planning instance
        ### (1) deletes all object meshes in the workspace
        for obj_idx, object_info in self.object_geometries.items():
            p.removeBody(object_info.geo)
        ### (2) delete visualization meshes in the workspace
        for obj_idx, object_mesh in self.goal_visualization_mesh.items():
            p.removeBody(object_mesh)
        ### (3) delete initial_object_position_meshes in the workspace
        for position_idx, position_mesh in self.initial_object_position_meshes.items():
            p.removeBody(position_mesh)

    def clear_execution_instance(self):
        ### this function clears the execution instance
        ### (1) deletes all object meshes in the workspace
        for obj_idx, object_info in self.object_geometries.items():
            p.removeBody(object_info.geo)

    def set_scene_for_objects(self, arrangement):
        ### put the object at the position specified by the arrangement
        for obj_idx in range(len(arrangement)):
            position_idx = arrangement[obj_idx]
            if position_idx >= self.num_candidates:
                ### put the object back to its initial position
                p.resetBasePositionAndOrientation(self.object_geometries[obj_idx].geo, 
                    self.object_initial_infos[obj_idx].pos, [0, 0, 0, 1.0], physicsClientId=self.server)
                self.object_geometries[obj_idx].setCurrPosition(position_idx, 
                    self.object_initial_infos[obj_idx].collision_position_idx, self.object_initial_infos[obj_idx].pos)
            else:
                ### put the object back to a candidate position
                p.resetBasePositionAndOrientation(self.object_geometries[obj_idx].geo, 
                    self.candidate_geometries[position_idx].pos, [0, 0, 0, 1.0], physicsClientId=self.server)
                self.object_geometries[obj_idx].setCurrPosition(position_idx, position_idx, self.candidate_geometries[position_idx].pos)
        # print("=================================")
        # for obj_idx, obj_info in self.object_geometries.items():
        #     print(obj_idx)
        #     print(obj_info.curr_pos)
        #     print(obj_info.curr_position_idx)
        #     print(obj_info.collision_position_idx)
        #     print("\n")

    def selectNoCollisionBuffer(self, object_idx, target_position_idx):
        ### this function selects a buffer to put a specified object without collision
        max_trials = 3
        current_trials = 1
        buffer_select_success = False
        other_object_curr_geometries =[obj_info.geo for obj_info in self.object_geometries.values() if obj_info.object_index != object_idx]
        while (current_trials <= max_trials) and (buffer_select_success == False):
            buffer_idx = random.choice(range(self.num_candidates))
            while buffer_idx == self.object_geometries[object_idx].curr_position_idx or buffer_idx == target_position_idx:
                ### the buffer is selected as the chosen object's current position or its target position
                buffer_idx = random.choice(range(self.num_candidates))
            ### now make sure the selected buffer has safe distance with other existing objects
            isCollision = self.collisionAgent.collisionCheck_instance_cylinder_objects(
                self.candidate_geometries[buffer_idx].geo, other_object_curr_geometries, self.cylinder_radius)
            if not isCollision:
                buffer_select_success = True
                break
            else:
                current_trials += 1
        ### reach here either success or not
        return buffer_select_success, buffer_idx

    def getObjectConstraintRanking(self, objects_to_move, final_arrangement):
        ### this functions ranks the objects in the objects_to_move
        ### giving the reasoning about the their constraints
        ### i.e., how constraining and constrained they are
        object_constraints_degrees = {}
        for obj_idx in objects_to_move:
            ### (0) constraining (inner degree) (1) constrained (outer degree)
            object_constraints_degrees[obj_idx] = [0, 0]
        ### Now check the constraint for the current geometry of the object
        for obj_idx in objects_to_move:
            geo = self.object_geometries[obj_idx].geo
            other_object_geometries = {}
            for other_obj_idx in objects_to_move:
                if other_obj_idx == obj_idx: continue
                other_object_geometries[other_obj_idx] = self.candidate_geometries[final_arrangement[other_obj_idx]].geo
                # other_object_geometries[other_obj_idx] = self.candidate_geometries[self.object_geometries[other_obj_idx].goal_position_idx].geo
            object_goals_to_collide = self.collisionAgent.collisionCheck_objectAndObjects(geo, other_object_geometries)
            for object_to_collide in object_goals_to_collide:
                ### constraint: object_to_collide --> obj_idx
                object_constraints_degrees[obj_idx][0] += 1
                object_constraints_degrees[object_to_collide][1] += 1

        object_degrees = {}
        for obj_idx, degrees in object_constraints_degrees.items():
            object_degrees[obj_idx] = degrees[0] + degrees[1]

        object_ranking = sorted(object_degrees, key=lambda k : object_degrees[k], reverse=True)
        return object_ranking
        

        







### general class of object in the workspace
class AnyObject(object):
    def __init__(self, index, pos, geo):
        self.object_index = index
        self.curr_pos = pos
        self.geo = geo
    
class CylinderObject(AnyObject):
    def __init__(self, index, pos, geo, cylinder_radius, cylinder_height):
        ### invoking the __init__ of the parent class
        AnyObject.__init__(self, index, pos, geo)
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height

    def setCurrPosition(self, curr_position_idx, collision_position_idx, curr_pos=None):
        self.curr_position_idx = curr_position_idx
        self.collision_position_idx = collision_position_idx
        if curr_pos != None:
            self.curr_pos = curr_pos

    def setGoalPosition(self, goal_pos, goal_position_idx):
        self.goal_pos = goal_pos
        self.goal_position_idx = goal_position_idx

class CylinderCandidate(object):
    def __init__(self, position_idx, pos, geo, cylinder_radius, cylinder_height):
        self.position_idx = position_idx
        self.pos = pos
        self.geo = geo
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height

class objectInitialInfo(object):
    def __init__(self, object_idx, pos, position_idx, collision_position_idx):
        self.object_index = object_idx 
        self.pos = pos
        self.position_idx = position_idx
        self.collision_position_idx = collision_position_idx