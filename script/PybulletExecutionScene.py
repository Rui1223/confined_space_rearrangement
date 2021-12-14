#!/usr/bin/env python
from __future__ import division

import pybullet as p
import pybullet_data

import time
import sys
import os
import numpy as np
# from cv_bridge import CvBridge, CvBridgeError
import cv2

from MotomanRobot import MotomanRobot
from WorkspaceTable import WorkspaceTable
from SimulatedCamera import SimulatedCamera
from Executor import Executor

import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image

from confined_space_rearrangement.srv import GenerateInstanceCylinder, GenerateInstanceCylinderResponse
from confined_space_rearrangement.srv import CylinderPoseEstimate, CylinderPoseEstimateResponse
from confined_space_rearrangement.srv import ClearExecutionInstance, ClearExecutionInstanceResponse
from confined_space_rearrangement.srv import ExecuteTrajectory, ExecuteTrajectoryResponse
from confined_space_rearrangement.srv import AttachObject, AttachObjectResponse

############################### description ###############################
### This class defines a PybulletExecutionScene class which
### generates an execution scene that
### (1) sets up the robot, table and camera
### (2) generate an arrangement instance
### (2) execute the computed plan
###########################################################################


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class PybulletExecutionScene(object):

    def __init__(self, args):
        ### read in relevant ros parameters for execution scene
        basePosition, baseOrientation, urdfFile, \
        leftArmHomeConfiguration, rightArmHomeConfiguration, torsoHomeConfiguration, \
        standingBase_dim, table_dim, table_offset_x, \
        camera_extrinsic, camera_intrinsic, \
        cylinder_radius, cylinder_height, \
        side_clearance_x, side_clearance_y, \
        ceiling_height, thickness_flank, \
        object_mesh_path = self.readROSParam()

        ### set the rospkg path
        rospack = rospkg.RosPack()
        self.rosPackagePath = rospack.get_path("confined_space_rearrangement")

        ### set the server for the pybullet execution scene
        # self.executingClientID = p.connect(p.DIRECT)
        self.executingClientID = p.connect(p.GUI)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.egl_plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        # print("plugin=", self.egl_plugin)

        ### configure the robot
        self.configureMotomanRobot(urdfFile, basePosition, baseOrientation, \
            leftArmHomeConfiguration, rightArmHomeConfiguration, torsoHomeConfiguration, True)
        ### setup the workspace
        self.setupWorkspace(standingBase_dim, table_dim, table_offset_x, object_mesh_path, True)
        self.workspace_e.addConstrainedArea(ceiling_height, thickness_flank)
        self.workspace_e.setDeploymentParam(
                cylinder_radius, cylinder_height, side_clearance_x, side_clearance_y)
        self.setupCamera(camera_extrinsic, camera_intrinsic)

        ### create an executor assistant
        self.executor_e = Executor(self.executingClientID,
            isObjectInLeftHand=False, isObjectInRightHand=False,
            objectInLeftHand=None, objectInRightHand=None)


    def configureMotomanRobot(self, 
            urdfFile, basePosition, baseOrientation,
            leftArmHomeConfiguration, rightArmHomeConfiguration, torsoHomeConfiguration, isPhysicsTurnOn):
        ### This function configures the robot in the real scene
        self.robot_e = MotomanRobot(
            os.path.join(self.rosPackagePath, urdfFile), 
            basePosition, baseOrientation, 
            leftArmHomeConfiguration, rightArmHomeConfiguration, torsoHomeConfiguration,
            isPhysicsTurnOn, self.executingClientID)

    def setupWorkspace(self,
            standingBase_dim, table_dim, table_offset_x,
            object_mesh_path, isPhysicsTurnOn):
        ### This function sets up the workspace
        self.workspace_e = WorkspaceTable(self.rosPackagePath, self.robot_e.basePosition,
            standingBase_dim, table_dim, table_offset_x, 
            os.path.join(self.rosPackagePath, object_mesh_path),
            isPhysicsTurnOn, self.executingClientID)

    def setupCamera(self, camera_extrinsic, camera_intrinsic):
        ### This function sets up the camera
        ### indicate which scene you are working on and whether you want to save images

        self.scene_index = "1"
        self.saveImages = False ### decide whether to save images or not
        # self.scene_index = args[1]
        # self.saveImages = (args[2] in ('y', 'Y')) ### decide whether to save images or not
        self.camera_e = SimulatedCamera(
            self.workspace_e.tablePosition, self.workspace_e.table_dim,
            camera_extrinsic, camera_intrinsic,
            self.scene_index, self.saveImages,
            self.executingClientID
        )


    def rosInit(self):
        ### This function specifies the role of a node instance for this class ###
        ### and initialize a ros node
        self.color_im_pub = rospy.Publisher('rgb_images', Image, queue_size=10)
        self.depth_im_pub = rospy.Publisher('depth_images', Image, queue_size=10)

        self.generate_instance_cylinder_server = rospy.Service(
            "generate_instance_cylinder", GenerateInstanceCylinder, 
            self.generate_instance_cylinder_callback)

        self.cylinder_pose_estimate_server = rospy.Service(
            "cylinder_pose_estimate", CylinderPoseEstimate, 
            self.cylinder_pose_estimate_callback)
        
        self.execute_trajectory_server = rospy.Service(
            "execute_trajectory", ExecuteTrajectory, self.execute_traj_callback)

        self.attach_object_server = rospy.Service(
                "attach_object", AttachObject, self.attach_object_callback)

        self.clear_execution_instance_server = rospy.Service(
            "clear_execution_instance", ClearExecutionInstance,
            self.clear_execution_instance_callback)

        rospy.init_node("pybullet_execution_scene", anonymous=True)

    
    def cylinder_pose_estimate_callback(self, req):
        rospy.logwarn("GET THE INFORMATION OF OBJECTS")
        cylinder_objects = self.workspace_e.obtainCylinderObjectsInfo()
        if cylinder_objects != []:
            print("successfully obtain objects information")
        else:
            print("fail to obtain object information")
        return CylinderPoseEstimateResponse(cylinder_objects)

    def generate_instance_cylinder_callback(self, req):
        ### given the request data: num_objects (int32)
        rospy.logwarn("GENERATE REARRANGEMENT INSTANCE")
        if req.isNewInstance:
            success = self.workspace_e.generateInstance_cylinders(req.num_objects)
        else:
            success = self.workspace_e.loadInstance_cylinders(req.num_objects, req.instance_number)
        
        if success == True:
            print("successfully generate an instance")
        else:
            print("fail to generate an instance")
        return GenerateInstanceCylinderResponse(success)

    def clear_execution_instance_callback(self, req):
        ### clear the instance in the execution scene, which involves
        ### (i) delete all object meshes in the workspace, empty object_geometries,
        ###     as well as deleting goal visualization mesh
        self.workspace_e.clear_execution_instance()
        ### (ii) reset the robot back to the home configuration
        self.robot_e.resetRobotToHomeConfiguration()
        return ClearExecutionInstanceResponse(True)

    def execute_traj_callback(self, req):
        ### given the request data: an ArmTrajectory object
        ### execute the trajectory on a specified arm
        self.executor_e.executeTrajectory(
            req.arm_trajectory.trajectory, self.robot_e, req.arm_trajectory.armType)
        return ExecuteTrajectoryResponse(True)

    def attach_object_callback(self, req):
        ### given the request data: attach (bool) + armType (string)
        if req.attach:
            self.executor_e.attachObject(req.object_idx, self.workspace_e, self.robot_e, req.armType)
            # print("successfully attached the object")
            return AttachObjectResponse(True)
        else:
            self.executor_e.detachObject(self.workspace_e, self.robot_e, req.armType)
            # print("successfully detached the object")
            return AttachObjectResponse(True)


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

        while not rospy.has_param('/simulated_camera/camera_extrinsic'):
            rospy.sleep(0.2)
        camera_extrinsic = rospy.get_param('/simulated_camera/camera_extrinsic')

        while not rospy.has_param('/simulated_camera/camera_intrinsic'):
            rospy.sleep(0.2)
        camera_intrinsic = rospy.get_param('/simulated_camera/camera_intrinsic')

        while not rospy.has_param('/uniform_cylinder_object/radius'):
            rospy.sleep(0.2)
        cylinder_radius = rospy.get_param('/uniform_cylinder_object/radius')

        while not rospy.has_param('/uniform_cylinder_object/height'):
            rospy.sleep(0.2)
        cylinder_height = rospy.get_param('/uniform_cylinder_object/height')

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
            camera_extrinsic, camera_intrinsic, \
            cylinder_radius, cylinder_height, \
            side_clearance_x, side_clearance_y, \
            ceiling_height, thickness_flank, \
            object_mesh_path


def main(args):
    pybullet_execution_scene = PybulletExecutionScene(args)
    pybullet_execution_scene.rosInit()
    rate = rospy.Rate(10) ### 10hz
    # bridge = CvBridge()

    count = 0

    while not rospy.is_shutdown():
        ### get the time stamp
        time_stamp = rospy.get_time()
        # rospy.loginfo("time stamp for image and joint state publisher %s" % time_stamp)

        ### disable image rendering temporarily
        # rgbImg, depthImg = pybullet_execution_scene.camera_e.takeRGBImage()
        # # rgb_msg = bridge.cv2_to_imgmsg(rgbImg, 'rgb8')
        # # depth_msg = bridge.cv2_to_imgmsg((1000 * depthImg).astype(np.uint16), 'mono16')
        # # pybullet_execution_scene.color_im_pub.publish(rgb_msg)
        # # pybullet_execution_scene.depth_im_pub.publish(depth_msg)
        # if count == 0:
        #     cv2.imwrite(
        #         os.path.expanduser(os.path.join(pybullet_execution_scene.rosPackagePath, "sensor_images/color.png")), 
        #         cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR))
        #     cv2.imwrite(
        #         os.path.expanduser(os.path.join(pybullet_execution_scene.rosPackagePath, "sensor_images/depth.png")), 
        #         (1000 * depthImg).astype(np.uint16))
            

        count += 1


        rate.sleep()

if __name__ == '__main__':
    main(sys.argv)
