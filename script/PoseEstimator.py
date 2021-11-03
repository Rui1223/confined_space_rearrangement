#!/usr/bin/env python
from __future__ import division

import time
import sys
import os
import numpy as np

import rospy
import rospkg

from confined_space_rearrangement.msg import CylinderObj
from confined_space_rearrangement.srv import CylinderPositionEstimate, CylinderPositionEstimateResponse
from confined_space_rearrangement.srv import CylinderPoseEstimate, CylinderPoseEstimateRequest

############################### description ################################
### This class temporarily defines a PostEstimator node which obtain the
### current position of the objects in the scene
#############################################################################

class PoseEstimator(object):

    def __init__(self, args):
        rospack = rospkg.RosPack()
        self.rosPackagePath = rospack.get_path("confined_space_rearrangement")

    def rosInit(self):
        ### This function specifies the role of a node instance for this class
        ### and initializes a ros node  
        self.cylinder_position_estimate_server = rospy.Service(
            "cylinder_position_estimate", CylinderPositionEstimate,
            self.cylinder_position_estimate_callback)
        rospy.init_node("fake_pose_estimator", anonymous=True)


    def cylinder_position_estimate_callback(self, req):
        ### call the service provided by the execution scene
        cylinder_objects = self.serviceCall_CylinderPoseEstimate()
        return CylinderPositionEstimateResponse(cylinder_objects)


    def serviceCall_CylinderPoseEstimate(self):
        rospy.wait_for_service("cylinder_pose_estimate")
        request = CylinderPoseEstimateRequest()
        try:
            cylinderPoseEstimate_proxy = rospy.ServiceProxy(
                "cylinder_pose_estimate", CylinderPoseEstimate)
            cylinder_pose_estimate_response = cylinderPoseEstimate_proxy(request)
            return cylinder_pose_estimate_response.cylinder_objects
        except rospy.ServiceException as e:
            print("cylinde_pose_estimate service call failed: %s" % e)        


def main(args):
    pose_estimator = PoseEstimator(args)
    pose_estimator.rosInit()
    rate = rospy.Rate(10) ### 10hz

    while not rospy.is_shutdown():
        rate.sleep()


if __name__ == '__main__':
    main(sys.argv)