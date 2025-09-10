import math
import time
from collections import Counter
import cv2
import cv_methods as cm
import numpy as np
from ultralytics import YOLO
import pos
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from std_msgs.msg import String
from mavros_msgs.msg import WaypointReached
import tf
import datetime
import os
from math import pi
import argparse

init=PoseStamped()
def init_result_sub(msg):
    global init
    init=msg

rospy.init_node("rostest_node")
rate = rospy.Rate(20)
result_pub = rospy.Publisher("final_result", String, queue_size = 1,latch=True)
target_pub = rospy.Publisher("final_pos",PoseStamped, queue_size = 1,latch=True)
permission_pub=rospy.Publisher("permission",Float64,queue_size=1)


rospy.Subscriber("init_result",PoseStamped, init_result_sub, queue_size = 1)

wp=0
local_x=0
local_y=0
local_z=0
local_yaw=0 
local_vel_x=0 
local_vel_y=0
local_vel_z=0
euler1=0

def loc_pose_callback(msg):
    global local_x, local_y, local_z, local_yaw, local_vel_x, local_vel_y, local_vel_z
    local_x = msg.pose.pose.position.x
    local_y = msg.pose.pose.position.y
    local_z = msg.pose.pose.position.z
    # local_x = msg.pose.position.x
    # local_y = msg.pose.position.y
    # local_z = msg.pose.position.z
    euler1 = tf.transformations.euler_from_quaternion(
        [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
         msg.pose.pose.orientation.w],axes='rzyx')   
    # euler1 = tf.transformations.euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    local_vel_x = msg.twist.twist.linear.x
    local_vel_y = msg.twist.twist.linear.y
    local_vel_z = msg.twist.twist.linear.z

    local_yaw = euler1[2]

    # print("callback is used")
    # print("yaw = %f", euler1[2])
    # print("yaw = %f", local_yaw)

    # rospy.loginfo("yaw in /global_positon/local topic: %f", np.rad2deg(euler1[2]))

def wp_reach_cb(msg):
    global wp
    wp = msg.wp_seq

rospy.Subscriber("/mavros/mission/reached",WaypointReached, wp_reach_cb, queue_size = 1)
rospy.Subscriber("/mavros/global_position/local", Odometry, loc_pose_callback, queue_size=1)

# savepath="/path/to/output.txt"

while not rospy.is_shutdown():
    # with open(savepath,'a') as file:
    #     file.write(f'wp:{wp}'+'\n')
    #     file.write(f'local_x:{local_x}'+'\n')
    #     file.write(f'local_y:{local_y}'+'\n')
    #     file.write(f'local_z:{local_z}'+'\n')
    #     file.write(f'eulur:{euler1}'+'\n')
    #     file.write(f"local_yaw:{local_yaw}"+'\n')
    #     file.write(f"local_vel_x:{local_vel_x}"+'\n')
    #     file.write(f"local_vel_y:{local_vel_y}"+'\n')
    #     file.write(f"local_vel_z:{local_vel_z}"+'\n')
    # final_pos__=PoseStamped()
    # final_pos__.pose.position.x=np.float64(10)
    # final_pos__.pose.position.y=np.float64(15)
    # final_pos__.pose.position.z=np.float64(20)
    target_pub.publish(init)
    permission_pub.publish(1)
    result_pub.publish(str([0,1,2]))
    rate.sleep()

