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
from sensor_msgs.msg import NavSatStatus
from geometry_msgs.msg import Point
import tf
import datetime
import os
from math import pi
import argparse
import random
import multiprocessing
import sys
import subprocess
import shutil
import serial

local_x, local_y, local_z, local_yaw, local_vel_x, local_vel_y, local_vel_z=0,0,0,0,0,0,0
odom_x, odom_y, odom_z, odom_yaw, odom_vel_x, odom_vel_y, odom_vel_z=0,0,0,0,0,0,0

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

    local_yaw = euler1[0]

def odom_cb(msg):
    global odom_x, odom_y, odom_z, odom_yaw, odom_vel_x, odom_vel_y, odom_vel_z
    odom_x = msg.pose.pose.position.x
    odom_y = msg.pose.pose.position.y
    odom_z = msg.pose.pose.position.z
    # odom_x = msg.pose.position.x
    # odom_y = msg.pose.position.y
    # odom_z = msg.pose.position.z
    euler1 = tf.transformations.euler_from_quaternion(
        [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w],axes='rzyx')   
    # euler1 = tf.transformations.euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
    odom_vel_x = msg.twist.twist.linear.x
    odom_vel_y = msg.twist.twist.linear.y
    odom_vel_z = msg.twist.twist.linear.z

    odom_yaw = euler1[0]


rospy.init_node("backup_node")
rate = rospy.Rate(30)
# result_pub = rospy.Publisher("final_result", String, queue_size = 1,latch=True)
# target_pub = rospy.Publisher("final_pos",PoseStamped, queue_size = 1,latch=True)
# permission_pub=rospy.Publisher("permission",Float64,queue_size=1)
# rospy.Subscriber("/mavros/mission/reached",WaypointReached, wp_reach_cb, queue_size = 1)
rospy.Subscriber("/mavros/global_position/local", Odometry, loc_pose_callback, queue_size=1)
# rospy.Subscriber("/mavros/gpsstatus/gps_status",NavSatStatus,status_cb,queue_size=1)
rospy.Subscriber("/mavros/local_position/odom",Odometry,odom_cb,queue_size=1)

path='/home/amov/odom'
os.makedirs(path,exist_ok=True)
with open(os.path.join(path,f"{time.time()}.txt"),'w') as f:
    while True:
        f.write(str(time.localtime())+'\n')
        f.write("   global: "+f"{local_x} {local_y} {local_z} {local_vel_x} {local_vel_y} {local_vel_z} {local_yaw}\n")
        f.write("   local: "+f"{odom_x} {odom_y} {odom_z} {odom_vel_x} {odom_vel_y} {odom_vel_z} {odom_yaw}\n")
        rate.sleep()

