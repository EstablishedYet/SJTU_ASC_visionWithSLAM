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
from std_msgs.msg import Int32
from mavros_msgs.msg import WaypointReached
from sensor_msgs.msg import NavSatStatus
from geometry_msgs.msg import Point
from picname_and_time_msgs.msg import picnameAndTime
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
# import serial

# def getHSV(h,s,v):
#     return np.array([int(h/2),int(s/100*255),int(v/100*255)])

# def getMask(img):
#     bluerange=[
#         (getHSV(160,50,70),getHSV(180,100,100)),
#         # (getHSV(160,40,80),getHSV(180,50,100)),
#         # (getHSV(160,25,80),getHSV(180,45,100)),

#         # (getHSV(180,80,65),getHSV(200,100,100)),
#         (getHSV(180,60,65),getHSV(200,100,100)),
#         (getHSV(180,50,80),getHSV(200,60,100)),

#         # (getHSV(200,80,55),getHSV(240,100,100)),
#         (getHSV(200,50,50),getHSV(240,100,100)),
#         # (getHSV(200,40,60),getHSV(240,50,100)),
#     ]

#     redrange=[
#         (getHSV(320,80,45),getHSV(340,100,100)),
#         (getHSV(320,40,65),getHSV(359,80,100)),

#         (getHSV(340,80,55),getHSV(359,100,100)),
#         # (getHSV(340,40,65),getHSV(359,80,100)),

#         (getHSV(0,40,55),getHSV(25,100,100)),
#         # (getHSV(0,40,60),getHSV(25,80))
#     ]
#     h,w=img.shape[:2]
#     redMask=np.zeros((h,w),np.uint8)
#     blueMask=np.zeros((h,w),np.uint8)
#     for i in redrange:
#         mask=cv2.inRange(img,i[0],i[1])
#         redMask=cv2.bitwise_or(redMask,mask)
#     for i in bluerange:
#         mask=cv2.inRange(img,i[0],i[1])
#         blueMask=cv2.bitwise_or(blueMask,mask)
#     return cv2.bitwise_or(redMask,blueMask)

# redrange=[(np.array([0, 80, 150]),np.array([20, 255, 255])),
#           (np.array([150, 80, 150]),np.array([180, 255, 255])),
#          ]
# bluerange=[(np.array([95, 150, 140]),np.array([130, 255, 255])),
#            (np.array([80,100,150]),np.array([95,255,255])),
#            (np.array([95, 100, 175]),np.array([130, 150, 255])),
#           ]


def main():
    # def cameraAdjust():
    #     SERIAL_PORT = "/dev/ttyUSB0"   # 如果是 COM10 以上要写成 r"\\.\COM10"
    #     BAUD_RATE = 115200

    #     # 打开串口
    #     ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

    #     # 要发送的数据
    #     send_buf = bytearray([0x55, 0x66, 0x01, 0x04, 0x00, 0x00, 0x00, 
    #                         0x0E, 0x00, 0x00, 0x7C, 0xFC, 0x4F, 0xA4])

    #     print("Sending:", send_buf.hex())
    #     ser.write(send_buf)
    #     ser.flush()

    #     # 等待响应
    #     time.sleep(0.5)
    #     response = ser.read(ser.in_waiting or 1)

    #     if response:
    #         print("Received:", response.hex())
    #     else:
    #         print("No response received.")

    #     ser.close()

    # cameraAdjust()

    parser=argparse.ArgumentParser()
    parser.add_argument('--move',type=int)
    # parser.add_argument('--retinex',type=int,default=1)
    # parser.add_argument('--clear',type=int,default=0)
    # parser.add_argument('--c1start',type=int,default=7)
    # # parser.add_argument('--c1end',type=int,default=8)
    # parser.add_argument('--c2start',type=int,default=13)
    # parser.add_argument('--c2end',type=int,default=14)
    # parser.add_argument('--numOfFrame',type=int,default=18)
    # parser.add_argument('--numOfCircle',type=int,default=2)
    # parser.add_argument('--numOfProcs',type=int,default=6)
    # parser.add_argument('--camera',type=str)
    # parser.add_argument('--checkpoint',type=int,default=3)
    arg=parser.parse_args()
    needtomove=arg.move
    # mode=arg.mode
    # retinex=arg.retinex
    # clear=arg.clear
    # c1start=arg.c1start
    # c1end=arg.c1end
    # c2start=arg.c2start
    # c2end=arg.c2end
    # numOfFrame=arg.numOfFrame
    # numOfProcs=arg.numOfProcs
    # numOfCircle=arg.numOfCircle
    # # cameraType=arg.camera
    # # expo=arg.expo
    # checkpoint=arg.checkpoint
    # if numOfCircle==1:
        # c2end=c2start=-1

    modelclassify_number=r"/home/amov/sjtu_asc_v2_ws-main/src/mission_offboard/script/models/902detect.pt"
    modelclassify_pattern=r"/home/amov/sjtu_asc_v2_ws-main/src/mission_offboard/script/models/902classify.pt"
    # modelclassify_number_old=r"/home/amov/sjtu_asc_v2_ws-main/src/mission_offboard/script/models/yolo_cls.pt"
    
    wp = 0

    conf_thresh=0.7

    # circle_number=2-numOfCircle

    local_x = 0
    local_y = 0
    local_z = 0
    local_vel_x = 0
    local_vel_y = 0
    local_vel_z = 0
    local_yaw = 0

    GPSstatus=0
    odom_x=0
    odom_y=0
    odom_z=0
    odom_vel_x=0
    odom_vel_y=0
    odom_vel_z=0
    odom_yaw=0

    referenceMode=-1
    permission=-1

    maxdet = 3
    max_num = 3

    fw=1920
    fh=1080
    # expo=1
    mtx = np.array([[2.8724e+03,0.00000000e+00,1.2342e+03],
    [0.00000000e+00,2.8657e+03,6.891308e+02],
    [0.00000000e+00,0.00000000e+00,1.00000000e+00]])

    dist= np.array([-0.5018,0.2920,-0.0034,0.0010,-0.2113])
    # if cameraType=='siyi':
    #     mtx=mtx_siyi
    #     dist=dist_siyi
    # elif cameraType=='self1':
    #     mtx=mtx_self1
    #     dist=dist_self1
    nmtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (fw,fh), alpha=1)
    exposures=[1,4,8,15,30,50,90]

    # if mode=="number":
    #     MODELCLASSIFY = modelclassify_number
    # elif mode=="pattern":
    #     MODELCLASSIFY = modelclassify_pattern
    # # elif mode=="number_old":
    # #     MODELCLASSIFY = modelclassify_number_old
    # else:
    #     raise ValueError("Wrong mode! Please input 'number' or 'pattern'!")

    MODELOBB = r"/home/amov/sjtu_asc_v2_ws-main/src/mission_offboard/script/models/18_obb.pt"
    # print("loading obb model")
    # modelObb = YOLO(MODELOBB)  # 通常是pt模型的文件
    # print("loading classify model")
    # modelClassify = YOLO(MODELCLASSIFY)

    pattern_dic={0:0,1:1,2:10,3:11}
    dic_append={}
    for i in range(4,12):
        dic_append[i]=i-2
    pattern_dic.update(dic_append)

    dataList_s = []
    trusted_dataList=[]
    num_list = []
    trusted_num_list=[]

    # ideal_nums=[]

    circle_failed=[0,0]

    def find_max_numeric_folder(path):
        max_val = -1
        for name in os.listdir(path):
            folder_path = os.path.join(path, name)
            if os.path.isdir(folder_path) and name.isdigit():
                num = int(name)
                if num > max_val:
                    max_val = num
        return max_val

    # source=r"/home/amov/sjtu_asc_v2_ws-main/log_shi" 
    # os.makedirs(source,exist_ok=True)
    # path=os.path.join(source,str(find_max_numeric_folder(source)+1))
    # os.makedirs(path,exist_ok=True)
    # outpath=os.path.join(path,'frames')
    # os.makedirs(outpath,exist_ok=True)

    # console=open(os.path.join(path,'console.txt'),'a')
    # sys.stdout=console
    # sys.stderr=console

    # def obb_predict(inputpath):
    #     results_obb = modelObb.predict(
    #         source=inputpath,
    #         imgsz=1280,  # 此处可以调节
    #         half=True,
    #         iou=0.4,
    #         conf=0.5,
    #         device='0',  # '0'使用GPU运行
    #         max_det=maxdet,
    #         save=False,
    #         project=path,
    #         # workers=4,
    #         batch=10
    #         # classes=red
    #         # augment = True
    #     )
    #     return results_obb

    # def most_common_numbers(all_numbers):
    #     # 使用Counter计算每个字符串的出现次数
    #     count = Counter(all_numbers)
    #     most_common = count.most_common(max_num)
    #     return most_common

    # def get_ideal(list):
    #     newlist=sorted(list)
    #     if mode=="number":
    #         return newlist[1]
    #     else:
    #         return newlist[2]
        
    # def guess_ideal(list):
    #     newlist=sorted(list)
    #     return newlist[-1]
        
    # def coordinate_change(rank,height=25, pos_=[0, 0, 25], yaw=0.00000000, cropTensorList=[[0, 0], [0, 0], [0, 0], [0, 0]],
    #                     speed=[0, 0, 0], ):
        
    #     well_width = (cropTensorList[0][0] + cropTensorList[2][0]) / 2
    #     well_height = (cropTensorList[0][1] + cropTensorList[2][1]) / 2

    #     # 通过ros获取飞行的高度以及X与Y的值  此处默认设置为20
    #     X0 = pos_[0]
    #     Y0 = pos_[1]
    #     Z0 = pos_[2]

    #     x = ((well_width - nmtx[0][2]) / nmtx[0][0]) * Z0
    #     y = ((nmtx[1][2] - well_height) / nmtx[1][1]) * Z0

    #     x_ = x * np.sin(yaw) + y * np.cos(yaw)
    #     y_ = y * np.sin(yaw) - x * np.cos(yaw)
    #     # current_speed = math.pow(speed[0] * speed[0] + speed[1] * speed[1], 0.5)
    #     # delt = current_speed * 1.15#12.5# current_speed * 1.08288

    #     # savepath=os.path.join(path,f'output{rank}.txt')
    #     # with open(savepath, 'a') as file:
    #     #     file.write("yaw is: " + str(yaw) + " ")
    #     #     file.write("pose is: " + str(pos_[0]) + " " + str(pos_[1]) + " " + str(pos_[2]) + " ")
    #     #     file.write("target is: " + str(well_width) + " " + str(well_height) + " ")
    #     #     file.write("speed is:" + str(speed[0]) + " " + str(speed[1]) + " " + str(speed[2]) + "\n")
    #     result_ = [x_ + X0, y_ + Y0 , 0]

    #     return np.array(result_)

    # def common_checked(common_trusted_num_list,allnum):
    #     if len(common_trusted_num_list)<3:
    #         return False
    #     else:
    #         threshold=1/8   
    #         def ratio_check():
    #             for i in range(3):
    #                 if common_trusted_num_list[i][1]/allnum<threshold:
    #                     return False
    #             return True
    #         return ratio_check()
        
    def loc_pose_callback(msg):
        nonlocal local_x, local_y, local_z, local_yaw, local_vel_x, local_vel_y, local_vel_z
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

        # print("callback is used")
        # print("yaw = %f", euler1[2])
        # print("yaw = %f", local_yaw)

        # rospy.loginfo("yaw in /global_positon/local topic: %f", np.rad2deg(euler1[2]))

    def wp_reach_cb(msg):
        nonlocal wp
        wp = msg.wp_seq

    def status_cb(msg):
        nonlocal GPSstatus
        GPSstatus=msg.status.status

    def odom_cb(msg):
        nonlocal odom_x, odom_y, odom_z, odom_yaw, odom_vel_x, odom_vel_y, odom_vel_z
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

    def referenceMode_sub_func(msg):
        nonlocal referenceMode
        referenceMode=msg.data

    def permission_sub_func(msg):
        nonlocal permission
        permission=msg.data

    rospy.init_node("vision_node")
    rate = rospy.Rate(10)
    result_pub = rospy.Publisher("final_result", String, queue_size = 1,latch=True)
    target_pub = rospy.Publisher("final_pos",PoseStamped, queue_size = 1,latch=True)
    # permission_pub=rospy.Publisher("permission",Float64,queue_size=1)
    traditionalTarget_pub=rospy.Publisher("traditionalTarget",PoseStamped,queue_size=1)

    pic_pub=rospy.Publisher("picnameAndTime",picnameAndTime,queue_size=100,latch=True) # /// reconsider the size of queue!!!
    firstLoopEnd_pub=rospy.Publisher("firstLoopEnd",Int32,queue_size=1)
    firstScout_pub=rospy.Publisher("firstScout",Int32,queue_size=1)
    secondLoopEnd_pub=rospy.Publisher("secondLoopEnd",Int32,queue_size=1)
    slamTarget_pub=rospy.Publisher("slamTarget",Point,queue_size=1)
    accepttest_pub=rospy.Publisher('test',Int32,queue_size=1)
    accepttest_pub.publish(1)
    rospy.Subscriber("referenceMode",Int32,referenceMode_sub_func,queue_size=1)
    # rospy.Subscriber("permission",,permission_sub_func,queue_size=1)

    rospy.Subscriber("/mavros/mission/reached",WaypointReached, wp_reach_cb, queue_size = 1)
    rospy.Subscriber("/mavros/global_position/local", Odometry, loc_pose_callback, queue_size=1)
    rospy.Subscriber("/mavros/gpsstatus/gps_status",NavSatStatus,status_cb,queue_size=1)
    rospy.Subscriber("/mavros/local_position/odom",Odometry,odom_cb,queue_size=1)
    path="/home/established/6"
    framepath=os.path.join(path,"2frames")
    os.makedirs(framepath,exist_ok=True)
    if needtomove:
        needtomovefrom=os.path.join(path,"2clearframes")
        
        framelist=os.listdir(framepath)
        # if len(framelist)==0:
        startid=0
        movesourcelist=os.listdir(needtomovefrom)
        movesourcelist=sorted(movesourcelist,key=lambda x:int(os.path.splitext(x)[0]))
        for file in movesourcelist:
            while True:
                if f"{startid}.jpg" in framelist:
                    startid+=1
                else:
                    shutil.copy(os.path.join(needtomovefrom,file),os.path.join(framepath,f"{startid}.jpg"))
                    framelist.append(f"{startid}.jpg")
                    startid+=1
                    break
    # framelist=sorted(framelist,lambda x:int(os.path.splitext(x)[0]))
    ideal_num=43
    ideal_log=os.path.join(path,f"{ideal_num}.txt")
    outputlogpath=os.path.join(path,"output.txt")
    os.makedirs(os.path.join(path,"2slamlogs"),exist_ok=True)
    curtime=0
    rate.sleep()
    while pic_pub.get_num_connections()==0:
        rospy.sleep(0.01)
    print(pic_pub.get_num_connections(),"sub,end")
    filelist=os.listdir(framepath)
    filelist=sorted(filelist,key=lambda x:int(os.path.splitext(x)[0]))
    # curPicnameAndTime=picnameAndTime()
    # curPicnameAndTime.name='test'
    # curPicnameAndTime.time=-1
    # pic_pub.publish(curPicnameAndTime)
    # time.sleep(1)
    for i in filelist:
        curPicnameAndTime=picnameAndTime()
        curPicnameAndTime.name=os.path.join(framepath,i)
        curPicnameAndTime.time=curtime
        pic_pub.publish(curPicnameAndTime)
        print(curtime)
        curtime+=0.1
        time.sleep(0.1)
        # break
    # time.sleep(5)
    firstLoopEnd_pub.publish(1)
    while True:
        if referenceMode!=-1:
            break
        rate.sleep()
    coordinates={}
    # oneflag=0
    def getCoordinate(idealfile):
        nonlocal coordinates
        with open(idealfile,'r') as f:
            for line in f:
                if not line.strip().startswith('['):
                    coordinates[line.strip().split()[2]]=np.zeros((4,2),dtype=np.float32)
                    coordinates[line.strip().split()[2]][0][0]=float(line.strip().split()[8])
                    coordinates[line.strip().split()[2]][0][1]=float(line.strip().split()[9].split(']')[0])
                    oneflag=1
                    curobject=coordinates[line.strip().split()[2]]
                else:
                    curobject[oneflag][0]=float(line.strip().split()[1])
                    curobject[oneflag][1]=float(line.strip().split()[-1].split(']')[0])
                    oneflag+=1
    getCoordinate(ideal_log)
    def getSLAMTarget(logfile,circlr_num):
        SLAM_conf_thresh=0.7
        SLAM_target=np.zeros(3,dtype=float)
        SLAM_target_pic=np.zeros(2,dtype=float)
        SLAM_num=0
        # logfile=""
        with open(logfile,'r') as logf:
            for line in logf:
                if line.startswith("Classify"):
                    line=line.strip()
                    tokens=line.split()
                    num=int(tokens[2].split(':')[-1])
                    conf=float(tokens[3].split(',')[0].split('(')[-1])
                    file=tokens[5]
                    if num==ideal_num and conf>=SLAM_conf_thresh:
                        for i in range(4):
                            SLAM_target_pic[0]+=coordinates[file][i][0]
                            SLAM_target_pic[1]+=coordinates[file][i][1]
                        SLAM_target_pic/=4
                        with open(os.path.join(path,f'{circlr_num}slamlogs',f"{file.split('/')[-1].split('.')[0].split('_')[0]}.txt"),'r') as f:
                            minDistance=10000
                            for line in f:
                                line=line.strip()
                                datas=line.split()
                                distance=pow(float(datas[3])-SLAM_target_pic[0],2)+pow(float(datas[4])-SLAM_target_pic[1],2)
                                if distance<minDistance:
                                    minTarget=np.array([float(datas[0]),float(datas[1]),float(datas[2])])
                                    minDistance=distance
                        
                        SLAM_target+=minTarget
                        SLAM_num+=1
        SLAM_target/=SLAM_num
        return SLAM_target,SLAM_num
    if referenceMode!=0:
        target,num=getSLAMTarget(outputlogpath,2)
        print(target,num)

main()