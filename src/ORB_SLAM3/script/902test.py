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
from package__.msg import picnameAndTime
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
    def cameraAdjust():
        SERIAL_PORT = "/dev/ttyUSB0"   # 如果是 COM10 以上要写成 r"\\.\COM10"
        BAUD_RATE = 115200

        # 打开串口
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

        # 要发送的数据
        send_buf = bytearray([0x55, 0x66, 0x01, 0x04, 0x00, 0x00, 0x00, 
                            0x0E, 0x00, 0x00, 0x7C, 0xFC, 0x4F, 0xA4])

        print("Sending:", send_buf.hex())
        ser.write(send_buf)
        ser.flush()

        # 等待响应
        time.sleep(0.5)
        response = ser.read(ser.in_waiting or 1)

        if response:
            print("Received:", response.hex())
        else:
            print("No response received.")

        ser.close()

    cameraAdjust()

    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',type=str)
    # parser.add_argument('--retinex',type=int,default=1)
    parser.add_argument('--clear',type=int,default=0)
    parser.add_argument('--c1start',type=int,default=7)
    parser.add_argument('--c1end',type=int,default=8)
    parser.add_argument('--c2start',type=int,default=13)
    parser.add_argument('--c2end',type=int,default=14)
    parser.add_argument('--numOfFrame',type=int,default=18)
    parser.add_argument('--numOfCircle',type=int,default=2)
    parser.add_argument('--numOfProcs',type=int,default=8)
    parser.add_argument('--camera',type=str)
    arg=parser.parse_args()
    
    mode=arg.mode
    # retinex=arg.retinex
    clear=arg.clear
    c1start=arg.c1start
    c1end=arg.c1end
    c2start=arg.c2start
    c2end=arg.c2end
    numOfFrame=arg.numOfFrame
    numOfProcs=arg.numOfProcs
    numOfCircle=arg.numOfCircle
    cameraType=arg.camera
    if numOfCircle==1:
        c2end=c2start=-1

    modelclassify_number=r"/home/amov/sjtu_asc_v2_ws-main/src/mission_offboard/script/models/902detect.pt"
    modelclassify_pattern=r"/home/amov/sjtu_asc_v2_ws-main/src/mission_offboard/script/models/902classify.pt"
    # modelclassify_number_old=r"/home/amov/sjtu_asc_v2_ws-main/src/mission_offboard/script/models/yolo_cls.pt"
    
    wp = 0

    conf_thresh=0.7

    circle_number=2-numOfCircle

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

    maxdet = 3
    max_num = 3

    fw=1920
    fh=1080
    expo=1
    mtx_siyi = np.array([[1075.5,0.00000000e+00,997.3],
                [0.00000000e+00,1079.4,566],
                [0.00000000e+00,0.00000000e+00,1.00000000e+00]],dtype=np.float64)
    dist_siyi = np.array([[-0.077,0.0624,0,0,0]])
    mtx_self1=np.array([[2820.3,0.00000000e+00,1267],
                [0.00000000e+00,2837.5,612],
                [0.00000000e+00,0.00000000e+00,1.00000000e+00]],dtype=np.float64)
    dist_self1=np.array([[-0.52,0.7131,-0.0064,-0.00123,-1.333]])
    if cameraType=='siyi':
        mtx=mtx_siyi
        dist=dist_siyi
    elif cameraType=='self1':
        mtx=mtx_self1
        dist=dist_self1
    nmtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (fw,fh), alpha=1)

    if mode=="number":
        MODELCLASSIFY = modelclassify_number
    elif mode=="pattern":
        MODELCLASSIFY = modelclassify_pattern
    # elif mode=="number_old":
    #     MODELCLASSIFY = modelclassify_number_old
    else:
        raise ValueError("Wrong mode! Please input 'number' or 'pattern'!")

    MODELOBB = r"/home/amov/sjtu_asc_v2_ws-main/src/mission_offboard/script/models/18_obb.pt"
    # print("loading obb model")
    modelObb = YOLO(MODELOBB)  # 通常是pt模型的文件
    # print("loading classify model")
    modelClassify = YOLO(MODELCLASSIFY)

    pattern_dic={0:0,1:1,2:10,3:11}
    dic_append={}
    for i in range(4,12):
        dic_append[i]=i-2
    pattern_dic.update(dic_append)

    dataList = []
    trusted_dataList=[]
    num_list = []
    trusted_num_list=[]

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

    source=r"/home/amov/sjtu_asc_v2_ws-main/log_shi" 
    os.makedirs(source,exist_ok=True)
    path=os.path.join(source,str(find_max_numeric_folder(source)+1))
    os.makedirs(path,exist_ok=True)
    outpath=os.path.join(path,'frames')
    os.makedirs(outpath,exist_ok=True)

    console=open(os.path.join(path,'console.txt'),'a')
    sys.stdout=console
    sys.stderr=console

    def obb_predict(inputpath):
        results_obb = modelObb.predict(
            source=inputpath,
            imgsz=1280,  # 此处可以调节
            half=True,
            iou=0.4,
            conf=0.5,
            device='0',  # '0'使用GPU运行
            max_det=maxdet,
            save=False,
            project=path,
            # workers=4,
            batch=10
            # classes=red
            # augment = True
        )
        return results_obb

    def most_common_numbers(all_numbers):
        # 使用Counter计算每个字符串的出现次数
        count = Counter(all_numbers)
        most_common = count.most_common(max_num)
        return most_common

    def get_ideal(list):
        newlist=sorted(list)
        if mode=="number":
            return newlist[1]
        else:
            return newlist[2]
        
    def guess_ideal(list):
        newlist=sorted(list)
        return newlist[-1]
        
    def coordinate_change(rank,height=25, pos_=[0, 0, 25], yaw=0.00000000, cropTensorList=[[0, 0], [0, 0], [0, 0], [0, 0]],
                        speed=[0, 0, 0], ):
        
        well_width = (cropTensorList[0][0] + cropTensorList[2][0]) // 2
        well_height = (cropTensorList[0][1] + cropTensorList[2][1]) // 2

        # 通过ros获取飞行的高度以及X与Y的值  此处默认设置为20
        X0 = pos_[0]
        Y0 = pos_[1]
        Z0 = pos_[2]

        x = ((well_width - nmtx[0][2]) / nmtx[0][0]) * Z0
        y = ((nmtx[1][2] - well_height) / nmtx[1][1]) * Z0

        x_ = x * np.cos(yaw) - y * np.sin(yaw)
        y_ = y * np.cos(yaw) + x * np.sin(yaw)
        # current_speed = math.pow(speed[0] * speed[0] + speed[1] * speed[1], 0.5)
        # delt = current_speed * 1.15#12.5# current_speed * 1.08288

        # savepath=os.path.join(path,f'output{rank}.txt')
        # with open(savepath, 'a') as file:
        #     file.write("yaw is: " + str(yaw) + " ")
        #     file.write("pose is: " + str(pos_[0]) + " " + str(pos_[1]) + " " + str(pos_[2]) + " ")
        #     file.write("target is: " + str(well_width) + " " + str(well_height) + " ")
        #     file.write("speed is:" + str(speed[0]) + " " + str(speed[1]) + " " + str(speed[2]) + "\n")
        result_ = [x_ + X0, y_ + Y0 , 0]

        return np.array(result_)

    def common_checked(common_trusted_num_list,allnum):
        if len(common_trusted_num_list)<3:
            return False
        else:
            threshold=1/8   
            def ratio_check():
                for i in range(3):
                    if common_trusted_num_list[i][1]/allnum<threshold:
                        return False
                return True
            return ratio_check()
        
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

    rospy.init_node("vision_node")
    rate = rospy.Rate(30)
    # result_pub = rospy.Publisher("final_result", String, queue_size = 1,latch=True)
    # target_pub = rospy.Publisher("final_pos",PoseStamped, queue_size = 1,latch=True)
    # permission_pub=rospy.Publisher("permission",Float64,queue_size=1)
    rospy.Subscriber("/mavros/mission/reached",WaypointReached, wp_reach_cb, queue_size = 1)
    rospy.Subscriber("/mavros/global_position/local", Odometry, loc_pose_callback, queue_size=1)
    rospy.Subscriber("/mavros/gpsstatus/gps_status",NavSatStatus,status_cb,queue_size=1)
    rospy.Subscriber("/mavros/local_position/odom",Odometry,odom_cb,queue_size=1)
    frameid=0
    framestart=0
    cameraAdjust()
    while True:
        if wp==c1start or wp==c2start:
            
            id=0
            while id<=30:
                cap = cv2.VideoCapture(id)
                if not cap.isOpened():
                    id+=1
                else:
#                     subprocess.run(["v4l2-ctl", f"--device=/dev/video{id}", "--set-ctrl", "exposure_auto=1"])

# # # 设置曝光值为90谢谢
#                     subprocess.run(["v4l2-ctl", f"--device=/dev/video{id}", "--set-ctrl", f"exposure_absolute={expo}"])
                    break
            cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, fw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fh)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 8)

            # frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            # frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # frameFps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
            # buffer_size = cap.get(cv2.CAP_PROP_BUFFERSIZE)
            # print(frameWidth, frameHeight, frameFps, buffer_size)

            circle_number+=1
            mapofclear={}
            alldataList=[]
            sectionid=0
            os.makedirs(os.path.join(path,f'{circle_number}crop'),exist_ok=True)
            os.makedirs(os.path.join(path,f'{circle_number}crop1'),exist_ok=True)
            os.makedirs(os.path.join(path,f'{circle_number}crop2'),exist_ok=True)

            # os.makedirs(os.path.join(path,f'{circle_number}masks'),exist_ok=True)
            # os.makedirs(os.path.join(path,f'{circle_number}masks_'),exist_ok=True)
            # os.makedirs(os.path.join(path,f'{circle_number}origin'),exist_ok=True)
            # os.makedirs(os.path.join(path,f'{circle_number}rotate'),exist_ok=True)

            clearframesPath=os.path.join(path,f'{circle_number}clearframes')
            os.makedirs(clearframesPath,exist_ok=True)
            # out = cv2.VideoWriter(f"/home/amov/Desktop/well{folder_name}/output{circle_number}.mp4", fourcc, 30, (1920, 1080))
            # file=open(os.path.join(path,"odom.txt"),'a')
            while not rospy.is_shutdown():                                
                if cap.isOpened():
                    # print("camera is opened")
                    success, frame = cap.read()
                    if success == True:
                        if wp == c1end or wp== c2end:
                            # print("code exit by point")
                            break
                        # out.write(frame)
                        cv2.imwrite(os.path.join(outpath,f'{frameid}.jpg'),frame)
                        frameid+=1
                        allimgdata = pos.Imgdata(pos=[local_x, local_y, local_z],  num=-1, 
                                                yaw=local_yaw, cropTensorList=[(0,0),(0,0),(0,0),(0,0)],
                                                speed=[local_vel_x, local_vel_y, local_vel_z],)
                        alldataList.append(allimgdata)
                        with open(os.path.join(path,"odom.txt"),'a') as file:
                            file.write("global: "+str(local_x)+' '+str(local_y)+' '+str(local_z)+' '+str(local_vel_x)+' '+str(local_vel_y)+' '+str(local_vel_z)+' '+str(local_yaw)+'\n')
                            file.write("local: "+str(odom_x)+' '+str(odom_y)+' '+str(odom_z)+' '+str(odom_vel_x)+' '+str(odom_vel_y)+' '+str(odom_vel_z)+' '+str(odom_yaw)+'\n')
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break
                    else:
                        break
                # rate.sleep()
            cap.release()
            # file.close()
            # cv2.destroyAllWindows()
            
            maxvar=0
            clearid=0
            samples=[]
            if frameid-framestart<numOfFrame:
                ratio=1
            else:
                ratio=(frameid-framestart)//numOfFrame
            if clear==0:
                for i in range(framestart,frameid):
                    if (i-framestart)%ratio==0:
                        # frame=cv2.imread(os.path.join(outpath,f'{i}.jpg'))
                        clearid=i-framestart
                        shutil.move(os.path.join(outpath,f'{clearid}.jpg'),os.path.join(clearframesPath,f'{sectionid:04d}.jpg'))
                        # cv2.imwrite(os.path.join(clearframesPath,f'{sectionid:04d}.jpg'),cv2.imread(os.path.join(outpath,f'{clearid}.jpg')))
                        mapofclear[sectionid]=clearid
                        sectionid+=1
                    # if (i+1)%ratio==0 or i==frameid-1:
                    #     cv2.imwrite(os.path.join(clearframesPath,f'{sectionid:04d}.jpg'),cv2.imread(os.path.join(outpath,f'{clearid}.jpg')))
                    #     mapofclear[sectionid]=clearid
                    #     sectionid+=1
            else:
                for i in range(framestart,frameid):
                    if (i-framestart)%ratio==0 :
                        samples=random.sample(range(i+1,i+ratio),(ratio-1)//2)
                        frame=cv2.imread(os.path.join(outpath,f'{i}.jpg'))
                        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                        var=cv2.Laplacian(gray,cv2.CV_64F).var()
                        maxvar=var
                        clearid=i-framestart
                    elif i in samples:
                        frame=cv2.imread(os.path.join(outpath,f'{i}.jpg'))
                        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                        var=cv2.Laplacian(gray,cv2.CV_64F).var()
                        if var>maxvar:
                            maxvar=var
                            clearid=i-framestart
                    if (i-framestart+1)%ratio==0 or i==frameid-1:
                        shutil.move(os.path.join(outpath,f'{clearid}.jpg'),os.path.join(clearframesPath,f'{sectionid:04d}.jpg'))
                        # cv2.imwrite(os.path.join(clearframesPath,f'{sectionid:04d}.jpg'),cv2.imread(os.path.join(outpath,f'{clearid}.jpg')))
                        mapofclear[sectionid]=clearid
                        sectionid+=1
            framestart=frameid
            t0=time.time()
            results=obb_predict(clearframesPath)
            # results=[r.to('cpu') for r in results]
            divided_results=[]
            for i in range(numOfProcs):
                divided_results.append([i,[],path,circle_number,mtx,dist,nmtx])
            for i,result in enumerate(results):
                rank=i%numOfProcs
                cupresult=[result.path,cv2.cvtColor(result.orig_img,cv2.COLOR_BGR2GRAY),result.obb.xyxyxyxy.cpu().numpy()]
                divided_results[rank][1].append(cupresult)

            with multiprocessing.Pool(processes=numOfProcs) as pool:
                cls_sources=pool.map(cls_predict,divided_results)
            file=open(os.path.join(path,'output.txt'), 'a')
            if mode=="number":
                # dict1={0:{},1:{},2:{},3:{},4:{},5:{},6:{},7:{}}
                # dict1={}
                # for i in range(numOfProcs):
                #     dict1[i]={}
                # for cls_source in cls_sources: # cls_source is a tensors
                    # starts[i]=cls_source[1]
                results_classify1 = modelClassify.predict(
                    source=os.path.join(path,f'{circle_number}crop'),
                    imgsz=640,
                    device='0',
                    save=False,
                    batch=12,
                    half=True,
                    max_det=2
                )
                for result in results_classify1:
                    name=result.path
                    boxes=result.boxes
                    x1=boxes.xywh[0][0]
                    prob1=boxes.conf[0]
                    num1=(int)(boxes.cls[0])
                    x2=boxes.xywh[1][0]
                    prob2=boxes.conf[1]
                    num2=(int)(boxes.cls[1])
                    if x1<=x2:
                        num=num1*10+num2
                    else:
                        num=num1+num2*10
                    # dict1[int(name.split('/')[-1].split('.')[0].split('_')[1])][name.split('/')[-1].split('.')[0].split('_')[-1]]=(int(result.probs.top1),result.probs.top1conf)
                    alldata=alldataList[mapofclear[int(name.split('/')[-1].split('.')[0].split('_')[0])]]
                    imgdata = pos.Imgdata(pos=alldata.pos,num=num,
                                        yaw=alldata.yaw,cropTensorList=cls_sources[int(name.split('/')[-1].split('.')[0].split('_')[1])][int(name.split('/')[-1].split('.')[0].split('_')[-1])],
                                        speed=alldata.speed)
                    prob=min(prob1,prob2)
                    num_and_prob=(num,prob)
                    num_list.append(num_and_prob)
                    dataList.append(imgdata)
                    if prob>=0.7:
                        trusted_num_list.append(num)
                        trusted_dataList.append(imgdata)
                    # print("Classify num is:" + str(num))
                    # savepath=os.path.join(path,'output.txt')
                    # with open(savepath, 'a') as file:
                    file.write("Classify num is:" + str(num) +' '+str(prob)+' '+str(name)+"\n")
                # results_classify2=modelClassify.predict(
                #     source=os.path.join(path,f'{circle_number}crop2'),
                #     imgsz=640,
                #     device='0',
                #     save=False,
                #     batch=12,
                #     half=True,
                # )
                # for result in results_classify2:
                #     name=result.path
                #     result1=dict1[int(name.split('/')[-1].split('.')[0].split('_')[1])][name.split('/')[-1].split('.')[0].split('_')[-1]]
                #     num=10*result1[0]+int(result.probs.top1)
                #     alldata=alldataList[mapofclear[int(name.split('/')[-1].split('.')[0].split('_')[0])]]
                #     imgdata = pos.Imgdata(pos=alldata.pos,num=num,
                #                         yaw=alldata.yaw,cropTensorList=cls_sources[int(name.split('/')[-1].split('.')[0].split('_')[1])][int(name.split('/')[-1].split('.')[0].split('_')[-1])],
                #                         speed=alldata.speed)
                #     num_and_prob=(num,result1[1]*result.probs.top1conf)
                #     num_list.append(num_and_prob)
                #     dataList.append(imgdata)
                #     if result1[1]>=0.7 and result.probs.top1conf>=0.7:
                #         trusted_num_list.append(num)
                #         trusted_dataList.append(imgdata)
                #     # print("Classify num is:" + str(num))
                #     # savepath=os.path.join(path,'output.txt')
                #     # with open(savepath, 'a') as file:
                #     file.write("Classify num is:" + str(num) +' '+str(result1[1])+' '+str(result.probs.top1conf) +' '+str(name)+"\n")
            else:
                # for cls_source in cls_sources: # cls_source[0] is a tensors
                    # starts[i]=cls_source[1]
                results_classify=modelClassify.predict(
                    source=os.path.join(path,f'{circle_number}crop'),
                    imgsz=640,
                    device='0',
                    save=False,
                    batch=12,
                    half=True
                )
                for result in results_classify:
                    name=result.path
                    
                    num=pattern_dic[int(result.probs.top1)]
                    alldata=alldataList[mapofclear[int(name.split('/')[-1].split('.')[0].split('_')[0])]]
                    imgdata = pos.Imgdata(pos=alldata.pos,num=num,
                                        yaw=alldata.yaw,cropTensorList=cls_sources[int(name.split('/')[-1].split('.')[0].split('_')[1])][int(name.split('/')[-1].split('.')[0].split('_')[-1])],
                                        speed=alldata.speed)
                    num_and_prob=(num,result.probs.top1conf)
                    num_list.append(num_and_prob)
                    dataList.append(imgdata)
                    if result.probs.top1conf>=0.7:
                        trusted_num_list.append(num)
                        trusted_dataList.append(imgdata)
                    # print("Classify num is:" + str(num))
                    # savepath=os.path.join(path,'output.txt')
                    # with open(savepath, 'a') as file:
                    file.write("Classify num is:" + str(num) +' '+str(result.probs.top1conf)+' '+str(name)+"\n")
                    # alldata=alldataList[mapofclear[i+bias]]e.write("Classify num is:" + str(num) +' '+str(results_classify1[0].probs.top1conf)+' '+str(results_classify2[0].probs.top1conf) +"\n")
                    # alldata=alldataList[mapofclear[i+bias]]

            common_trusted_num_list=most_common_numbers(trusted_num_list)
            zero_poses=[np.zeros(3),np.zeros(3),np.zeros(3)]
            if common_checked(common_trusted_num_list,len(num_list)):
                for i in range(3):
                    common_trusted_num_list[i]=common_trusted_num_list[i][0]
                ideal_num = get_ideal(common_trusted_num_list)
                object_sum = len(trusted_dataList)
                length=3
                middles=np.zeros(length)
                files={common_trusted_num_list[0]:open(os.path.join(path,f'{common_trusted_num_list[0]}.txt'), 'a'),
                       common_trusted_num_list[1]:open(os.path.join(path,f'{common_trusted_num_list[1]}.txt'), 'a'),
                       common_trusted_num_list[2]:open(os.path.join(path,f'{common_trusted_num_list[2]}.txt'), 'a')}
                for k in range(object_sum):  # 识别到数字的有效
                    for rank in range(length):
                        if common_trusted_num_list[rank] == trusted_dataList[k].get_num():
                            # print(num_list[0])
                            cur_pos = coordinate_change(rank,height=25, pos_=[trusted_dataList[k].get_pos()[0], trusted_dataList[k].get_pos()[1], trusted_dataList[k].get_pos()[2]],
                                                            yaw=trusted_dataList[k].get_yaw(), cropTensorList=trusted_dataList[k].get_cropTensorList(),
                                                            speed=trusted_dataList[k].get_speed())
                            # with open(os.path.join(path,f'{common_trusted_num_list[rank]}.txt'), 'a') as file1:
                            files[common_trusted_num_list[rank]].write(str(cur_pos[0]) + " " + str(cur_pos[1]) + "\n")
                            zero_poses[rank] += cur_pos
                            middles[rank] += 1
                            break
                final_poses=[np.zeros(3),np.zeros(3),np.zeros(3)]
                for i in range(length):
                    final_poses[i]=[zero_poses[i][0] / middles[i], zero_poses[i][1] / middles[i], zero_poses[i][2] / middles[i]]
                    if ideal_num==common_trusted_num_list[i]:
                        real_final_pos = final_poses[i]

                num_list_only_num=common_trusted_num_list   #for output
                for i in files.values():
                    i.close()
            else:
                num_list_noconf=[]
                for unit in num_list:
                    num_list_noconf.append(unit[0])
                num_list_noconf = most_common_numbers(num_list_noconf)
                length=len(num_list_noconf)
                num_list_conf=np.zeros(length)
                for unit in num_list:
                    for index in range(length):
                        if unit[0]==num_list_noconf[index][0]:
                            num_list_conf[index]+=unit[1]
                            break
                average_conf=0
                num_list_only_num=[]
                for i in range(length):
                    num_list_only_num.append(num_list_noconf[i][0])
                    num_list_conf[i]/=num_list_noconf[i][1]
                    average_conf+=num_list_conf[i]
                average_conf/=length
                if length>=3:
                    ideal_num = get_ideal(num_list_only_num)
                else:
                    ideal_num=guess_ideal(num_list_only_num)
                    # print("bad detection")
                    circle_failed[circle_number-1]=1
                object_sum = len(dataList)
                middles=np.zeros(length)
                files={}
                for i in range(length):
                    files[num_list_only_num[i]]=open(os.path.join(path,f'{num_list_only_num[i]}.txt'), 'a')
                for k in range(object_sum):  # 识别到数字的有效
                    for rank in range(length):
                        if num_list_only_num[rank] == dataList[k].get_num():
                            # print(num_list[0])
                            cur_pos = coordinate_change(rank,height=25, pos_=[dataList[k].get_pos()[0], dataList[k].get_pos()[1], dataList[k].get_pos()[2]],
                                                            yaw=dataList[k].get_yaw(), cropTensorList=dataList[k].get_cropTensorList(),
                                                            speed=dataList[k].get_speed())
                            # with open(os.path.join(path,f'{num_list_noconf[rank][0]}.txt'), 'a') as file1:
                            files[num_list_only_num[rank]].write(str(cur_pos[0]) + " " + str(cur_pos[1]) + "\n")
                            zero_poses[rank] += cur_pos
                            middles[rank] += 1
                            break
                final_poses=[np.zeros(3),np.zeros(3),np.zeros(3)]
                for i in range(length):
                    final_poses[i]=[zero_poses[i][0] / middles[i], zero_poses[i][1] / middles[i], zero_poses[i][2] / middles[i]]
                    if ideal_num==num_list_noconf[i][0]:
                        real_final_pos = final_poses[i]
                for i in files.values():
                    i.close()
            # print("the final coordinate1 is:" + str(final_poses[0]))
            # print("the final coordinate2 is:" + str(final_poses[1]))
            # print("the final coordinate3 is:" + str(final_poses[2]))
            # print("the final number is" + str(num_list_noconf))
            # print("ideal_num is: " + str(ideal_num))
            final_pos__=PoseStamped()
            final_pos__.pose.position.x=np.float64(real_final_pos[0])
            final_pos__.pose.position.y=np.float64(real_final_pos[1])
            final_pos__.pose.position.z=np.float64(20)
            # if circle_number==1:
            #     if circle_failed[0]==0 and average_conf>=conf_thresh:
            #         target_pub.publish(final_pos__)
            #         permission_pub.publish(1)
            #         result_pub.publish(str(num_list_only_num))   
            #         # print(1)
            #     else:
            #         # print(2)
            #         permission_pub.publish(0)
            #         result_pub.publish("circle fail")
            # else:
            #     # print(3)
            #     target_pub.publish(final_pos__)
            #     permission_pub.publish(1)
            # # for i in range(100):
            #     result_pub.publish(str(num_list_only_num))  
            savepath=os.path.join(path,"output.txt")
            # with open(savepath, 'a') as file: 
            # with open(f'/home/amov/Desktop/well{folder_name}/output.txt', 'a') as file:
            file.write("the final coordinate1 is:" + str(final_poses[0]) + "\n")
            file.write("the final coordinate2 is:" + str(final_poses[1]) + "\n")
            file.write("the final coordinate3 is:" + str(final_poses[2]) + "\n")
            file.write("the final number is" + str(num_list_only_num) + "\n")
            file.write("ideal_num is: " + str(ideal_num) + "\n")
            file.close()
            print(time.time()-t0)
        if circle_number == 2:
            console.close()
            break

def plot(results):
    cropTensors=[]
    try:
        for result in results:
            # annotatedFrame = result.plot()  # 获取框出的图像
            cropTensors.append(result[2])  # 矩形的四个坐标b
            # cv2.imshow("target", annotatedFrame)
            # 将帧写入视频文件
            # out.write(annotatedFrame)
    except AttributeError:
        print("No result.obb, maybe you have used a classify model")
    return cropTensors

def cropTarget(rawImage, cropTensorList, width, height,mtx,dist,nmtx):
    # 将Tensor转换为列表(该列表内有四个元素，每一个元素是一个坐标)
    # 检查列表长度是否为4，如果不是，则可能存在问题
    if len(cropTensorList) != 4:
        raise ValueError("cropTensor must contain exactly 4 elements")
    h,w=rawImage.shape[:2]
    # flag=True
    # for i in range(4):
    #     if cropTensorList[i][0]>=w or cropTensorList[i][0]<0:
    #         flag=False
    #         break
    #     if cropTensorList[i][1]>=h or cropTensorList[i][1]<0:
    #         flag=False
    #         break
    # 根据条件选择不同的点集合
    rawImage=cv2.undistort(rawImage,mtx,dist,None,nmtx)
    cropTensorList=cv2.undistortPoints(np.resize(cropTensorList,(4,1,2)),mtx,dist,P=nmtx)
    cropTensorList=np.resize(cropTensorList,(4,2))
    if (cropTensorList[0][0] - cropTensorList[1][0]) ** 2 + (cropTensorList[0][1] - cropTensorList[1][1]) ** 2 > (
            cropTensorList[1][0] - cropTensorList[2][0]) ** 2 + (cropTensorList[1][1] - cropTensorList[2][1]) ** 2:
        rectPoints = np.array([cropTensorList[0], cropTensorList[1], cropTensorList[2], cropTensorList[3]],
                              dtype=np.float32)
    else:
        rectPoints = np.array([cropTensorList[3], cropTensorList[0], cropTensorList[1], cropTensorList[2]],
                              dtype=np.float32)

    dstPoints = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)

    affineMatrix = cv2.getPerspectiveTransform(rectPoints, dstPoints)

    return cv2.warpPerspective(rawImage, affineMatrix, (width, height),flags=cv2.INTER_LANCZOS4),cropTensorList

def auto_rotate(img,rank,rotate_num,number,path,circle_number):
    # if retinex==0:
    #     if needRetinex: #img-retinexed  mask-unreti
    #         maskimg=img
    #         img=cm.MSRCP(img)
    #         maskimg=cv2.cvtColor(maskimg,cv2.COLOR_BGR2HSV)
    #         # img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #     else: #img-unreti mask-unreti
    #         img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #         maskimg=img
    # else:
    #     if needRetinex: #img-reti mask-reti
    #         img=cm.MSRCP(img)
    #         # img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #         maskimg=img
    #     else: #img-unreti mask-unreti
    #         img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #         maskimg=img
    
    # # img=cm.laplacian(img)
    # h,w=maskimg.shape[:2]
    # rawmask=getMask(maskimg)
    premaskup=cv2.imread(r"/home/amov/sjtu_asc_v2_ws-main/src/mission_offboard/script/masks/maskup.jpg",cv2.IMREAD_GRAYSCALE)
    premaskdown=cv2.imread(r"/home/amov/sjtu_asc_v2_ws-main/src/mission_offboard/script/masks/maskdown.jpg",cv2.IMREAD_GRAYSCALE)
    # maskup=np.bitwise_and(img,premaskup)
    # maskdown=np.bitwise_and(img,premaskdown)
    ROIup=img[premaskup==255]
    ROIdown=img[premaskdown==255]
    varup=np.var(ROIup)
    vardown=np.var(ROIdown)
    if varup>vardown:
        img=cv2.rotate(img,cv2.ROTATE_180)
    h,w=img.shape[:2]
    img=img[h//3+2*h//3//8:h-2*h//3//8,w//8:w-w//8] #h//3+2*h//3//8
    img=cv2.resize(img,(640,640),interpolation=cv2.INTER_LANCZOS4)
    mean=np.mean(img)
    gamma=np.log(140/255)/np.log(mean/255)
    img=cm.gamma_correction(img,gamma)
    img_=cv2.Laplacian(img,cv2.CV_64F)
    img=np.clip(img.astype(np.float64)-1.5*img_,0,255).astype(np.uint8)
    savepath1=os.path.join(path,f'{circle_number}crop',f"{number}_{rank}_{rotate_num}.jpg")
    cv2.imwrite(savepath1,img)
    # if np.sum(maskdown) > np.sum(maskup):  # 检测下面的部分
    #     img = cv2.rotate(img, cv2.ROTATE_180)
    #     # maskimg=cv2.rotate(maskimg,cv2.ROTATE_180)
    #     if retinex==0 and needRetinex:
    #         maskimg=cv2.rotate(maskimg,cv2.ROTATE_180)
    # if retinex==0 and needRetinex:
    #     img=img[int(h/3):]
    #     img=cv2.resize(img,(640,640),interpolation=cv2.INTER_CUBIC)
    #     # img=cv2.copyMakeBorder(img,30,30,30,30,cv2.BORDER_CONSTANT,value=(0,90,250))
    #     maskimg=maskimg[int(h/3):]
    #     maskimg=cv2.resize(maskimg,(640,640),interpolation=cv2.INTER_CUBIC)
    #     # maskimg=cv2.copyMakeBorder(maskimg,30,30,30,30,cv2.BORDER_CONSTANT,value=(0,90,250))
    # else:
    #     img=img[int(h/3):]
    #     img=cv2.resize(img,(640,640),interpolation=cv2.INTER_CUBIC)
    #     # img=cv2.copyMakeBorder(img,30,30,30,30,cv2.BORDER_CONSTANT,value=(0,90,250))
    #     maskimg=img
    # return img,maskimg

# def apply_num_rec_package(img,maskimg,rank,rotate_num,number,path,mode,circle_number):
#     # if mode=="number_new":
#         # if img is not None:
#     mask = getMask(maskimg)
#     kernel = np.ones((13, 13), np.uint8)
    
#     # savepath1=os.path.join(path,f'{circle_number}masks',f'{rank}_{rotate_num}.jpg')
#     # cv2.imwrite(savepath1, mask)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     # # mask=cv2.GaussianBlur(mask,(11,11),100)
#     # # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     # # mask=cv2.GaussianBlur(mask,(11,11),100)
#     # mask=cv2.dilate(mask,(9,9))
#     # kernel=np.ones((5,5),np.uint8)
#     # mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
#     # kernel=np.ones((9,9),np.uint8)
#     # mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
#     # cv2.imshow('b',mask)
#     # cv2.waitKey(0)
#     mask[0:80,:]=255
#     mask[560:640,:]=255
#     mask[:,0:80]=255
#     mask[:,560:640]=255
#     hstart=80
#     hend=560
#     width=480
#     for i in range(480):
#         if np.sum(mask[i+80,80:560])<0.05*width*255:
#             mask[i+80,80:560]=255
#             hstart+=1
#         else:
#             break
#     for i in range(480):
#         if np.sum(mask[559-i,80:560])<0.05*width*255:
#             mask[559-i,80:560]=255
#             hend-=1
#         else:
#             break
#     height=hend-hstart
#     wstart=80
#     wend=560
#     for i in range(480):
#         if np.sum(mask[hstart:hend,i+80])<0.05*height*255:
#             mask[hstart:hend,i+80]=255
#             wstart+=1
#         else:
#             break
#     for i in range(480):
#         if np.sum(mask[hstart:hend,559-i])<0.05*height*255:
#             mask[hstart:hend,559-i]=255
#             wend-=1
#         else:
#             break
#     kernels=[9,15,25]
#     for kernel in kernels:
#         k=cv2.getStructuringElement(cv2.MORPH_RECT,(kernel,kernel))
#         mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
#     # _, mask = cv2.threshold(mask, 170, 255, cv2.THRESH_BINARY)

#     # savepath2=os.path.join(path,f'{circle_number}masks_',f'{rank}_{rotate_num}.jpg')
#     # cv2.imwrite(savepath2, mask)
#     if mode=="number_new":
#         contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#         areas= []
#         quad = []
#         # print(contours)
#         for cnt in contours:
#             epsilon = 0.05* cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, epsilon, True)
#             hull=cv2.convexHull(approx)
#             epsilon=0.05*cv2.arcLength(hull,True)
#             approx=cv2.approxPolyDP(hull,epsilon,True)
#             if len(approx) == 4 :
#                 area = cv2.contourArea(approx)
#                 if cm.legal_region(area,approx):  
#                     areas.append(area)
#                     quad.append(approx)
#             elif len(approx)>4:
#                 area = cv2.contourArea(approx)
#                 if cm.legal_region(area,approx):
#                     rect=cv2.minAreaRect(cnt)
#                     box=cv2.boxPoints(rect)
#                     areas.append(area)
#                     approx=box.reshape((-1,2))
#                     quad.append(approx)
#         areas=np.array(areas)
#         quad=np.array(quad)
#         no_crop_flag=False
#         if areas.size !=0:
#             # print(file_name,":")
#             # print(best_quad)
#             if areas.size==1:
#                 points=quad[0].reshape(-1,2)
#                 sum_xy=points.sum(axis=1)
#                 diff_xy=points[:,0]-points[:,1]
#                 corners=np.zeros((4,2),dtype=np.float32)
#                 corners[0]=points[np.argmin(sum_xy)]
#                 corners[1]=points[np.argmax(diff_xy)]
#                 corners[2]=points[np.argmax(sum_xy)]
#                 corners[3]=points[np.argmin(diff_xy)]
#                 corners[0]-=30
#                 corners[2]+=30
#                 corners[1][0]+=30
#                 corners[1][1]-=30
#                 corners[3][0]-=30
#                 corners[3][1]+=30
#                 pts_src=np.float32([
#                     corners[0],
#                     corners[1],
#                     corners[2],
#                     corners[3]
#                 ])
#                 # print(name,pts_src)
#                 pts_dest=np.float32([
#                         [0,0],
#                         [639,0],
#                         [639,639],
#                         [0,639],
#                     ]
#                 )
#                 M=cv2.getPerspectiveTransform(pts_src,pts_dest)
#                 img=cv2.warpPerspective(img,M,(640,640),borderValue=(255,255,255),flags=cv2.INTER_CUBIC)
#             else:
#                 min1_val = np.min(areas)
#                 min1_idx = np.argmin(areas)

#                 # 把该值临时设置为无穷大
#                 arr_temp = areas.copy()
#                 arr_temp[min1_idx] = np.inf

#                 # 再找第二小的
#                 min2_val = np.min(arr_temp)
#                 min2_idx = np.argmin(arr_temp)
#                 points1=quad[min1_idx].reshape(-1,2)
#                 points2=quad[min2_idx].reshape(-1,2)
#                 points=np.concatenate((points1,points2),axis=0)
#                 # print(name,points)
#                 sum_xy=points.sum(axis=1)
#                 diff_xy=points[:,0]-points[:,1]
#                 corners=np.zeros((4,2),dtype=np.float32)
#                 corners[0]=points[np.argmin(sum_xy)]
#                 corners[1]=points[np.argmax(diff_xy)]
#                 corners[2]=points[np.argmax(sum_xy)]
#                 corners[3]=points[np.argmin(diff_xy)]
#                 corners[0]-=30
#                 corners[2]+=30
#                 corners[1][0]+=30
#                 corners[1][1]-=30
#                 corners[3][0]-=30
#                 corners[3][1]+=30
#                 pts_src=np.float32([
#                     corners[0],
#                     corners[1],
#                     corners[2],
#                     corners[3]
#                 ])
#                 # print(name,pts_src)
#                 pts_dest=np.float32([
#                         [0,0],
#                         [639,0],
#                         [639,639],
#                         [0,639],
#                     ]
#                 )
#                 M=cv2.getPerspectiveTransform(pts_src,pts_dest)
#                 img=cv2.warpPerspective(img,M,(640,640),borderValue=(255,255,255),flags=cv2.INTER_CUBIC)
#                 # print(f"{name}:divi")
#         else:
#             # print(name,"no")
#             # print(contours)
#             img=img[80:560,80:560]
#             no_crop_flag=True
#             if hstart-80<=120:
#                 newhstart=hstart-80
#             else:
#                 newhstart=0
#             if hend-80>=360:
#                 newhend=hend-80
#             else:
#                 newhend=480
#             if wstart-80<=120:
#                 newwstart=wstart-80
#             else:
#                 newwstart=0
#             if wend-80>=360:
#                 newwend=wend-80
#             else:
#                 newwend=480
            

#         img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

#         # img=cm.biFilter(img)
#         # img=cm.CLAHE_and_wiener(img)
#         # img=biFilter(img)
#         img=cm.laplacian(img)
#         # img=biFilter(img)
#         # img=opening(img)
#         # img=cm.opening(img,15)
#         # img=cm.opening(img,21)
#         # img=cm.opening(img,27),
#         # img=cm.closing(img,15)
#         # img=cm.biFilter(img)
#         # return img,no_crop_flag,newhstart,newhend,newwstart,newwend

#         # flag=False
        
#         # if mode=="number_new":

#         h,w=img.shape[:2]
#         img1=img[:,:int(6*w/11)]
#         img2=img[:,int(5*w/11):]
#         if no_crop_flag:
#             img1=img1[newhstart:newhend,newwstart:]
#             img2=img2[newhstart:newhend,:newwend-int(5*w/11)]
#         # img1=img1[:,newwstart:]
#         # img2=img2[:,:newwend]
#         img1=cv2.resize(img1,(640,640),interpolation=cv2.INTER_CUBIC)
#         img2=cv2.resize(img2,(640,640),interpolation=cv2.INTER_CUBIC)
#         savepath1=os.path.join(path,f'{circle_number}crop1',f"{number}_{rank}_{rotate_num}.jpg")
#         savepath2=os.path.join(path,f'{circle_number}crop2',f"{number}_{rank}_{rotate_num}.jpg")
#         # os.makedirs(savepath1,exist_ok=True)
#         # os.makedirs(savepath2,exist_ok=True)
#         cv2.imwrite(savepath1,img1)
#         cv2.imwrite(savepath2,img2)
#         # results_classify1 = modelClassify.predict(
#         #     source=img1,
#         #     imgsz=640,
#         #     device='0',
#         #     save=False,
#         #     workers=4
#         # )
#         # results_classify2=modelClassify.predict(
#         #     source=img2,
#         #     imgsz=640,
#         #     device='0',
#         #     save=False,
#         #     workers=4
#         # )
#         # num=10*int(results_classify1[0].probs.top1)+int(results_classify2[0].probs.top1)

#         # if results_classify1[0].probs.top1conf>=0.7 and results_classify2[0].probs.top1conf>=0.7:
#         #     trusted_num_list.append(num)
#         #     flag=True
#         # num_and_prob=(10*int(results_classify1[0].probs.top1)+int(results_classify2[0].probs.top1),results_classify1[0].probs.top1conf*results_classify2[0].probs.top1conf)
#         # print("Classify num is:" + str(num))
#         # savepath=os.path.join(path,'output.txt')
#         # with open(savepath, 'a') as file:
#         #     file.write("Classify num is:" + str(num) +' '+str(results_classify1[0].probs.top1conf)+' '+str(results_classify2[0].probs.top1conf) +"\n")
#         # num_list.append(num_and_prob)
#         # return num,flag
#     # else:
#     #     if no_crop_flag:
#     #         img=img[newhstart:newhend,newwstart:newwend]
#     #         img=cv2.resize(img,(640,640),interpolation=cv2.INTER_CUBIC)
#     #     savepath1=os.path.join(path,f'{circle_number}crop',f"{number}_{rank}_{rotate_num}.jpg")
#     #     cv2.imwrite(savepath1,img)
#         # results_classify = modelClassify.predict(
#         #     source=img,
#         #     imgsz=640,
#         #     device='0',
#         #     save=False,
#         #     workers=4
#         # )
#         # if mode=="number_old":
#         #     num=int(results_classify[0].probs.top1)
#         # else:
#         #     num=pattern_dic[int(results_classify[0].probs.top1)]
#         # if results_classify[0].probs.top1conf>=0.7:
#         #     trusted_num_list.append(num)
#         #     flag=True
#         # num_and_prob=(num,results_classify[0].probs.top1conf)
#         # print("Classify num is:" + str(num))
#         # savepath=os.path.join(path,'output.txt')
#         # with open(savepath, 'a') as file:
#         #     file.write("Classify num is:" + str(num) + str(results_classify[0].probs.top1conf)+"\n")
#         # num_list.append(num_and_prob)
#         # return num,flag
#     else:
#         img=img[80:560,80:560]
#         if hstart-80<=120:
#             newhstart=hstart-80
#         else:
#             newhstart=0
#         if hend-80>=360:
#             newhend=hend-80
#         else:
#             newhend=480
#         if wstart-80<=120:
#             newwstart=wstart-80
#         else:
#             newwstart=0
#         if wend-80>=360:
#             newwend=wend-80
#         else:
#             newwend=480

#         img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

#         # img=cm.biFilter(img)
#         # img=cm.CLAHE_and_wiener(img)
#         # img=biFilter(img)
#         img=cm.laplacian(img)
#         # img=biFilter(img)
#         # img=opening(img)
#         # img=cm.opening(img,15)
#         # img=cm.opening(img,21)
#         # img=cm.opening(img,27),
#         # img=cm.closing(img,15)
#         # img=cm.biFilter(img)
#         img=img[newhstart:newhend,newwstart:newwend]
#         img=cv2.resize(img,(640,640),interpolation=cv2.INTER_CUBIC)
#         savepath1=os.path.join(path,f'{circle_number}crop',f"{number}_{rank}_{rotate_num}.jpg")
#         cv2.imwrite(savepath1,img)



def cls_predict(arg):
    rank=arg[0]
    results=arg[1]
    path=arg[2]
    circle_number=arg[3]
    mtx=arg[4]
    dist=arg[5]
    nmtx=arg[6]
    
    tensors=[]
    cropTensorsList = plot(results=results)
    rotate_num=0

    for i,cropTensors in enumerate(cropTensorsList):
        number=int(results[i][0].split('/')[-1].split('.')[0])
        for cropTensor in cropTensors:
            cropTensorList=cropTensor.tolist()
            framet,cropTensorList = cropTarget(results[i][1], cropTensorList, 320, 640,mtx,dist,nmtx)
            
                # savepath1=os.path.join(path,f'{circle_number}origin',f'{rank}_{origin_num}.jpg')
            # cv2.imwrite(savepath1, framet)
            # cv2.imshow("five", framet)
            auto_rotate(framet,rank,rotate_num,number,path,circle_number)
                # savepath2=os.path.join(path,f'{circle_number}rotate',f'{rank}_{rotate_num}.jpg')
                # cv2.imwrite(savepath2, cv2.cvtColor(framet,cv2.COLOR_HSV2BGR))
            # apply_num_rec_package(framet,maskimg,rank,rotate_num,number,path,mode,circle_number)
            # else:
            #     apply_num_rec_package(framet,maskimg,rank,rotate_num,number)
                # answers.append(img)
            rotate_num += 1
            tensors.append(cropTensorList)
                # alldata=alldataList[mapofclear[i+bias]]
                # alldata=alldataList[mapofclear[int(results[i].path.split('/')[-1].split('.')[0])]]
                # imgdata = pos.Imgdata(pos=alldata.pos,num=-1,
                #                       yaw=alldata.yaw,cropTensorList=cropTensorList,
                #                       speed=alldata.speed)
                # dataList.append(imgdata)
                # if trusted_flag:
                #     trusted_dataList.append(imgdata)
                # cv2.imshow("img_num" + str(j), img_num)
     
    return tensors


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn',force=True)
    main()
