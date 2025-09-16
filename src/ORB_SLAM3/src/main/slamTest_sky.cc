#include<iostream>
#include<algorithm>
#include<fstream>
#include<sstream>
#include<vector>
#include<chrono>
#include<queue>
#include<mutex>
#include<cmath>
#include<chrono>
#include<System.h>
#include<ImuTypes.h>
#include<string>

#include<opencv2/opencv.hpp>
#include<Eigen/Dense>
#include<sophus/se3.hpp>

#include <ros/ros.h>
#include <tf/transform_datatypes.h> 
#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <picname_and_time_msgs/picnameAndTime.h>

#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/GPSRAW.h>
#include <mavros_msgs/Altitude.h>
#include <mavros_msgs/VFR_HUD.h>
#include <mavros_msgs/Thrust.h>
#include <mavros_msgs/WaypointReached.h>
#include <mavros_msgs/RCIn.h>

using namespace std;

// struct point
// {
//     int x,y,z;
// }
mutex queue_mtx,firstLoopEndFlag_mtx,firstScoutFlag_mtx,secondLoopEndFlag_mtx,slamTarget_mtx,traditionalTarget_mtx,wayPoint_mtx,globalPosition_mtx;

queue<string> nameQ;
queue<double> timeQ;
// queue<float> hQ;
// queue<float> biasAngleQ;
int firstLoopEndFlag=0,firstScoutFlag=-1,secondLoopFlag=0,secondLoopEndFlag=0,wayPoint=0;
int a=0;
geometry_msgs::Point slamTarget;
geometry_msgs::PoseStamped traditionalTarget;
float global_x,global_y,global_z,global_yaw;

void pic_sub_func(const picname_and_time_msgs::picnameAndTime::ConstPtr &picInfo)
{
    unique_lock<mutex> lock(queue_mtx);
    nameQ.push(picInfo->name);
    timeQ.push(picInfo->time);
    // float h=picInfo->h;
    // if(h!=0)
    // {
    //     hQ.push(h);
    //     biasAngleQ.push(picInfo->biasAngle) ////no longer need to receive h and  bias, calculate yourself, but need more ros topic!
    // }
}

void firstLoopEnd_sub_func(const std_msgs::Int32::ConstPtr &firstLoopEndFlag_)
{
    unique_lock<mutex> lock(firstLoopEndFlag_mtx);
    firstLoopEndFlag=firstLoopEndFlag_->data;
}

// void firstScout_sub_func(const std_msgs::Int32::ConstPtr &firstScoutFlag_)
// {
//     unique_lock<mutex> lock(firstScoutFlag_mtx);
//     firstScoutFlag=firstScoutFlag_->data;
// }

// void secondLoopEnd_sub_func(const std_msgs::Int32::ConstPtr &secondLoopEndFlag_)
// {
//     unique_lock<mutex> lock(secondLoopEndFlag_mtx);
//     secondLoopEndFlag=secondLoopEndFlag_->data;
// }

void slamTarget_sub_func(const geometry_msgs::Point::ConstPtr &slamTarget_)
{
    unique_lock<mutex> lock(slamTarget_mtx);
    slamTarget=*slamTarget_;
}

void traditionalTarget_sub_func(const geometry_msgs::PoseStamped::ConstPtr &traditionalTarget_)
{
    unique_lock<mutex> lock(traditionalTarget_mtx);
    traditionalTarget=*traditionalTarget_;
}

void wayPoint_sub_func(const mavros_msgs::WaypointReached::ConstPtr &wayPoint_)
{
    unique_lock<mutex> lock(wayPoint_mtx);
    wayPoint=wayPoint_->wp_seq;
}

void globalPosition_sub_func(const nav_msgs::Odometry::ConstPtr &globalPosition_)
{
    
    unique_lock<mutex> lock(globalPosition_mtx);
    global_x=globalPosition_->pose.pose.position.x;
    global_y=globalPosition_->pose.pose.position.y;
    global_z=globalPosition_->pose.pose.position.z;
    tf::Quaternion q(globalPosition_->pose.pose.orientation.x, globalPosition_->pose.pose.orientation.y, globalPosition_->pose.pose.orientation.z, globalPosition_->pose.pose.orientation.w);
    
    double roll, pitch, yaw;
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw); 
    global_yaw=yaw;
}

int main(int argc,char **argv)
{
    /// ./summer2025 path/to/voc path/to/setting wayPointStartToTrackForBombing
    string bombPoint=argv[3];
    // int l=bombPoint.length();
    // int bombPoint_int=0;
    // for(int i=0;i<l;++i)
    // {
    //     bombPoint_int+=(bombPoint[0]-'0')*pow(10,l-1-i);
    // }
    int bombPoint_int=stoi(bombPoint);
    ros::init(argc, argv, "slam");
    ros::NodeHandle nh;
    ros::Rate rate1(30);
    // ros::Rate rate2(30);
    ros::Subscriber pic_sub=nh.subscribe<picname_and_time_msgs::picnameAndTime>("picnameAndTime",1,pic_sub_func);
    ros::Subscriber firstLoopEnd_sub=nh.subscribe<std_msgs::Int32>("firstLoopEnd",1,firstLoopEnd_sub_func);
    // ros::Subscriber firstScout_sub=nh.subscribe<std_msgs::Int32>("firstScout",1,firstScout_sub_func);
    // ros::Subscriber secondLoopEnd_sub=nh.subscribe<std_msgs::Int32>("secondLoopEnd",1,secondLoopEnd_sub_func);
    ros::Subscriber slamTarget_sub=nh.subscribe<geometry_msgs::Point>("slamTarget",1,slamTarget_sub_func);
    ros::Subscriber traditionalTarget_sub=nh.subscribe<geometry_msgs::PoseStamped>("traditionalTarget",1,traditionalTarget_sub_func);
    ros::Subscriber wayPoint_sub=nh.subscribe<mavros_msgs::WaypointReached>("/mavros/mission/reached", 1, wayPoint_sub_func);
    ros::Subscriber globalPosition_sub=nh.subscribe<nav_msgs::Odometry>("/mavros/global_position/local", 1, globalPosition_sub_func);
    // ros::Publisher firstTrack_pub=nh.advertise<std_msgs::Int32>("firstTrack",1)
    ros::Publisher referenceMode_pub=nh.advertise<std_msgs::Int32>("referenceMode",1);
    // ros::Publisher distance_pub=nh.advertise<std_msgs::Float32>("distance",1);
    // ros::Publisher angle_pub=nh.advertise<std_msgs::Float32>("angle",1);
    // ros::Publisher permission_pub=nh.advertise<std_msgs::Float64>("permission",1);
    // ros::Publisher finalPos_pub=nh.advertise<geometry_msgs::PoseStamped>("final_pos",1);
    ros::AsyncSpinner spinner(1);
    spinner.start();

    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR, false);
    cout<<"SLAM initialized"<<endl;
    string curName;
    double curTime;
    int flag=0; //set flag as int because it's received from ros
    bool resetActiveMapFlag=false;
    double t0=-1;
    bool emptyFlag=true;
    while(!flag||!emptyFlag)
    {
        if(!emptyFlag)
        {
            {
                unique_lock<mutex> lock(queue_mtx);
                curName=nameQ.front();
                curTime=timeQ.front();//此time应以秒计
                nameQ.pop();
                timeQ.pop();
            }
            if(t0==-1)
            {
                t0=curTime;
            }
            SLAM.TrackMonocular(cv::imread(curName,cv::IMREAD_GRAYSCALE),curTime,vector<ORB_SLAM3::IMU::Point>(),curName,true);
            if(SLAM.GetTrackingState()==1&&curTime-t0>=3)
            {
                SLAM.ResetActiveMap();
                resetActiveMapFlag=true;
                break;
            }
            //先判断有没有initialized，若在3s内未完成，视同lost，若完成，才判断后续有无lost
            if(SLAM.isLost())///若第一次lost且keyframe过少，应舍去，否则保留，还要注意修改relocalizemonocular中相关逻辑，因为涉及reset
            {
                resetActiveMapFlag=SLAM.getResetActiveMapFlag();
                break;
            }
        }
        {
            unique_lock<mutex> lock(firstLoopEndFlag_mtx);
            flag=firstLoopEndFlag;
        }
        {
            unique_lock<mutex> lock(queue_mtx);
            emptyFlag=nameQ.empty();
        }
        // ros::spinOnce();
        rate1.sleep();
    }
    // std_msgs::Int32 firstTrackFlag;
    // firstTrackFlag=int (!(SLAM.isLost()));//不用发出了，直接接受关于confidence high/low的值，本地判断
    // firstTrack_pub.publish(firstTrackFlag);
    int abandon;
    if(!resetActiveMapFlag)
    {
        SLAM.SaveFrame(1);
        abandon=1;
    }
    else
    {
        abandon=0;
    }


//     while(true)
//     {
//         {
//             unique_lock<mutex> lock(firstScoutFlag_mtx);
//             if(firstScoutFlag!=-1) 
//                 break;
//         }
//         // ros::spinOnce();
//         rate1.sleep();
//     }
//     std_msgs::Int32 permission_ros;
//     if(SLAM.isLost()||resetActiveMapFlag)//secondLoopFlag==1 -> scout again do not expand (if track success in first, but confidence is low) | 2->scout again expand  | 3 -> no more scout (track success, confidence high)
//     {
//         secondLoopFlag=2;
//         permission_ros.data=0;
//         permission_pub.publish(permission_ros);
//     }
//     else if(firstScoutFlag==0)
//     {
//         secondLoopFlag=1;
//         permission_ros.data=0;
//         permission_pub.publish(permission_ros);
//     }
//     else
//     {
//         secondLoopFlag=3;
//         permission_ros.data=1;
//         permission_pub.publish(permission_ros);
//         finalPos_pub.publish(traditionalTarget);
//     }
    // int abandon=-1;//0->fail 1->1st 2->2nd 3->1st&2nd
//     flag=0;
//     emptyFlag=true;
//     bool relocalized=false;
//     switch(secondLoopFlag)
//     {
//     case 1:
//         SLAM.ActivateLocalizationMode();
//         while(!flag||!emptyFlag)
//         {
// ///cannot depend on the state prodeced by last loop since it's uncertain, have to specify? or rewrite the track() logic
// ///relocalization() repeatively until return true, then change the state and trackingmode(?), may can track() as usual
//             if(!emptyFlag)
//             {
//                 {
//                     unique_lock<mutex> lock(queue_mtx);
//                     curName=nameQ.front();
//                     curTime=timeQ.front();
//                     nameQ.pop();
//                     timeQ.pop();
//                 }
//                 if(!relocalized)
//                 {
//                     SLAM.RelocalizeMonocular(cv::imread(curName,cv::IMREAD_GRAYSCALE),curTime,curName,relocalized);
//                 }
//                 else
//                 {
//                     SLAM.TrackMonocular(cv::imread(curName,cv::IMREAD_GRAYSCALE),curTime,vector<ORB_SLAM3::IMU::Point>(),curName,true);
//                     ///即使recentlylost也不中断，因为可能可以根据第一轮relocalize回来，同时修改了save逻辑，recentlylost不再保存
//                     ///recentlylost 确无保存必要，因为点太少
//                 }
//             }
//             {
//                 unique_lock<mutex> lock(secondLoopEndFlag_mtx);
//                 flag=secondLoopEndFlag;
//             }
//             {
//                 unique_lock<mutex> lock(queue_mtx);
//                 emptyFlag=nameQ.empty();
//             }
//             rate1.sleep();
//         }
//         if(relocalized)
//         {
//             abandon=3;
//         }
//         else
//         {
//             abandon=1;
//         }
//         break;
//     case 2:
//         if(resetActiveMapFlag)
//         {
//             t0=-1;
//             while(!flag||!emptyFlag)
//             {
//                 if(!emptyFlag)
//                 {
//                     {
//                         unique_lock<mutex> lock(queue_mtx);
//                         curName=nameQ.front();
//                         curTime=timeQ.front();//此time应以秒计
//                         nameQ.pop();
//                         timeQ.pop();
                        
//                     }
//                     if(t0==-1)
//                     {
//                         t0=curTime;
//                     }
//                     SLAM.TrackMonocular(cv::imread(curName,cv::IMREAD_GRAYSCALE),curTime,vector<ORB_SLAM3::IMU::Point>(),curName,true);
//                     if(SLAM.GetTrackingState()==1&&curTime-t0>=5)
//                     {
//                         abandon=0;
//                         break;
//                     }///如果是initialize失败直接放弃slam导航，如果是lost可以在结果里翻一下，如果有靶标相关点，在投弹时继续追踪，哪怕只有一次relocalize，也能调整一次，其他时候采取坐标计算
//                     if(SLAM.isLost())
//                     {
//                         if(SLAM.getResetActiveMapFlag())
//                         {
//                             abandon=0;
//                         }
//                         else
//                         {
//                             abandon=2;
//                         }
//                         break;
//                     }
//                 }
//                 {
//                     unique_lock<mutex> lock(secondLoopEndFlag_mtx);
//                     flag=secondLoopEndFlag;
//                 }
//                 {
//                     unique_lock<mutex> lock(queue_mtx);
//                     emptyFlag=nameQ.empty();
//                 }
//                 rate1.sleep();
//             }
//             if(!SLAM.isLost())
//             {
//                 abandon=2;
//             }
//         }
//         else
//         {
//             while(!flag||!emptyFlag)
//             {
//                 if(!emptyFlag)
//                 {
//                     {
//                         unique_lock<mutex> lock(queue_mtx);
//                         curName=nameQ.front();
//                         curTime=timeQ.front();//此time应以秒计
//                         nameQ.pop();
//                         timeQ.pop();
                        
//                     }
//                     if(!relocalized)
//                     {
//                         SLAM.RelocalizeMonocular(cv::imread(curName,cv::IMREAD_GRAYSCALE),curTime,curName,relocalized);
//                     }
//                     else
//                     {
//                         SLAM.TrackMonocular(cv::imread(curName,cv::IMREAD_GRAYSCALE),curTime,vector<ORB_SLAM3::IMU::Point>(),curName,true);
//                     }
//                 }
//                 {
//                     unique_lock<mutex> lock(secondLoopEndFlag_mtx);
//                     flag=secondLoopEndFlag;
//                 }
//                 {
//                     unique_lock<mutex> lock(queue_mtx);
//                     emptyFlag=nameQ.empty();
//                 }
//                 rate1.sleep();
//             }
//             if(relocalized)//可以考虑加一个若recentlylost大于一定时间也abandon=1
//             {
//                 abandon=3;
//             }
//             else
//             {
//                 abandon=1;
//             }
//         }
//         break;
//     case 3:
//         abandon=1;
//     }
//     if(abandon==2||abandon==3)
//     {
//         SLAM.SaveFrame(2);
//     }
//     permission_ros.data=1;
//     permission_pub.publish(permission_ros);

    std_msgs::Int32 referenceMode;
    referenceMode.data=abandon;
    referenceMode_pub.publish(referenceMode);
    //根据abandon进行数据挑选和计算
    float x=0,y=0,z=0;
    while(true&&abandon!=0)
    {
        {
            unique_lock<mutex> lock(slamTarget_mtx);
            x=slamTarget.x;
            y=slamTarget.y;
            z=slamTarget.z;
        }
        if(x!=0||y!=0||z!=0)
        {
            if(x==-1&&y==-1&&z==-1)
            {
                abandon=0;
            }
            break;
        }
        rate1.sleep();
    }
    if(abandon!=0)//may can consider give the use of camera to this prog since then
    {///todo get the camera, get picture and calculate, use the data get, and change the control logic using waypoint
        stringstream s;
        s<<curName;
        vector<string> v;
        string part;
        while(getline(s,part,'/'))
        {
            v.push_back(part);
        }
        v.pop_back();
        v.pop_back();
        v.erase(v.begin());
        string logname;
        for(auto &token:v)
        {
            logname+='/';
            logname+=token;
        }
        logname+='/';
        logname+="slamlog.txt";
        ofstream f(logname);  //todo create log
        int cameraID=0;
        while(cameraID<=30)
        {
            cv::VideoCapture cap(cameraID);
            if(!cap.isOpened())
            {
                cameraID+=1;
            }
            else
            {
                cap.release();
                break;
            }
        }
        cv::VideoCapture cap(cameraID);
        cap.set(cv::CAP_PROP_FOURCC,  cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        // cap.set(cv::CAP_PROP_FOURCC,cv::VideoWriter_fourcc(*'MJPG'))
        cap.set(cv::CAP_PROP_FPS, 15);
        int fw=2560;
        int fh=1440;
        cap.set(cv::CAP_PROP_FRAME_WIDTH, fw);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, fh);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        bool relocalized=false;
        Sophus::SE3f currentPose,inversedPose;
        Eigen::Vector3f rawTarget(x,y,z);
        Eigen::Vector3f newTarget,cameraPoint;
        float scale,dist;
        SLAM.ActivateLocalizationMode();
        emptyFlag=true;
        cv::Mat frame;
        float x,y,z,yaw;
        int curWayPoint;
        while(true)
        {
            {
                unique_lock<mutex> lock(wayPoint_mtx);
                curWayPoint=wayPoint;
            }
            if(curWayPoint>=bombPoint_int)
            {
                {
                    unique_lock<mutex> lock(globalPosition_mtx);
                    x=global_x;
                    y=global_y;
                    z=global_z;
                    yaw=global_yaw;
                }
                cap>>frame;
                cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
                
                // auto now = std::chrono::high_resolution_clock::now();
                // auto ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
                // auto epoch = ns.time_since_epoch();  
                // long long nanoseconds = epoch.count();
                curTime= 1e-9*std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()).time_since_epoch().count();
                if(!relocalized)
                {
                    currentPose=SLAM.RelocalizeMonocular(frame,curTime,"",relocalized);
                }
                else
                {
                    currentPose=SLAM.TrackMonocular(frame,curTime,vector<ORB_SLAM3::IMU::Point>(),"",true);
                }
                if(relocalized&&SLAM.GetTrackingState()==2)
                {
                    newTarget=currentPose*rawTarget;
                    scale=fabs(z/newTarget[2]);
                    float x_=scale*newTarget[0],y_=scale*newTarget[1];
                    float xBias=-y_*cos(yaw)+x_*sin(yaw),yBias=-y_*sin(yaw)-x_*cos(yaw);
                    float xReal=x+xBias,yReal=y+yBias;
                    // rawDist=sqrt(pow(newTarget[0],2)+pow(newTarget[1],2));
                    // newDist=scale*rawDist;
                    // angle=atan2(newTarget[1],newTarget[0])+3.1415926536/2;
                    // std_msgs::Float32 rosDist=newDist;
                    // std_msgs::Float32 rosAngle=angle+biasAngle;
                    // distance_pub.publish(rosDist);
                    // angle_pub.publish(rosAngle);
                    geometry_msgs::PoseStamped finalPos;
                    finalPos.pose.position.x=xReal;
                    finalPos.pose.position.y=yReal;
                    // finalPos_pub.publish(finalPos);
                    f<<x_<<' '<<y_<<' '<<xBias<<' '<<yBias<<' '<<xReal<<' '<<yReal<<endl;
                    dist=sqrt(pow(xBias,2)+pow(yBias,2));
                    if(dist<=20)
                    {
                        break;
                    }
                    // inversedPose=currentPose.inverse();
                    // cameraPoint=inversedPose.translation();
                }
            }
            rate1.sleep();
        }
    }
    ///shutdown etc...
    spinner.stop();
    SLAM.Shutdown();
    return 0;
}