#!/usr/bin/env python3
# coding: utf-8
import rospy
from Speech_Lib import Speech
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32, Int32, Bool
from Rosmaster_Lib import Rosmaster
car = Rosmaster()
spe = Speech()
class Ctrl_driver:
    def __init__(self):
        rospy.on_shutdown(self.cancel)	
        self.sub_cmd_vel = rospy.Subscriber('cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)
        self.sub_JoyState = rospy.Subscriber('/JoyState', Bool, self.JoyStateCallback)
        self.velPublisher = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.Joy_active = False
    def cmd_vel_callback(self,msg):
		# 小车运动控制，订阅者回调函数
		# Car motion control, subscriber callback function
        if not isinstance(msg, Twist): return
		# 下发线速度和角速度
		# Issue linear vel and angular vel
        vx = msg.linear.x
        vy = msg.linear.y
        angular = msg.angular.z
		# 小车运动控制,vel: ±1, angular: ±5
		# Trolley motion control,vesl=[-1, 1], angular=[-5, 5]
		# rospy.loginfo("cmd_velx: {}, cmd_vely: {}, cmd_ang: {}".format(vx, vy, angular))
        car.set_car_motion(vx, vy, angular)
    def JoyStateCallback(self,msg):
        if not isinstance(msg, Bool): return
        self.Joy_active = msg.data
        self.velPublisher.publish(Twist())
    def cancel(self):
        self.velPublisher.unregister()
        self.sub_JoyState.unregister()
        self.sub_cmd_vel.unregister()
        rospy.loginfo("Close the robot...")
        rospy.sleep(1)
ctrl = Ctrl_driver()
if __name__ == '__main__':
    rospy.init_node("driver_node", anonymous=False)
    #ctrl = Ctrl_driver()
    try:
         while not rospy.is_shutdown():
            speech_r = spe.speech_read()
            if speech_r == 2 :
                vx = 0.0
                vy = 0.0
                angular = 0
                car.set_car_motion(vx, vy, angular)
                spe.void_write(speech_r)
# 前进
            elif speech_r == 4 :
                vx = 0.5
                vy = 0.0
                angular = 0
                car.set_car_motion(vx, vy, angular)
                spe.void_write(speech_r)
# 后退
            elif speech_r == 5 :
                vx = -0.5
                vy = 0.0
                angular = 0
                car.set_car_motion(vx, vy, angular)
                spe.void_write(speech_r)
# 左转
            elif speech_r == 6 :
                vx = 0.2
                vy = 0.0
                angular = 0.5
                car.set_car_motion(vx, vy, angular)
                spe.void_write(speech_r)
# 右转    
            elif speech_r == 7 :
                vx = 0.2
                vy = 0.0
                angular = -0.5
                car.set_car_motion(vx, vy, angular) 
                spe.void_write(speech_r)
# 左旋  
            elif speech_r == 8 :
                vx = 0.0
                vy = 0.0
                angular = 0.5
                car.set_car_motion(vx, vy, angular)
                spe.void_write(speech_r)
# 右旋
            elif speech_r == 9 :
                vx = 0.0
                vy = 0.0
                angular = -0.5
                car.set_car_motion(vx, vy, angular)
                spe.void_write(speech_r)
            elif speech_r == 11 :
                car.set_colorful_lamps(0xFF,255,0,0)
                spe.void_write(speech_r) 

            elif speech_r == 17 :
                car.set_colorful_effect(3, 6, parm=1)
                spe.void_write(speech_r)

            elif speech_r == 10 :
                car.set_colorful_effect(0, 6, parm=1)
                spe.void_write(speech_r)
# 亮绿灯
            elif speech_r == 12 :
                car.set_colorful_lamps(0xFF,0,255,0)
                spe.void_write(speech_r)
# 亮蓝灯
            elif speech_r == 13 :
                car.set_colorful_lamps(0xFF,0,0,255)
                spe.void_write(speech_r)
# 亮黄灯
            elif speech_r == 14 :
                car.set_colorful_lamps(0xFF,255,255,0)
                spe.void_write(speech_r)
# 打开流水灯
            elif speech_r == 15 :
                car.set_colorful_effect(1, 6, parm=1)
                spe.void_write(speech_r)
# 打开渐变灯
            elif speech_r == 16 :
                car.set_colorful_effect(4, 6, parm=1)
                spe.void_write(speech_r)
# 打开呼吸灯
            elif speech_r == 17 :
                car.set_colorful_effect(3, 6, parm=1)
                spe.void_write(speech_r)
# 显示电量
            elif speech_r == 18 :
                car.set_colorful_effect(1, 6, parm=1)
                spe.void_write(speech_r)
# 语音模块调节
            elif speech_r == 32 :
                spe.void_write(speech_r)

            elif speech_r == 33 :
                spe.void_write(speech_r)

            elif speech_r == 34 :
                spe.void_write(speech_r)

            elif speech_r == 35 :
                spe.void_write(speech_r)

            elif speech_r == 36 :
                spe.void_write(speech_r)

            elif speech_r == 37 :
                spe.void_write(speech_r)


    except KeyboardInterrupt:
        print("Close!")
