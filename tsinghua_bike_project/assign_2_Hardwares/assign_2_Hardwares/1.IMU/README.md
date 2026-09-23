# IMU配置

IMU通过USB连接到树莓派上。树莓派通过读取USB中的数据，就能获取到IMU的所有数据。

IMU封装得太好了！总共就两步：
1. 运行它的库安装脚本
2. 往树莓派上插IMU并运行代码

## 运行安装脚本

找到 Driver -> YbImuLib，并把该文件夹复制到树莓派里

```bash
cd xxxxx/YbImuLib
sudo python3 setup.py install
sudo apt install python3-smbus2
sudo apt install python3-serial
```

## 校准并运行demo

找到 IMU_calibration

找到源码 -> 多主控通讯 -> Raspberry5 -> UART

需要注意的是：**IMU的USB的路径要和代码里的一致**

```bash
# 把imu放在平面上静止然后校准IMU
python3 YbImu_Calibrate_IMU.py 
# 校准磁力计的时候要拿起来转动
python3 YbImu_Calibrate_Mag.py
# 最后运行demo
python3 YbImu_ReadData_Serial.py
```


## 任务

编写ROS2代码实现读取IMU数据并发送ROS2 Topic