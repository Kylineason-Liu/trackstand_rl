# 舵机的配置

我们的舵机可以连接驱动板；驱动板可以通过UART连接树莓派。树莓派通过UART与驱动板进行通信，就能实现对舵机的控制。

接下来我们开始调试。

## 硬件调试

### 驱动下载
windows运行 CH343SER.EXE安装ch343的驱动程序

(如果是Mac的话，已经安装了ch341就不用再装了)

### 硬件连接

1. URT-2调试板连接一个舵机，先只连接一个舵机
2. 拿出24V-12V降压器，输入为电池接的分电板，输出到URT-2调试板上
3. 使用typeC的USB适配器连接调试板和windows电脑(如果没有windows可以找助教)

### 上位机调试
1. 在windows上打开HostApplication文件夹里的FD.exe
2. 在设备管理器里找到对应CH343的端口号
3. 在FD.exe中,选择"端口号"为对应CH343的端口号
4. 修改波特率为 1000000
5. 点击下方的"打开"，以及更下方的"搜索"
6. 观察到左下角搜索到舵机之后，点击"关闭"
7. 双击舵机
8. 在中上部找到"编程"选项卡，并点击
9. 找到地址'5'，点击这一行
10. 在右下角找到用于修改的文本框，将ID改为2并点击保存

这下之后，两个舵机的ID就不一样了，完成了调试任务。其他的参数不建议修改，但可以玩玩舵机操作。

调试完成后，拔掉USB口


## 树莓派配置

### UART配置
我们先看看树莓派5里有哪些串口设备
```bash
ls /dev/ttyAMA*
```

应该会显示两个
```bash
/dev/ttyAMA0  /dev/ttyAMA10
```
其中AMA0对应的是GPIO14和GPIO15两个口，AMA10对应的是在两个HDMI插口中间的那个3pin的接口，他们一般负责debug或者是登录终端，一般不进行串口的通信，所以我们需要配置其他的GPIO来进行串口通信。

---

打开树莓派5的硬件配置文件

```bash
sudo nano /boot/firmware/config.txt
```
在 "enable_uart=1"的下一行增加:
```bash
dtoverlay=uart2-pi5
```
然后按"Ctrl+O"保存，"Ctrl+X"退出

以此来启用GPIO4（PIN7）和GPIO5（PIN29）作为UART2的TX和RX口

最后，需要重启树莓派
```bash
sudo reboot
```

### UART配置测试
重启之后，看看串口设备的变化
```bash
ls /dev/ttyAMA*
```

需要让用户能够访问硬件
```bash
#把当前用户加入 dialout 组：
sudo usermod -aG dialout $USER
#然后执行更新
newgrp dialout
```

然后，使用杜邦线，短接GPIO4（PIN7）和GPIO5（PIN29）。随后我们将向串口发送消息，由于TX和RX短接，如果一切正常，我们就会收到自己发送的消息。

恰好python的serial库有这个功能

```bash
#先安装serial库
sudo apt update
sudo apt install python3-serial
#然后调用serial库
sudo python3 -m serial.tools.miniterm /dev/ttyAMA2 115200
#打开了之后随便按键盘上，如果屏幕上会出现对应的字，则说明UART功能正常；如果看不到任何字符出现，说明UART功能异常。
```

## 例程运行

参考树莓派引脚图，将树莓派的GPIO4和GPIO5分别连接到URT-2调试板的TX和RX引脚，树莓派的3.3V和GND接到URT-2调试板的VCC和GND引脚。

根据Driver Lib文件夹中的README.md进行操作即可。如果不理解CMakeLists.txt是什么，可以在VSCode里询问Claude Code，顺便学习一下CMake

## 任务

编写ROS2控制程序,发送舵机状态的topic，并提供从ROS2 Topic控制舵机的接口，