# 电机调试与控制

电机由我们的6S电池供电，电机可以通过两条线(CAN线)连接到分电板的CAN端口上(就好像进入了群聊)，树莓派也通过USB-CAN转换器接到分电板上后(就好像进入了群聊之后)，就能通过CAN往分电板上的CAN总线发送消息(往群聊里发送消息)，从而控制电机。同样的，电机也能往CAN总线发送消息进行反馈。

不过在一切之前我们先需要配置树莓派使用USB2CAN模块进行CAN通信

## 准备CAN通信相关的配置

## USB2CAN模块(Markerbase)固件更新

我们购买的USB2CAN模块的默认固件需要升级

1. 打开你的windows电脑并下载并安装chorme
2. 使用chorme进入固件更新网站https://canable.io/updater/canable2.html
3. 拆开USB2CAN模块，使用杜邦线连接boot的两个引脚，并把模块插入到windows电脑上(没有windows电脑可以找助教)
4. 在网站上选择candelight
5. 点击"Connect and Update"

## 电机连接

拿出电机盒里的4pin的线，把"ABHL"中HL对应的两根线连接的接头，(称为CAN头)，接到分电板上的CAN接口，就是那个有很多插头的板子。

同样的，找出在assign_1中升级过驱动的那个USB2CAN模块(Markerbase)。将它插到树莓派上，它的CANH和CANL两端，通过一根CAN线接到分电板的另一个CAN接口上。CAN线的红线对应CANH，黑线对应CANL。


## 使能树莓派的CAN接口

通过ip link命令，我们能看到当前树莓派连接的网络设备情况，可以发现can0的state是DOWN,也就是没有使能。
```bash
ubuntu@ubuntu:~$ ip link
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
2: eth0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN mode DEFAULT group default qlen 1000
    link/ether 88:a2:9e:5c:1c:e2 brd ff:ff:ff:ff:ff:ff
3: can0: <NOARP,ECHO> mtu 16 qdisc noop state DOWN mode DEFAULT group default qlen 10
    link/can 
4: wlan0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP mode DORMANT group default qlen 1000
    link/ether 88:a2:9e:5c:1c:e3 brd ff:ff:ff:ff:ff:ff
```

我们需要使能它，并设置树莓派的CAN进行通讯的波特率。
```bash
# 这个命令的意思是：使能can0，它的类型是can，比特率是1000000
# 这里的比特率是需要和电机匹配的，就像电台的调频一样。电机的比特率在CAN通信协议的文档里有写
sudo ip link set can0 up type can bitrate 1000000
```

现在再看，就已经完成使能操作了
```bash
ubuntu@ubuntu:~$ ip link
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
2: eth0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc pfifo_fast state DOWN mode DEFAULT group default qlen 1000
    link/ether 88:a2:9e:5c:1c:e2 brd ff:ff:ff:ff:ff:ff
3: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP mode DEFAULT group default qlen 10
    link/can 
4: wlan0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc pfifo_fast state UP mode DORMANT group default qlen 1000
    link/ether 88:a2:9e:5c:1c:e3 brd ff:ff:ff:ff:ff:ff
```

## 电机配置
在CAN总线上通讯需要使用CAN地址，也就是CAN ID，如果我们想要向电机发送指令，需要以它对应的CAN ID为帧头，向CAN总线发送CAN消息；一般来说，电机在收到了指令后，会以同样的CAN ID在总线上发送CAN消息。

我们的电机，默认ID是001，如果两个电机直接同时接入CAN总线的话，我们通过ID 001不能区分两个电机，所以我们需要修改其中一个电机的ID。例如修改转向电机的ID为2。转向电机的型号是GIM4305，对应形状上矮一点的那个电机。

我们可以使用CAN通讯来修改转向电机的ID

在CAN Protocol Doc文件夹中的《自定义CAN通信协议》的第3页，可以找到

![CAN COMMANDS](images_for_readme/CAN_Command.png)

其中在"参数"栏的最后可以看到"读取或设置设备地址"，所以我们可以通过CAN来修改电机的ID。

具体的CAN帧应该怎么发，同样在CAN Protocol Doc文件夹中的《自定义CAN通信协议》里的第12页和第13页可以找到

![CAN_ID para](images_for_readme/CAN_ID_para.png)


### 树莓派安装CAN通讯的命令行工具
安装can-utils
```bash
sudo apt install can-utils
```

candump 用来观察某个can总线上的消息
```bash
candump can0
```

cansend 用来在某个can总线上以某个CANID发送消息
```
cansend can0 [CAN_ID]#[CAN_MESSAGE]
```

### 通过CAN通讯命令行工具配置电机


## 根据通信协议编写驱动

这部分已经有驱动了，不过可以使用Claude Code让它读取通信协议的文件，根据CAN通信的协议重新写一份驱动程序。

## 任务

编写ROS2代码，提供从ROS2 topic控制电机的接口，并发送电机的状态的 ROS2 topic。