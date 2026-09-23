# 树莓派ubuntu系统搭建与配置

## 树莓派硬件组装

参考视频

https://www.bilibili.com/video/BV1yyK8zxESP/?spm_id_from=333.337.search-card.all.click&vd_source=8c5e18181b0dbd2120dbe353ab7a1130


## 系统写入SD卡
我们的任务是实现摩托车的平衡，需要接收来自电机和传感器的信息，并对电机发出控制指令；在这个过程中实时性非常重要，我们希望树莓派能够尽快执行我们的程序的任务，所以我们选用树莓派的**实时内核**。

另一方面，我们需要各个控制程序之间能够高效地传输信息，所以会用到ROS2

我们找到了一个已经提前装好了ROS2和实时内核的系统存档（或者说镜像），我们需要把它复制到我们的树莓派上。请按下列步骤完成复制

1. 前往raspberry官网(raspberrypi.com)并在software中下载Raspberry Pi Imager
2. 从树莓派的盒子里拿出SD卡和读卡器，将SD插到读卡器里，再将读卡器插到电脑的USB接口中
3. 打开Raspberry Pi Imager
4. 使用自定义镜像
5. 选择本文件同级的文件夹，"系统安装"里的"ubuntu-24.04-rt-ros2-arm64+raspi.img"，并点击打开，点击下一步
6. 选择储存设备为读卡器读取的SD卡对应的设备（一般默认排除了系统驱动器的话，只有一个可选项），点击下一步
7. 点击写入->我理解，擦除并写入
8. 等待完成
9. 完成后，将对应的设备弹出，然后取出SD卡并装到树莓派上
10. 此时给树莓派通电，就能正常开机了


## Putty串口登录

如果有显示器、键盘和鼠标，就能把他们连接到树莓派上，然后像正常使用电脑一样配置树莓派了。（不过咱下载的是没有可视化界面的镜像，所以其实也不需要鼠标）

但很多时候我们可能没有显示器，只有自己的电脑，我们希望使用自己的电脑来打开树莓派的终端来实现控制。这可以通过网络或者串口的方式实现，但由于树莓派还没有配置网络，所以我们要使用串口的方式打开树莓派的终端。(其实有更简单的方法，)

主要是使用Putty这个软件。

### Putty安装
在我们的windows主机上，putty可在网络上搜索下载，或是在微软的应用商店里下载安装

### 安装串口转USB驱动
主机没有串口，只有USB，所以我们需要一个串口转USB模块，但是需要先安装驱动

运行CH341SER.EXE给主机安装驱动

### Putty使用

1. 连接串口转USB模块的RX到树莓派的GPIO 14，RX到树莓派的GPIO 15，GND接到树莓派的GND(查看引脚对照图可知，例如Pin9)
2. 将串口转USB模块连接到电脑上，此时打开设备管理器查看CH340的端口号是多少
3. 鼠标右键状态栏里的Putty，鼠标左键单击New Session
4. 然后再右边的 Connection type里选择Serial
5. 更改上方的Serial line为之前步骤2看到的端口号
6. 更改Speed为115200
7. 点击软件右下角的Open，会弹出一个终端窗口
8. 此时如果重启树莓派(拔电再上电)，能在窗口里看到树莓派的输出
9. 如果不重启树莓派，可以在窗口里按enter键(没反应的话就还是断电重新上电吧)


## ubuntu登录

最终应该能看见如下的输出
```bash
ubuntu login:
```
这是提醒你也要登录，就像windows一样要登录，先输入用户名 ubuntu ，按下回车

初始密码是ubuntu，输入密码 ubuntu，按下回车

此时就完成了登录，根据操作完成密码的重置就行

> 这里的用户是镜像里提前创建好的，如果是自己下载了一个全新的image的话，可以自己定义用户名和密码

## 网络配置

第一次登录进去之后，我们需要先连接网络

```bash
# 打开网络的配置文件
sudo nano /etc/netplan/50-cloud-init.yaml
```
修改成大概如下的内容，注意冒号后面有空格！！并且缩进也要严格对应
```
network:
  version: 2
  ethernets:
    eth0:
      optional: true
      dhcp4: true
  wifis:
    wlan0:          # 替换成你实际的无线网卡名
      dhcp4: true
      optional: true
      access-points:
        "你的WiFi名称":    # 替换成你的 WiFi SSID
          password: "你的WiFi密码"   # 替换成你的 WiFi 密码
```

先按下 Ctrl + O保存，按下enter键确定保存，再按下Ctrl + X退出

接下来就可以让ubuntu实际应用我们的配置:
```bash
sudo netplan apply
```

可能会弹出如下的一堆东西，如果只弹出一次，可以忽略，直接按下enter键开始新的一行
```bash
[  475.028537] brcmfmac: brcmf_set_channel: set chanspec 0x100c fail, reason -52
[  475.028771] brcmfmac: brcmf_set_channel: set chanspec 0x100d fail, reason -52
[  475.028996] brcmfmac: brcmf_set_channel: set chanspec 0x100e fail, reason -52
[  475.029220] brcmfmac: brcmf_set_channel: set chanspec 0xd022 fail, reason -52
[  475.132579] brcmfmac: brcmf_set_channel: set chanspec 0xd026 fail, reason -52
[  475.236566] brcmfmac: brcmf_set_channel: set chanspec 0xd02a fail, reason -52
[  475.340574] brcmfmac: brcmf_set_channel: set chanspec 0xd02e fail, reason -52
```

我们测试一下是否真的有网络：
```bash
ping baidu.com
```

如果输出类似下面的样子就是有网了，按下Ctrl+C终止命令
```bash
ubuntu@ubuntu:~$ ping baidu.com
PING baidu.com (111.63.65.103) 56(84) bytes of data.
64 bytes from 111.63.65.103: icmp_seq=1 ttl=46 time=35.9 ms
64 bytes from 111.63.65.103: icmp_seq=2 ttl=46 time=36.2 ms
```

## 国内镜像源配置

现在网络已经配置好了，但是ubuntu的官方软件下载源下载得很慢，所以需要更改下载源为更快的清华镜像源

```bash
#打开ubuntu负责管理软件下载源的配置文件
sudo nano /etc/apt/sources.list.d/ubuntu.sources
```

手动修改第一个URIs为: https://mirros.tuna.tsinghua.edu.cn/ubuntu-ports

```bash
#URIs: http://ports.ubuntu.com/ubuntu-ports/       这个改成下面的
URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports
```

总之，最后的配置文件看起来像这样：
```bash

## Note, this file is written by cloud-init on first boot of an instance
## modifications made here will not survive a re-bundle.
##
## If you wish to make changes you can:
## a.) add 'apt_preserve_sources_list: true' to /etc/cloud/cloud.cfg
##     or do the same in user-data
## b.) add supplemental sources in /etc/apt/sources.list.d
## c.) make changes to template file
##      /etc/cloud/templates/sources.list.ubuntu.deb822.tmpl
##

# See http://help.ubuntu.com/community/UpgradeNotes for how to upgrade to
# newer versions of the distribution.

## Ubuntu distribution repository
##
## The following settings can be adjusted to configure which packages to use fr>
## Mirror your choices (except for URIs and Suites) in the security section bel>
## ensure timely security updates.
##
## Types: Append deb-src to enable the fetching of source package.
## URIs: A URL to the repository (you may add multiple URLs)
## Suites: The following additional suites can be configured
##   <name>-updates   - Major bug fix updates produced after the final release >
##                      distribution.
##   <name>-backports - software from this repository may not have been tested >
##                      extensively as that contained in the main release, alth>
##                      newer versions of some applications which may provide u>
##                      Also, please note that software in backports WILL NOT r>
##                      or updates from the Ubuntu security team.
## Components: Aside from main, the following components can be added to the li>
##   restricted  - Software that may not be under a free license, or protected >
##   universe    - Community maintained packages. Software in this repository r>
##                 from volunteers in the Ubuntu community, or a 10 year securi>
##                 commitment from Canonical when an Ubuntu Pro subscription is>
##   multiverse  - Community maintained of restricted. Software from this repos>
##                 ENTIRELY UNSUPPORTED by the Ubuntu team, and may not be unde>
##                 licence. Please satisfy yourself as to your rights to use th>
##                 Also, please note that software in multiverse WILL NOT recei>
##                 review or updates from the Ubuntu security team.
##
## See the sources.list(5) manual page for further settings.

Types: deb
URIs: https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports # 只改了这一行
Suites: noble noble-updates noble-backports
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

## Ubuntu security updates. Aside from URIs and Suites,
## this should mirror your choices in the previous section.
Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports/
Suites: noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
```

同样的，按下Ctrl+O申请保存，按Enter键确认保存,按Ctrl+X退出编辑

最后，让apt从配置文件里再次读取、更新下载源
```bash
sudo apt update
```

## ssh配置

我们之前都是使用串口连接树莓派的终端，但是这总是需要连接线，很不方便。而在我们配置好了网络之后，我们就可以通过网络来连接树莓派的终端。也就是通过ssh登录Ubuntu。

ubuntu24.04默认情况是不允许使用密码登录的，这在开发期间会带来一定的麻烦，我们更改设置为允许使用密码登录

```bash
# 打开ssh的配置文件，并把文件中的 PasswordAuthentication后的 no 改成 yes
sudo nano /etc/ssh/sshd_config.d/50-cloud-init.conf
# 接下来重启ssh服务
sudo systemctl restart ssh
```

接下来我们就可以使用自己的电脑，通过ssh连接树莓派上的ubuntu24.04的系统了，也就可以使用vscode进行方便快捷的开发了
```bash
# 树莓派上运行此命令查看自己的ip，其中wlan0是无线网卡，其下对应的ip地址就是树莓派的ip地址
ip addr
```

一般来说，在国内的局域网，通过wifi里获取到的ip不会是公网ip，所以需要电脑和树莓派连上同一个网络。

```bash
ssh 用户名@网络地址
```

## 个人电脑安装VSCode并配置Remote Explorer

可详细参考https://www.cnblogs.com/emanlee/p/18807002


## 准备C++相关配置

### 安装g++和其他构建的工具
```bash
sudo apt update 
sudo apt install g++ build-essential
```


## 安装AI工具
参考 AI Config