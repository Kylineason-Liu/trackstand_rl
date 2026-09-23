1.遇到“Fail To Open Serial”

使用sudo运行，或者使用sudo usermod -a -G dialout $USER添加权限，然后重启终端再试。

2.使用前，请先安装库：cd YbImuLib && sudo python3 setup.py install

3.树莓派记得打开I2C开关

4.I2C程序可能需要更改代码里的总线，使用 i2cdetect -r -y  来查询，地址是0x23

5.运行前需要有两个环境：

sudo pip3 install pyserial
sudo pip3 install smbus2