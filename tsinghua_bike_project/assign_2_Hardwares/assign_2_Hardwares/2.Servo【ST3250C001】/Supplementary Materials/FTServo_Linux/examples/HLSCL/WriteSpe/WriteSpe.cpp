#include <iostream>
#include "SCServo.h"

HLSCL hlscl;

int main(int argc, char **argv)
{
	if(argc<2){
        std::cout<<"argc error!"<<std::endl;
        return 0;
	}
	std::cout<<"serial:"<<argv[1]<<std::endl;
    if(!hlscl.begin(115200, argv[1])){
        std::cout<<"Failed to init sms/sts motor!"<<std::endl;
        return 0;
    }
	hlscl.WheelMode(1);//恒速模式
	std::cout<<"mode = "<<1<<std::endl;
	while(1){
		//舵机(ID1/ID2)以加速度A=50*8.7deg/s^2，加速至最高速度V=60*0.732=43.92rpm，并保持恒速正向旋转，最大扭矩电流T=500*6.5=3250mA
		hlscl.WriteSpe(1, 60, 50, 500);
		std::cout<<"speed = "<<2400<<std::endl;
		sleep(5);
		
		//舵机(ID1/ID2)以加速度A=50*8.7deg/s^2，减速至速度0停止旋转
		hlscl.WriteSpe(1, 0, 50, 500);
		std::cout<<"speed = "<<0<<std::endl;
		sleep(5);
		
		//舵机(ID1/ID2)以加速度A=50*8.7deg/s^2，加速至最高速度V=-60*0.732=-43.92rpm，并保持恒速反向旋转，最大扭矩电流T=500*6.5=3250mA
		hlscl.WriteSpe(1, -60, 50, 500);
		std::cout<<"speed = "<<-2400<<std::endl;
		sleep(2);
	
		//舵机(ID1/ID2)以加速度A=50*8.7deg/s^2，减速至速度0停止旋转
		hlscl.WriteSpe(1, 0, 50, 500);
		std::cout<<"speed = "<<0<<std::endl;
		sleep(2);
	}
	hlscl.end();
	return 1;
}

