/*
普通写例子在SCS15中测试通过，如果测试其它型号SCS系列舵机请更改合适的位置、速度与延时参数。
*/

#include <iostream>
#include "SCServo.h"

SCSCL sc;

int main(int argc, char **argv)
{
	if(argc<2){
        std::cout<<"argc error!"<<std::endl;
        return 0;
	}
	std::cout<<"serial:"<<argv[1]<<std::endl;
    if(!sc.begin(115200, argv[1])){
        std::cout<<"Failed to init scscl motor!"<<std::endl;
        return 0;
    }
	while(1){
		//舵机(ID1)以最高速度V=1500*0.059=88.5rpm，运行至P1=1000位置
		sc.WritePos(1, 1000, 0, 1500);
		std::cout<<"pos = "<<1000<<std::endl;
		usleep(((1000-20)*1000/(1500)+100)*1000);//[(P1-P0)/(V)]*1000 + 100(误差)
  
		//舵机(ID1)以最高速度V=1500*0.059=88.5rpm，运行至P0=20位置
		sc.WritePos(1, 20, 0, 1500);
		std::cout<<"pos = "<<20<<std::endl;
		usleep(((1000-20)*1000/(1500)+100)*1000);//[(P1-P0)/(V)]*1000 + 100(误差)
	}
	sc.end();
	return 1;
}

