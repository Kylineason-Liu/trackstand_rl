#include <iostream>
#include "SCServo.h"

SMS_STS sms_sts;

int main(int argc, char **argv)
{
	if(argc<2){
        std::cout<< "argc error!"<<std::endl;
        return 0;
	}
	std::cout<<"serial:"<<argv[1]<<std::endl;
    if(!sms_sts.begin(115200, argv[1])){
        std::cout<<"Failed to init sms/sts motor!"<<std::endl;
        return 0;
    }

	sms_sts.CalibrationOfs(1);
	std::cout<<"Calibration Ofs"<<std::endl;
	sms_sts.end();
	return 1;
}

