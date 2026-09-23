# FTServo_Linux
FEETECH BUS Servo Linux library

# Static library libSCServo.a
```bash
cd src
cmake .
cmake --build .
```

# TEST

```bash
cd examples/SMS_STS/Feedback
cmake .
cmake --build .
./Feedback /dev/ttyAMA2
```