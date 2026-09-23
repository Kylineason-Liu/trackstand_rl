ubuntu@ubuntu:~/source$ ros2 run new_balance roll_publisher
[INFO] [1784688709.494345878] [roll_info_publisher]: Opening IMU on /dev/ttyUSB1 …
YbImu Serial Opened! Baudrate=115200
----------------create receive threading--------------
[INFO] [1784688711.564991041] [roll_info_publisher]: RollInfoPublisher | port=/dev/ttyUSB1 pub=200.0 Hz | R_BS=[180.0, 180.0, 0.0]° bias=[0.0113, 0.0, 0.0] dps
Exception in thread task_serial_receive:
Traceback (most recent call last):
  File "/usr/lib/python3.12/threading.py", line 1073, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.12/threading.py", line 1010, in run
    self._target(*self._args, **self._kwargs)
  File "/home/ubuntu/assign_2_Hardwares/assign_2_Hardwares/1.IMU/Driver/YbImuLib/YbImuLib/YbImuSerialLib.py", line 212, in _data_handle
    data_array = self._dev.read_all()
                 ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/serial/serialutil.py", line 652, in read_all
    return self.read(self.in_waiting)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/serial/serialposix.py", line 595, in read
    raise SerialException(
serial.serialutil.SerialException: device reports readiness to read but returned no data (device disconnected or multiple access on port?)



ubuntu@ubuntu:~/source$ ros2 run new_balance rc_publisher
[INFO] [1784688723.252322943] [rc_publisher]: Opening SBUS receiver on /dev/ttyUSB1 …
[INFO] [1784688723.262194147] [rc_publisher]: RcPublisher | port=/dev/ttyUSB1 frame_len=35 throttle_ch=2 centre=992
Traceback (most recent call last):
  File "/home/ubuntu/source/install/new_balance/lib/new_balance/rc_publisher", line 33, in <module>
    sys.exit(load_entry_point('new-balance==0.1.0', 'console_scripts', 'rc_publisher')())
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/source/install/new_balance/lib/python3.12/site-packages/new_balance/rc_publisher.py", line 144, in main
    rclpy.spin(node)
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py", line 244, in spin
    executor.spin_once()
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 839, in spin_once
    self._spin_once_impl(timeout_sec)
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 834, in _spin_once_impl
    raise handler.exception()
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/task.py", line 269, in __call__
    self._handler.send(None)
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 518, in handler
    await call_coroutine()
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 382, in _execute
    await await_or_execute(tmr.callback)
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/executors.py", line 111, in await_or_execute
    return callback(*args)
           ^^^^^^^^^^^^^^^
  File "/home/ubuntu/source/install/new_balance/lib/python3.12/site-packages/new_balance/rc_publisher.py", line 69, in _read_data
    raw = self._ser.read(self._ser.in_waiting)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/serial/serialposix.py", line 595, in read
    raise SerialException(
serial.serialutil.SerialException: device reports readiness to read but returned no data (device disconnected or multiple access on port?)
[ros2run]: Process exited with failure 1



ubuntu@ubuntu:~/source$ ros2 run new_balance roll_publisher
[INFO] [1784688807.183713806] [roll_info_publisher]: Opening IMU on /dev/ttyUSB1 …
YbImu Serial Opened! Baudrate=115200
----------------create receive threading--------------
[INFO] [1784688809.245776682] [roll_info_publisher]: RollInfoPublisher | port=/dev/ttyUSB1 pub=200.0 Hz | R_BS=[180.0, 180.0, 0.0]° bias=[0.0113, 0.0, 0.0] dps
^CException in thread task_serial_receive:
Traceback (most recent call last):
  File "/usr/lib/python3.12/threading.py", line 1073, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.12/threading.py", line 1010, in run
    self._target(*self._args, **self._kwargs)
  File "/home/ubuntu/assign_2_Hardwares/assign_2_Hardwares/1.IMU/Driver/YbImuLib/YbImuLib/YbImuSerialLib.py", line 208, in _data_handle
    data_count = self._dev.inWaiting()
                 ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/serial/serialutil.py", line 594, in inWaiting
    return self.in_waiting
           ^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/serial/serialposix.py", line 549, in in_waiting
    s = fcntl.ioctl(self.fd, TIOCINQ, TIOCM_zero_str)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: [Errno 9] Bad file descriptor
Traceback (most recent call last):
  File "/home/ubuntu/source/install/new_balance/lib/new_balance/roll_publisher", line 33, in <module>
    sys.exit(load_entry_point('new-balance==0.1.0', 'console_scripts', 'roll_publisher')())
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/source/install/new_balance/lib/python3.12/site-packages/new_balance/roll_publisher.py", line 164, in main
    rclpy.shutdown()
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/__init__.py", line 134, in shutdown
    _shutdown(context=context)
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/utilities.py", line 82, in shutdown
    context.shutdown()
  File "/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/context.py", line 129, in shutdown
    self.__context.shutdown()
rclpy._rclpy_pybind11.RCLError: failed to shutdown: rcl_shutdown already called on the given context, at ./src/rcl/init.c:333
YbImu Serial Close!
[ros2run]: Process exited with failure 1

- 36.67426300048828
- 3.625035524368286
- 41239.51953125
- 41239.51953125
- 1924.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.46059876680374146
- 0.00019722220895346254
- -0.0
- -0.0
- 0.001319468836300075
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 41739.6328125
- 41739.6328125
- 2245.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.460690975189209
- 0.00019722220895346254
- -0.0
- -0.0
- 0.005057964473962784
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 42239.515625
- 42239.515625
- 2084.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.4610893726348877
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0011728612007573247
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 42739.51953125
- 42739.51953125
- 1961.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.46128854155540466
- 0.00019722220895346254
- -0.0
- -0.0
- 0.004471533466130495
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 43239.53125
- 43239.53125
- 1954.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.4613078236579895
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0023457224015146494
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 43739.53125
- 43739.53125
- 1872.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.4614402949810028
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0002932153001893312
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 44239.53125
- 44239.53125
- 1907.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.46139419078826904
- 0.00019722220895346254
- -0.0
- -0.0
- 0.00366519158706069
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 44739.5078125
- 44739.5078125
- 1908.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.461327463388443
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0028588492423295975
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 45239.515625
- 45239.515625
- 1967.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.46106934547424316
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0006597344181500375
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 45739.84375
- 45739.84375
- 4232.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.46121370792388916
- 0.00019722220895346254
- -0.0
- -0.0
- 0.004398229997605085
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 46239.55078125
- 46239.55078125
- 1935.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.46121254563331604
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0020525071304291487
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 46739.5234375
- 46739.5234375
- 1922.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.4612478017807007
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0009529498056508601
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 47239.51953125
- 47239.51953125
- 2108.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.46120956540107727
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 47739.54296875
- 47739.54296875
- 1879.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.4607318043708801
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0034452800173312426
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 48239.69140625
- 48239.69140625
- 2231.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.46104833483695984
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0037384952884167433
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 48739.55078125
- 48739.55078125
- 2140.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.4612598419189453
- 0.00019722220895346254
- -0.0
- -0.0
- 0.00021991149696987122
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 49239.7265625
- 49239.7265625
- 1986.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.46096688508987427
- 0.00019722220895346254
- -0.0
- -0.0
- 0.006230825092643499
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 49739.67578125
- 49739.67578125
- 1894.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.4611656963825226
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0011728612007573247
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 50239.7265625
- 50239.7265625
- 1916.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.4609068036079407
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0006597344181500375
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 50739.76953125
- 50739.76953125
- 3614.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.46077319979667664
- 0.00019722220895346254
- -0.0
- -0.0
- 0.00021991149696987122
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 51239.51171875
- 51239.51171875
- 1943.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.4607621133327484
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0006597344181500375
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 51739.51171875
- 51739.51171875
- 2103.0
---
layout:
  dim: []
  data_offset: 0
data:
- 0.46085551381111145
- 0.00019722220895346254
- -0.0
- -0.0
- 0.0011728612007573247
- -10.0
- -0.75
- 0.0
- 1.0
- 13.538069725036621
- 36.67426300048828
- 3.625035524368286
- 52239.51171875
- 52239.51171875
- 1905.0
---