"""Self-balancing bicycle control package.

Provides:
- BicycleParameters / LinearBicycleModel : 3-state speed-scheduled model
- NominalBikeController : pole-placement and LQR state-feedback
- BalanceNode : 50 Hz ROS2 node tying IMU, CAN motors, and the controllers
"""
