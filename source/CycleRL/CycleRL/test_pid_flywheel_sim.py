import argparse
from isaaclab.app import AppLauncher

# 1. Initialize the Simulation App FIRST (Required before any Isaac Lab imports)
parser = argparse.ArgumentParser(description="Standalone PID Tester for Flywheel Bike.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Now import Isaac Lab and PyTorch modules
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# IMPORT YOUR CUSTOM HARDWARE AND PID UTILITY
# Adjust these paths depending on where you save this script!
from tasks.manager_based.cyclerl.bicycle_cfg import XUAN_BIKE_FLYWHEEL_CFG
from tasks.manager_based.cyclerl.pid_utils import TensorPID

def main():
    # 3. Create the raw Simulation Context (200 Hz)
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = SimulationContext(sim_cfg)

    # 4. Spawn the Ground
    cfg_ground = sim_utils.GroundPlaneCfg()
    sim_utils.spawn_ground_plane("/World/ground", cfg_ground)

    # 5. Spawn the Bicycle (Just 1 instance for testing)
    robot_cfg = XUAN_BIKE_FLYWHEEL_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg=robot_cfg)

    # 6. Play the simulation and initialize physics handles
    sim.reset()
    print("[INFO] Simulation started. Press 'ESC' or close the window to exit.")

    # 7. Initialize your isolated PID controller
    flywheel_pid = TensorPID(
        kp=180.0,  
        ki=0.0, 
        kd=10.0,   
        out_limit=23.0, 
        i_limit=10.0, 
        num_envs=1, # Change back to env.num_envs when you return to RL!
        device=sim.device
    )
    
    joint_idx, _ = robot.find_joints("Revolute_11")

    # 8. Reset the robot to a slightly tilted state to test if it catches itself
    root_state = robot.data.default_root_state.clone()
    # Tilt the bike slightly on the roll axis (e.g., ~3 degrees)
    # root_state[:, 3:7] = torch.tensor([0.9996, 0.0262, 0.0, 0.0], device=sim.device) 
    root_state[:, 2] += 0.0 # Lift slightly above ground
    robot.write_root_state_to_sim(root_state)
    robot.reset()

    # 9. The Raw Physics Loop
    step_count = 0
    while simulation_app.is_running():
        # A. Calculate Lean Angle (Process Variable)
        quat = robot.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        up_in_local_y = 2.0 * (y * z + w * x)
        up_in_local_z = w**2 - x**2 - y**2 + z**2
        lean = torch.atan2(up_in_local_y, up_in_local_z)
        
        # B. Calculate Lean Rate (Derivative)
        lean_rate = robot.data.root_ang_vel_b[:, 0]
        
        # C. Calculate PID Effort
        effort = flywheel_pid.step(
            setpoint=0.0, 
            measured=lean, 
            measured_rate=lean_rate, 
            dt=0.005
        )
        
        # D. Apply Torque (Inverted for reaction wheel)
        final_effort = torch.clamp(effort * -1.0, -23.0, 23.0)
        robot.set_joint_effort_target(final_effort.unsqueeze(-1), joint_ids=joint_idx)
        
        # E. Step the physics engine forward and update buffers
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)
        
        # Print diagnostic data every 100 frames (0.5 seconds)
        if step_count % 10 == 0:
            flywheel_rpm = robot.data.joint_vel[:, joint_idx].squeeze().item() * 9.55
            print(f"Tilt: {lean.item():.3f} rad | Torque: {final_effort.item():.2f} Nm | Flywheel RPM: {flywheel_rpm:.0f}")
            
        step_count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()