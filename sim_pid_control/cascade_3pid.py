from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics, Sdf, Usd
import omni.ui as ui

import sys
import os
import carb
import carb.input
import omni.appwindow

# ---------------------------------------------------------
# T-MOTOR P80 III CONFIGURATION
# ---------------------------------------------------------
MAX_FLYWHEEL_TORQUE = 20.0
MAX_STEERING_TORQUE = 2.0
MAX_DRIVE_TORQUE = 32.5

MAX_FLYWHEEL_VEL = 200

# ======================================================================
# KEYBOARD CONTROL MAPPING
# ======================================================================
def on_keyboard_event(event, *args, **kwargs):
    is_pressed = (
        event.type == carb.input.KeyboardEventType.KEY_PRESS or 
        event.type == carb.input.KeyboardEventType.KEY_REPEAT
    )
    current_module = sys.modules[__name__]
    
    if event.input == carb.input.KeyboardInput.UP:
        setattr(current_module, 'ui_backward', is_pressed)
    elif event.input == carb.input.KeyboardInput.DOWN:
        setattr(current_module, 'ui_forward', is_pressed)
    elif event.input == carb.input.KeyboardInput.LEFT:
        setattr(current_module, 'ui_left', is_pressed)
    elif event.input == carb.input.KeyboardInput.RIGHT:
        setattr(current_module, 'ui_right', is_pressed)
    elif event.input == carb.input.KeyboardInput.SPACE:
        if is_pressed:
            setattr(current_module, 'ui_stop', True)
            
    return True

# ======================================================================
# PID CONTROLLER
# ======================================================================
class PID:
    def __init__(self, kp, ki, kd, out_limit=None, i_limit=5.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit = out_limit
        self.i_limit = i_limit
        self.integral = 0.0
        self.prev_error = 0.0
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        
    def step(self, setpoint, measured, measured_rate, dt):
        error = setpoint - measured
        self.integral += error * dt
        # NOTE: i_limit clips the raw error sum, not the final output!
        self.integral = float(np.clip(self.integral, -self.i_limit, self.i_limit))
        out = self.kp * error + self.ki * self.integral - self.kd * measured_rate
        if self.out_limit is not None:
            out = float(np.clip(out, -self.out_limit, self.out_limit))
        return out

# ======================================================================
# SIMULATION SETUP
# ======================================================================
PHYSICS_DT = 1.0 / 200.0
RENDER_DT  = 1.0 / 50.0  # MATCH ISAAC LAB (50 Hz)
CONTROL_DT = RENDER_DT

world = World(physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
USD_PATH = os.path.join(CURRENT_DIR, "..", "source", "CycleRL", "CycleRL", "tasks", "manager_based", "cyclerl", "assets", "xuan_bike_2kg5", "xuan_bike_2kg5.usda")
PRIM_PATH = "/World/Bike"
add_reference_to_stage(usd_path=USD_PATH, prim_path=PRIM_PATH)

bike = world.scene.add(
    Robot(prim_path=PRIM_PATH, name="balance_bike", position=np.array([0.0, 0.0, 0.0]))
)
world.reset()

# ======================================================================
# FRICTION SETUP
# ======================================================================
def setup_ground_friction():
    stage = omni.usd.get_context().get_stage()
    ground_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
    if ground_prim.IsValid():
        if not ground_prim.HasAPI(UsdPhysics.MaterialAPI):
            UsdPhysics.MaterialAPI.Apply(ground_prim)
        try:
            ground_prim.CreateAttribute("physxMaterial:staticFriction", Sdf.ValueTypeNames.Float).Set(0.8)
            ground_prim.CreateAttribute("physxMaterial:dynamicFriction", Sdf.ValueTypeNames.Float).Set(0.7)
            ground_prim.CreateAttribute("physxMaterial:restitution", Sdf.ValueTypeNames.Float).Set(0.1)
        except:
            pass

def setup_tire_friction():
    stage = omni.usd.get_context().get_stage()
    bike_prim = stage.GetPrimAtPath(PRIM_PATH)
    if not bike_prim.IsValid(): return
    for prim in Usd.PrimRange(bike_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            try:
                if not prim.HasAPI(UsdPhysics.MaterialAPI):
                    UsdPhysics.MaterialAPI.Apply(prim)
                prim.CreateAttribute("physxMaterial:staticFriction", Sdf.ValueTypeNames.Float).Set(1.0)
                prim.CreateAttribute("physxMaterial:dynamicFriction", Sdf.ValueTypeNames.Float).Set(0.9)
                prim.CreateAttribute("physxMaterial:restitution", Sdf.ValueTypeNames.Float).Set(0.05)
            except:
                pass

def setup_physics_scene():
    stage = omni.usd.get_context().get_stage()
    physics_scene = stage.GetPrimAtPath("/World/physicsScene")
    if physics_scene.IsValid():
        try:
            physics_scene.CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("TGS")
            physics_scene.CreateAttribute("physxScene:frictionType", Sdf.ValueTypeNames.Token).Set("patch")
            physics_scene.CreateAttribute("physxScene:enableEnhancedDeterminism", Sdf.ValueTypeNames.Bool).Set(True)
        except:
            pass

setup_ground_friction()
setup_tire_friction()
setup_physics_scene()

# ======================================================================
# JOINT MAPPING & FRICTION PARAMS
# ======================================================================
FLYWHEEL_JOINT = "Revolute_11"
STEERING_JOINT = "Revolute_7"
REAR_WHEEL_JOINT = "Revolute_9"

wheel_idx = bike.get_dof_index(FLYWHEEL_JOINT)
steering_idx = bike.get_dof_index(STEERING_JOINT)
rear_wheel_idx = bike.get_dof_index(REAR_WHEEL_JOINT)

controller = bike.get_articulation_controller()
view = bike._articulation_view

JOINT_FRICTION = {'flywheel': 0.01, 'steering': 0.15, 'rear_wheel': 0.05}
COULOMB_FRICTION = {'flywheel': 0.001, 'steering': 0.02, 'rear_wheel': 0.02}

# ======================================================================
# CASCADE PID CONTROLLER PARAMETERS (FINAL BOSS TUNE)
# ======================================================================
MAX_LEAN = np.deg2rad(45)
CTRL_SIGN = -1.0 
LEAN_NUM, LEAN_DEN = 1, 2

# 1. Outer Loop: Velocity Controller (Outputs Target Lean Angle in Radians)
pid_vel = PID(kp=0.00003, ki=0.00001, kd=0.0, out_limit=0.1, i_limit=2000.0)

# 2. Middle Loop: Angle Controller (Outputs Target Gyro/Lean Rate in Rad/s)
pid_angle = PID(kp=15.0, ki=0.0, kd=1.0, out_limit=5.0, i_limit=1.0)

# 3. Inner Loop: Gyro/Rate Controller (Outputs Motor Torque in Nm)
pid_gyro = PID(kp=30.0, ki=0.0, kd=2.0, out_limit=MAX_FLYWHEEL_TORQUE, i_limit=10.0)

# Peripheral PIDs
pid_steering = PID(kp=2.0, ki=0.0, kd=1.0, out_limit=MAX_STEERING_TORQUE, i_limit=0.5)
MAX_STEER_ANGLE = np.deg2rad(90)
pid_speed = PID(kp=20.0, ki=1.0, kd=0.0, out_limit=MAX_DRIVE_TORQUE, i_limit=3.0)
WHEEL_RADIUS = 0.35
MAX_SPEED = 3.0

# --- TRACKING VARIABLES ---
initial_flywheel_vel = None
max_flywheel_vel = 0.0
max_flywheel_torque = 0.0
target_lean_angle = 0.0  

# --- CONTROL STATE ---
target_speed = 0.0
target_direction = 0.0
ui_forward = False
ui_backward = False
ui_left = False
ui_right = False
ui_stop = False

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================
def physics_view_ready():
    return getattr(view, "_physics_view", None) is not None

def apply_joint_friction():
    try:
        velocities = view.get_joint_velocities()
        if velocities is None: return np.zeros(bike.num_dof)
        v = np.asarray(velocities).reshape(-1)
        n = bike.num_dof
        damping = np.zeros(n)
        
        if wheel_idx < n:
            damping[wheel_idx] = -JOINT_FRICTION['flywheel'] * v[wheel_idx]
            if abs(v[wheel_idx]) > 0.01: damping[wheel_idx] -= np.sign(v[wheel_idx]) * COULOMB_FRICTION['flywheel']
        if steering_idx < n:
            damping[steering_idx] = -JOINT_FRICTION['steering'] * v[steering_idx]
            if abs(v[steering_idx]) > 0.01: damping[steering_idx] -= np.sign(v[steering_idx]) * COULOMB_FRICTION['steering']
        if rear_wheel_idx < n:
            damping[rear_wheel_idx] = -JOINT_FRICTION['rear_wheel'] * v[rear_wheel_idx]
            if abs(v[rear_wheel_idx]) > 0.01: damping[rear_wheel_idx] -= np.sign(v[rear_wheel_idx]) * COULOMB_FRICTION['rear_wheel']
        return damping
    except:
        return np.zeros(bike.num_dof)

def arm_torque_mode():
    if not physics_view_ready(): return False
    
    n = bike.num_dof
    me = np.asarray(view.get_max_efforts()).reshape(1, -1).astype(np.float32)
    
    me[0, wheel_idx] = MAX_FLYWHEEL_TORQUE       
    me[0, rear_wheel_idx] = MAX_DRIVE_TORQUE
    me[0, steering_idx] = MAX_STEERING_TORQUE    
    
    view.set_max_efforts(me)
    controller.set_gains(kps=np.zeros(n, dtype=np.float32), kds=np.zeros(n, dtype=np.float32))
    
    mv = np.full((1, n), 200, dtype=np.float32)
    mv[0, wheel_idx] = MAX_FLYWHEEL_VEL
    mv[0, rear_wheel_idx] = 93.0
    
    try:
        view.set_max_joint_velocities(mv)
        print(f"[ARM] Cascade Hardware Mode Armed. Flywheel Effort: {me[0, wheel_idx]} Nm")
    except Exception as e:
        print("[ARM] Velocity limit setting failed:", e)
    
    return True

def joint_vel(idx):
    raw = view.get_joint_velocities()
    if raw is None: return 0.0
    v = np.asarray(raw).reshape(-1)
    return float(v[idx]) if v.size > idx else 0.0

def joint_pos(idx):
    raw = view.get_joint_positions()
    if raw is None: return 0.0
    v = np.asarray(raw).reshape(-1)
    return float(v[idx]) if v.size > idx else 0.0

def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z),   2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])

def body_up():
    _, quat = bike.get_world_pose()
    return quat_to_rot(quat).T @ np.array([0.0, 0.0, 1.0])

_prev_lean = None
def get_lean_and_rate(dt):
    global _prev_lean
    ub = body_up()
    lean = np.arctan2(ub[LEAN_NUM], ub[LEAN_DEN])
    if _prev_lean is None:
        rate = 0.0
    else:
        d = (lean - _prev_lean + np.pi) % (2*np.pi) - np.pi
        rate = d / dt
    _prev_lean = lean
    return lean, float(rate)

def get_bike_velocity():
    lin_vel = bike.get_linear_velocity()
    if lin_vel is None: return 0.0, 0.0
    _, quat = bike.get_world_pose()
    rot = quat_to_rot(quat)
    local_vel = rot @ lin_vel
    return float(local_vel[0]), float(local_vel[1])

def get_bike_yaw_rate():
    ang_vel = bike.get_angular_velocity()
    if ang_vel is None: return 0.0
    return float(ang_vel[2])

def process_ui_input(dt):
    global target_speed, target_direction
    if ui_forward and not ui_backward:
        target_speed = min(target_speed + 0.1, MAX_SPEED)
    elif ui_backward and not ui_forward:
        target_speed = max(target_speed - 0.2, -1.0)
    else:
        target_speed *= 0.98
        if abs(target_speed) < 0.01: target_speed = 0.0
    
    if ui_stop:
        target_speed = 0.0
        target_direction = 0.0
    
    if ui_left and not ui_right:
        target_direction = min(target_direction + 0.01, 1.0)
    elif ui_right and not ui_left:
        target_direction = max(target_direction - 0.01, -1.0)
    else:
        target_direction *= 0.9
        if abs(target_direction) < 0.001: target_direction = 0.0
    
    target_direction = np.clip(target_direction, -1.0, 1.0)

# ======================================================================
# CREATE UI
# ======================================================================
def create_ui():
    window = ui.Window("Bike Controller", width=350, height=650)
    
    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label("🚲 Bike Controls", height=30, alignment=ui.Alignment.CENTER)
            ui.Label("(Cascade PID Mode)", height=15, alignment=ui.Alignment.CENTER)
            ui.Spacer(height=5)
            
            status_label = ui.Label("Status: Ready", height=20)
            lean_label = ui.Label("Lean: 0.0°", height=20)
            
            ui.Spacer(height=10)
            
            with ui.HStack():
                with ui.VStack():
                    ui.Spacer(height=60)
                    ui.Button("◄\nLEFT", width=100, height=60,
                             mouse_pressed_fn=lambda x, y, b, m: setattr(sys.modules[__name__], 'ui_left', True),
                             mouse_released_fn=lambda x, y, b, m: setattr(sys.modules[__name__], 'ui_left', False))
                
                with ui.VStack():
                    ui.Button("▲\nFORWARD", width=100, height=60,
                             mouse_pressed_fn=lambda x, y, b, m: setattr(sys.modules[__name__], 'ui_forward', True),
                             mouse_released_fn=lambda x, y, b, m: setattr(sys.modules[__name__], 'ui_forward', False))
                    
                    ui.Button("■ STOP", width=100, height=40,
                             clicked_fn=lambda: setattr(sys.modules[__name__], 'ui_stop', True))
                    
                    ui.Button("▼\nBACK", width=100, height=60,
                             mouse_pressed_fn=lambda x, y, b, m: setattr(sys.modules[__name__], 'ui_backward', True),
                             mouse_released_fn=lambda x, y, b, m: setattr(sys.modules[__name__], 'ui_backward', False))
                
                with ui.VStack():
                    ui.Spacer(height=60)
                    ui.Button("►\nRIGHT", width=100, height=60,
                             mouse_pressed_fn=lambda x, y, b, m: setattr(sys.modules[__name__], 'ui_right', True),
                             mouse_released_fn=lambda x, y, b, m: setattr(sys.modules[__name__], 'ui_right', False))
            
            ui.Spacer(height=10)
            
            with ui.VStack():
                speed_display = ui.Label("Target Speed: 0.0 m/s", height=20)
                curr_vel_display = ui.Label("Curr Vel: 0.00 rad/s", height=20)
                curr_torque_display = ui.Label("Curr Torque: 0.00 Nm", height=20)
                forward_speed_display = ui.Label("Forward Speed: 0.0 m/s", height=20)
                lateral_speed_display = ui.Label("Lateral Slip: 0.00 m/s", height=20)
                steer_display = ui.Label("Steering: 0.0°", height=20)
                
            ui.Spacer(height=5)
            ui.Line(style={"color": 0xFF555555}, height=2)
            ui.Spacer(height=5)
            
            with ui.VStack():
                ui.Label("--- Flywheel Metrics ---", height=20)
                init_vel_display = ui.Label("Init Vel: 0.00 rad/s", height=20)
                max_vel_display = ui.Label("Max Vel: 0.00 rad/s", height=20)
                max_torque_display = ui.Label("Max Torque: 0.00 Nm", height=20)
    
    return window, status_label, lean_label, speed_display, forward_speed_display, lateral_speed_display, steer_display, curr_vel_display, curr_torque_display, init_vel_display, max_vel_display, max_torque_display

ui_window, ui_status, ui_lean, ui_speed, ui_fwd_speed, ui_lat_speed, ui_steer, ui_curr_vel, ui_curr_torque, ui_init_vel, ui_max_vel, ui_max_torque = create_ui()

appwindow = omni.appwindow.get_default_app_window()
if appwindow:
    keyboard = appwindow.get_keyboard()
    input_ext = carb.input.acquire_input_interface()
    keyboard_sub_id = input_ext.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

# ======================================================================
# MAIN LOOP
# ======================================================================
pid_vel.reset()
pid_angle.reset()
pid_gyro.reset()
pid_steering.reset()
pid_speed.reset()

_prev_lean = None
_dbg = 0
_armed = False

while simulation_app.is_running():
    world.step(render=True)

    if not world.is_playing():
        _armed = False
        pid_vel.reset()
        pid_angle.reset()
        pid_gyro.reset()
        pid_steering.reset()
        pid_speed.reset()
        _prev_lean = None
        initial_flywheel_vel = None
        max_flywheel_vel = 0.0
        max_flywheel_torque = 0.0
        continue

    if not _armed:
        try:
            view.initialize()
        except Exception:
            pass
        if arm_torque_mode():
            _armed = True
            pid_vel.reset()
            pid_angle.reset()
            pid_gyro.reset()
            pid_steering.reset()
            pid_speed.reset()
            _prev_lean = None
            initial_flywheel_vel = joint_vel(wheel_idx)
        else:
            continue

    process_ui_input(CONTROL_DT)
    if ui_stop: ui_stop = False
    
    lean, lean_rate = get_lean_and_rate(CONTROL_DT)
    forward_speed, lateral_speed = get_bike_velocity()
    yaw_rate = get_bike_yaw_rate()
    
    friction_forces = apply_joint_friction()
    
    # ------------------------------------------------------------------
    # CASCADE BALANCE CONTROL
    # ------------------------------------------------------------------
    if abs(lean) > MAX_LEAN:
        pid_vel.reset()
        pid_angle.reset()
        pid_gyro.reset()
        flywheel_cmd = 0.0
        target_lean_angle = 0.0
    else:
        wheel_speed = joint_vel(wheel_idx)
        
        # Track Max Velocity
        if abs(wheel_speed) > max_flywheel_vel:
            max_flywheel_vel = abs(wheel_speed)
            
        # LOOP 1: Outer Velocity Controller
        target_lean_angle = pid_vel.step(setpoint=0.0, measured=wheel_speed, measured_rate=0.0, dt=CONTROL_DT)
        
        # LOOP 2: Middle Angle Controller
        target_lean_rate = pid_angle.step(setpoint=target_lean_angle, measured=lean, measured_rate=0.0, dt=CONTROL_DT)
        
        # LOOP 3: Inner Gyro Controller
        pid_out = pid_gyro.step(setpoint=target_lean_rate, measured=lean_rate, measured_rate=0.0, dt=CONTROL_DT)
        
        flywheel_cmd = CTRL_SIGN * pid_out 
    
    # [MATCH ISAAC LAB] Removed manual friction compensation for the flywheel
    # if wheel_idx < len(friction_forces):
    #     flywheel_cmd += friction_forces[wheel_idx]

    current_flywheel_torque = flywheel_cmd
        
    if abs(flywheel_cmd) > max_flywheel_torque:
        max_flywheel_torque = abs(flywheel_cmd)
    
    # ------------------------------------------------------------------
    # STEERING CONTROL
    # ------------------------------------------------------------------
    if abs(forward_speed) > 0.0:
        desired_steer = target_direction * MAX_STEER_ANGLE
        speed_factor = max(0.2, 1.0 - abs(forward_speed) / MAX_SPEED)
        desired_steer *= speed_factor
    else:
        desired_steer = 0.0
    
    current_steer = joint_pos(steering_idx)
    steer_rate = joint_vel(steering_idx)
    steering_cmd = pid_steering.step(desired_steer, current_steer, steer_rate, CONTROL_DT)
    
    if steering_idx is not None and steering_idx < len(friction_forces):
        steering_cmd += friction_forces[steering_idx]
        if abs(target_direction) < 0.01:
            steering_cmd -= steer_rate * 0.5
            
    # ------------------------------------------------------------------
    # SPEED CONTROL
    # ------------------------------------------------------------------
    if abs(target_speed) > 0.01:
        speed_factor = 0.3 if abs(lean) > np.deg2rad(15) else 1.0
        target_wheel_vel = (target_speed / WHEEL_RADIUS) * speed_factor
        current_wheel_vel = joint_vel(rear_wheel_idx)
        speed_cmd = pid_speed.step(target_wheel_vel, current_wheel_vel, 0.0, CONTROL_DT)
    else:
        current_wheel_vel = joint_vel(rear_wheel_idx)
        speed_cmd = -current_wheel_vel * 0.5
        pid_speed.reset()
    
    if rear_wheel_idx is not None and rear_wheel_idx < len(friction_forces):
        speed_cmd += friction_forces[rear_wheel_idx]
    
    controller.apply_action(ArticulationAction(
        joint_efforts=np.array([flywheel_cmd, steering_cmd, speed_cmd], dtype=np.float32),
        joint_indices=np.array([wheel_idx, steering_idx, rear_wheel_idx]),
    ))

    # ------------------------------------------------------------------
    # UI UPDATE LOOP
    # ------------------------------------------------------------------
    _dbg += 1
    
    current_flywheel_vel = joint_vel(wheel_idx) 
    
    if _dbg % 15 == 0:
        if abs(lean) < np.deg2rad(5):
            ui_status.text = "Status: ✅ Stable"
        elif abs(lean) < np.deg2rad(15):
            ui_status.text = "Status: ⚠️ Leaning"
        else:
            ui_status.text = "Status: 🔴 Critical!"
        
        ui_lean.text = f"Lean: {np.rad2deg(lean):.1f}°"
        ui_speed.text = f"Target: {target_speed:.2f} m/s"
        ui_fwd_speed.text = f"Forward: {forward_speed:.2f} m/s"
        ui_lat_speed.text = f"Slip: {lateral_speed:.3f} m/s"
        ui_steer.text = f"Steer: {np.rad2deg(current_steer):.1f}°"
        
        ui_curr_vel.text = f"Curr Vel: {current_flywheel_vel:.2f} rad/s"
        ui_curr_torque.text = f"Curr Torque: {current_flywheel_torque:.2f} Nm"
        
        if initial_flywheel_vel is not None:
            ui_init_vel.text = f"Init Vel: {initial_flywheel_vel:.2f} rad/s"
        else:
            ui_init_vel.text = "Init Vel: --"
            
        ui_max_vel.text = f"Max Vel: {max_flywheel_vel:.2f} rad/s"
        ui_max_torque.text = f"Max Torque: {max_flywheel_torque:.2f} Nm"
    
    if _dbg % 60 == 0:
        print(f"Fwd: {forward_speed:.2f} | Slip: {lateral_speed:.3f} | "
              f"Lean: {np.rad2deg(lean):.1f}° | Tgt Lean: {np.rad2deg(target_lean_angle):.1f}° | "
              f"Curr Vel: {current_flywheel_vel:.2f} | "
              f"Curr Torq: {current_flywheel_torque:.2f} | "
              f"Max Torq: {max_flywheel_torque:.2f}")

simulation_app.close()