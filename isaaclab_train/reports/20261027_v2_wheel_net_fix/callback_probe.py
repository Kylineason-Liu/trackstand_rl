import sys,json
sys.path.insert(0,'/home/lmbike/trackstand_rl/isaaclab_train')
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import gymnasium as gym
import torch,warp as wp
import source.trackstand_rl.trackstand_rl.tasks
from isaaclab_physx.physics.physx_manager import IsaacEvents
cfg=gym.spec('Trackstand-Balance-v2').kwargs['env_cfg_entry_point']()
cfg.scene.num_envs=4;cfg.seed=42;cfg.sim.device='cuda:0'
env=gym.make('Trackstand-Balance-v2',cfg=cfg).unwrapped
env.reset(seed=42)
robot=env.scene['robot'];idx=robot.find_joints('rear_wheel_joint')[0][0]
def native():
    value=robot.root_view.get_dof_positions()
    return (value if isinstance(value,torch.Tensor) else wp.to_torch(value))[:,idx].clone()
callback_samples=[];after_samples=[]
handle=env.sim.physics_manager.register_callback(lambda dt:callback_samples.append(native()),event=IsaacEvents.POST_PHYSICS_STEP)
old=env.recorder_manager.record_post_physics_decimation_step
def post():
    after_samples.append(native());old()
env.recorder_manager.record_post_physics_decimation_step=post
try:
    for _ in range(5): env.step(torch.tensor([[0.,.1]]*4,device=env.device))
    a=torch.stack(callback_samples);b=torch.stack(after_samples)
    result={'callback_count':len(a),'recorder_count':len(b),'max_difference':float((a-b).abs().max()),'motion_range':float((a.max(0).values-a.min(0).values).max())}
    print('PROBE',json.dumps(result),flush=True)
    assert len(a)==40 and torch.equal(a,b) and result['motion_range']>.01
    open('/tmp/trackstand_v2_wheel_callback_probe.json','w').write(json.dumps(result,indent=2)+'\n')
finally:
    handle.deregister();env.close();app.close()
