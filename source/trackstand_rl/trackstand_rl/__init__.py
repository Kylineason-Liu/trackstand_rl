"""trackstand_rl: standalone SDF-wheel trackstand balance task.

Package layout mirrors source/CycleRL exactly:

    trackstand_rl/
    └── tasks/
        └── manager_based/
            └── trackstand/            (task family, like cyclerl/)
                ├── __init__.py        gym registrations
                ├── agents/            rsl_rl PPO + skrl configs
                ├── assets/trackstand_rl/  the SDF-wheel USD (+ payloads)
                ├── bicycle_cfg.py     robot ArticulationCfg
                ├── mdp/               stationary MDP term library (local copy)
                └── trackstand_bike/   env class + env config (like new_bike/)
"""
