#!/usr/bin/env python3
"""Headless in-app probe: load the patched stage, start physics, and ask
PhysX what it actually cooked for the two wheel collision prims.

Run with:
    isaac-sim.sh --headless --no-window --exec scripts/probe_sdf_in_app.py

Fully synchronous (kit's asyncio engine deadlocks under asyncio.run):
everything is driven by app.update() pumping. Results go to /tmp/sdf_probe_out.txt.
"""

import os
import time

import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from pxr import Usd

STAGE = ("/home/lmbike/lmbike_cyclerl-mod/trackstand_rl_0914_description_sdf/"
         "urdf/trackstand_rl_0914/trackstand_rl_0914.usda")
OUT = "/tmp/sdf_probe_out.txt"

WHEELS = [
    "/trackstand_rl_0914/Geometry/base_link/rear_wheel_link_1/rear_wheel_link_1_1/rear_wheel_link_1",
    "/trackstand_rl_0914/Geometry/base_link/steering_link_1/front_wheel_link_1/front_wheel_link_1_1/front_wheel_link_1",
]

_out = open(OUT, "w")


def log(msg):
    print(msg, flush=True)
    _out.write(msg + "\n")
    _out.flush()


def pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.05)


def main():
    app = omni.kit.app.get_app()
    try:
        ctx = omni.usd.get_context()
        ctx.open_stage(STAGE)
        pump(app, 1.0)
        stage = Usd.Stage.Open(STAGE)
        for p in WHEELS:
            prim = stage.GetPrimAtPath(p)
            log(f"[USD] {p.split('/')[-2]:20s} approximation = "
                f"{prim.GetAttribute('physics:approximation').Get()}")
            log(f"[USD] {p.split('/')[-2]:20s} hasAPI PhysxSDFMeshCollisionAPI = "
                f"{'PhysxSDFMeshCollisionAPI' in prim.GetAppliedSchemas()}")

        log("[physx] interfaces: " + str([n for n in dir(omni.physx) if 'interface' in n]))
        ck = omni.physx.get_physx_cooking_interface()
        log("[physx] cooking methods: " + str(
            [m for m in dir(ck) if not m.startswith('_') and callable(getattr(ck, m, None))][:30]))
        si = omni.physx.get_physx_simulation_interface()
        log("[physx] simulation methods: " + str(
            [m for m in dir(si) if not m.startswith('_') and callable(getattr(si, m, None))][:60]))
        pq = omni.physx.get_physx_property_query_interface()
        log("[physx] property query methods: " + str(
            [m for m in dir(pq) if not m.startswith('_') and callable(getattr(pq, m, None))][:60]))

        si = omni.physx.get_physx_simulation_interface()
        ctx2 = omni.usd.get_context()
        stage_id = ctx2.get_stage_id()
        try:
            si.pause_change_tracking(True)
            si.attach_stage(stage_id)
            log("[physx] stage attached to the simulation, change tracking paused")
        except Exception as e:  # noqa: BLE001
            log(f"[physx] attach_stage failed: {e}")
        pump(app, 6.0)  # scene built; async cooking finishes; bike never stepped

        # step a little so the scene builds its shapes (with change tracking
        # paused the bike barely moves: 10 steps = 1/6 s of sim)
        t_sim = 0.0
        for _ in range(10):
            si.simulate(1.0 / 60.0, t_sim)
            t_sim += 1.0 / 60.0
            si.fetch_results()
        pump(app, 2.0)

        # --- raycast discriminator: SDF torus (open hole band) vs filled hull ---
        from pxr import Gf, UsdGeom
        import math as _m
        rear_link = stage.GetPrimAtPath(
            "/trackstand_rl_0914/Geometry/base_link/rear_wheel_link_1")
        M = omni.usd.get_world_transform_matrix(rear_link)
        t = Gf.Vec3d(M.ExtractTranslation())
        rot = M.ExtractRotationMatrix()
        ax = Gf.Vec3d(rot[0][1], rot[1][1], rot[2][1])  # link-frame y -> world
        n1 = Gf.Vec3d(rot[0][0], rot[1][0], rot[2][0])
        n2 = Gf.Cross(ax, n1)
        n2 = n2.GetNormalized()
        n1 = Gf.Cross(ax, n2).GetNormalized()
        log(f"[raycast] rear axle world pos = ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
        sq = omni.physx.get_physx_scene_query_interface()
        # positive sanity: a ray across the base_link hull (convex, group mesh)
        Mb = omni.usd.get_world_transform_matrix(
            stage.GetPrimAtPath("/trackstand_rl_0914/Geometry/base_link"))
        tb = Gf.Vec3d(Mb.ExtractTranslation())
        sanity = sq.raycast_any(tuple(tb + Gf.Vec3d(1.0, 0.0, 0.0)), (-1.0, 0.0, 0.0), 3.0)
        log(f"[raycast] SANITY: ray across base_link hull hit = {sanity}")
        hits = {}
        for k in range(8):
            ang = k * _m.pi / 4.0
            d = n1 * _m.cos(ang) + n2 * _m.sin(ang)
            for rho in (0.04, 0.06):
                o = t - ax * 0.05 + d * rho  # start 5 cm outside the wheel
                hit = sq.raycast_any(tuple(o), tuple(ax), 0.1)
                hits[rho] = hits.get(rho, 0) + (1 if hit else 0)
        log(f"[raycast] hole band rho=0.04: {hits[0.04]}/8 rays hit  |  "
            f"tire band rho=0.06: {hits[0.06]}/8 rays hit")
        log("[raycast] interpretation: filled hull hits 8/8 in BOTH bands; "
            "SDF torus hits ~8/8 in the tire band but ~0/8 in the hole band")

        log("[DONE]")
    except Exception as e:  # noqa: BLE001
        import traceback
        log(f"[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc(file=_out)
        _out.flush()
    finally:
        app.quit()
        os._exit(0)


main()
