"""Show sampled episode values below plots without covering or rescaling traces."""

from functools import lru_cache
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def metadata():
    return json.loads((HERE / "sampled_dr_values.json").read_text())


def episodes_in_view(tag, env, window):
    return [item for item in metadata()["runs"][tag]["episodes"]
            if item["environment"] == env and item["start_s"] < window[1] and item["stop_s"] > window[0]]


def vector(values):
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def block(tag, env, window):
    records = episodes_in_view(tag, env, window)
    assert records, (tag, env, window)
    names = [name.replace("_link", "") for name in records[0]["body_order"]]
    lines = [f"SAMPLED DR: env {env}; body order: {', '.join(names)}",
             "Contact arrays: collision-shape order; restitution = 0.",
             "CoM XYZ: shared Isaac-link-frame offset (mm); gravity bias: unit gravity."]
    if tag == "isaac400":
        lines += ["Recovered with all 18 original capture arrays byte-identical."]
    else:
        lines += ["mu_d is sampled metadata; MuJoCo's native contact cone uses mu_s."]
    for item in records:
        lines += [
            f"E{item['episode']} [{item['start_s']:.2f}, {item['stop_s']:.2f}) s:",
            f"  mass x {vector(item['mass_scale'])}",
            f"  CoM XYZ mm {vector([1000 * v for v in item['com_shared_offset_m']])}",
            f"  mu_s {vector(item['contact_static_friction'])}",
            f"  mu_d {vector(item['contact_dynamic_friction'])}",
            f"  Kp [steer,rear] {vector(item['kp'])}; Kd {vector(item['kd'])}",
            f"  spawn roll {item['initial_roll_rad']:.6g} rad; rate {item['initial_roll_rate_rad_s']:.6g} rad/s",
            f"  gyro bias XYZ rad/s {vector(item['gyro_bias_rad_s'])}",
            f"  gravity bias YZ {vector(item['gravity_bias_yz'])}"]
    lines += ["Six significant digits shown; full precision: sampled_dr_values.json."]
    return "\n".join(lines)


def append_blocks(fig, columns):
    """Grow canvas downward; retain all existing artists' physical positions/sizes."""
    texts = [block(*column) for column in columns]
    if fig.get_layout_engine() is not None:
        fig.canvas.draw()
    width, height = fig.get_size_inches()
    line_height = 7.0 * 1.2 / 72
    extra = max(text.count("\n") + 1 for text in texts) * line_height + .32
    enlarged = height + extra
    fig.set_size_inches(width, enlarged, forward=True)
    for ax in fig.axes:
        x, y, w, h = ax.get_position().bounds
        ax.set_position([x, (y * height + extra) / enlarged, w, h * height / enlarged])
    for artist in list(fig.texts):
        x, y = artist.get_position()
        artist.set_position((x, (y * height + extra) / enlarged))
    # Freezing constrained layout prevents a second layout pass resizing the original axes.
    fig.set_layout_engine(None)
    for col, text in enumerate(texts):
        fig.text((col + .045) / len(texts), (extra - .12) / enlarged, text,
                 ha="left", va="top", fontsize=7, family="monospace", linespacing=1.2,
                 gid=f"sampled-dr-column-{col}")
