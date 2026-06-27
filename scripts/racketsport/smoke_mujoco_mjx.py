#!/usr/bin/env python3
from __future__ import annotations

import json

import jax
import mujoco
from mujoco import mjx


XML = """
<mujoco>
  <worldbody>
    <body pos="0 0 1">
      <freejoint/>
      <geom size="0.05" mass="0.1" type="sphere"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> int:
    model = mujoco.MjModel.from_xml_string(XML)
    mjx_model = mjx.put_model(model)
    data = mjx.make_data(mjx_model)
    for _ in range(5):
        data = mjx.step(mjx_model, data)

    payload = {
        "schema_version": 1,
        "jax_version": jax.__version__,
        "mujoco_version": mujoco.__version__,
        "devices": [str(device) for device in jax.devices()],
        "qpos_shape": list(data.qpos.shape),
        "qpos_head": [float(value) for value in jax.device_get(data.qpos).reshape(-1)[:3]],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
