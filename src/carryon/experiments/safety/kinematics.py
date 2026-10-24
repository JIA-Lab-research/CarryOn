"""Check the actual Shield FK dependencies before loading policy models."""
import json
import mujoco
import robosuite
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory


def main():
    robot = Panda()
    robot.add_gripper(gripper_factory("PandaGripper"))
    model = mujoco.MjModel.from_xml_string(robot.get_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if (model.nq, model.nv) != (9, 9):
        raise ValueError("Unexpected Panda arm/gripper degrees of freedom")
    print(json.dumps({"status": "passed", "nq": model.nq, "nv": model.nv,
                      "mujoco": mujoco.__file__, "robosuite": robosuite.__file__}))


if __name__ == "__main__":
    main()
