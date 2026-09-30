import joblib
import torch
import numpy as np

from gear_sonic.utils.motion_lib import torch_humanoid_batch
from omegaconf import OmegaConf


PKL_FILE = "data/elf3_motions/robot/Neutral_walk_forward_002__A057.pkl"


def main():
    data = joblib.load(PKL_FILE)

    name = list(data.keys())[0]
    motion = data[name]

    print("Motion:", name)
    print("DOF:", motion["dof"].shape)
    print("pose_aa:", motion["pose_aa"].shape)

    cfg = OmegaConf.create(
        {
            "asset": {
                "assetRoot": "gear_sonic/data/assets/robot_description/mjcf/elf3/",
                "assetFileName": "elf3.xml",
                "urdfFileName": "",
            },
            "extend_config": [],
        }
    )

    fk = torch_humanoid_batch.Humanoid_Batch(cfg)

    pose_aa = torch.tensor(
        motion["pose_aa"],
        dtype=torch.float32
    ).unsqueeze(0)

    root_trans = torch.tensor(
        motion["root_trans_offset"],
        dtype=torch.float32
    ).unsqueeze(0)

    result = fk.fk_batch(
        pose_aa,
        root_trans
    )

    print("FK result keys:")
    print(result.keys())

    print(
        "Body positions:",
        result["global_translation"].shape
    )


if __name__ == "__main__":
    main()