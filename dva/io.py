# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import cv2
import numpy as np
import copy
import importlib
from typing import Any, Dict

from dva.attr_dict import AttrDict
from dva.geom import compute_v2uv, compute_neighbours


def load_module(module_name, class_name=None, silent: bool = False):
    module = importlib.import_module(module_name)
    return getattr(module, class_name) if class_name else module


def load_class(class_name):
    return load_module(*class_name.rsplit(".", 1))


def load_from_config(config, **kwargs):
    """Instantiate an object given a config and arguments."""
    assert "class_name" in config and "module_name" not in config
    config = copy.deepcopy(config)
    class_name = config.pop("class_name")
    object_class = load_class(class_name)
    return object_class(**config, **kwargs)


def load_opencv_calib(extrin_path, intrin_path):
    cameras = {}

    fse = cv2.FileStorage()
    fse.open(extrin_path, cv2.FileStorage_READ)

    fsi = cv2.FileStorage()
    fsi.open(intrin_path, cv2.FileStorage_READ)

    names = [
        fse.getNode("names").at(c).string() for c in range(fse.getNode("names").size())
    ]

    for camera in names:
        rot = fse.getNode(f"R_{camera}").mat()
        R = fse.getNode(f"Rot_{camera}").mat()
        T = fse.getNode(f"T_{camera}").mat()
        R_pred = cv2.Rodrigues(rot)[0]
        assert np.all(np.isclose(R_pred, R))
        K = fsi.getNode(f"K_{camera}").mat()
        cameras[camera] = {
            "Rt": np.concatenate([R, T], axis=1).astype(np.float32),
            "K": K.astype(np.float32),
        }
    return cameras


def load_smplx_params(params):
    return {
        k: np.array(v[0], dtype=np.float32) for k, v in params[0].items() if k != "id"
    }


def load_smplx_topology(data_struct) -> Dict[str, Any]:
    # TODO: compute_
    topology = {
        "vi": data_struct["f"].astype(np.int64),
        "vti": data_struct["ft"].astype(np.int64),
        "vt": data_struct["vt"].astype(np.float32),
        "n_verts": data_struct["v_template"].shape[0],
    }
    topology["v2uv"] = compute_v2uv(
        topology["n_verts"], topology["vi"], topology["vti"]
    )

    nbs_idxs, nbs_weights = compute_neighbours(
        topology["v"].shape[0], topology["vi"], 8
    )

    topology.update({"nbs_idxs": nbs_idxs, "nbs_weights": nbs_weights})

    return {
        "topology": topology,
        "lbs_template_verts": data_struct["v_template"].astype(np.float32),
    }


def load_static_assets(config):
    data_struct = np.load(config.data.smplx_topology)

    n_verts = data_struct["v_template"].shape[0]

    topology = AttrDict(
        dict(
            vi=data_struct["f"].astype(np.int64),
            vt=data_struct["vt"].astype(np.float32),
            vti=data_struct["ft"].astype(np.int64),
            n_verts=n_verts,
        )
    )

    topology.v2uv = compute_v2uv(topology.n_verts, topology.vi, topology.vti)

    nbs_idxs, nbs_weights = compute_neighbours(topology.n_verts, topology["vi"])
    topology.nbs_idxs = nbs_idxs
    topology.nbs_weights = nbs_weights

    static_assets = AttrDict(
        dict(
            topology=topology,
            lbs_template_verts=data_struct["v_template"],
            smplx_path=config.smplx_dir,
        )
    )

    if "ref_frame" in config:
        with open(
            config.data.smplx_poses.format(frame=int(config.ref_frame)), "r"
        ) as fh:
            ref_frame = load_smplx_params(json.load(fh))

        static_assets["ref_frame"] = {k: v[np.newaxis] for k, v in ref_frame.items()}

    return static_assets
