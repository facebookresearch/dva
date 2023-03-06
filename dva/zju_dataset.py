# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob

from torch.utils.data.dataset import Dataset

import numpy as np
import cv2
import json

from .io import load_opencv_calib, load_smplx_params

import logging

logger = logging.getLogger(f"care.{__name__}")


class ZJUDataset(Dataset):
    def __init__(
        self,
        smplx_poses,
        image,
        image_mask,
        image_part_mask,
        extrinsics,
        intrinsics,
        frame_list=None,
        cameras=None,
        cond_cameras=None,
        sample_cameras=True,
        camera_id=None,
        image_height=1024,
        image_width=1024,
        **kwargs,
    ):
        super().__init__()

        self.image_height = image_height
        self.image_width = image_width

        if frame_list is None:
            n_frames = len(glob.glob(os.path.dirname(self.smplx_path) + "*.json"))
            self.frame_list = np.arange(n_frames).to(str)
        else:
            self.frame_list = np.loadtxt(frame_list, dtype=str)
        self.smplx_path = smplx_poses
        self.image_path = image
        self.image_mask_path = image_mask
        self.image_part_mask_path = image_part_mask

        all_cameras = load_opencv_calib(extrinsics, intrinsics)
        if cameras is None:
            cameras = all_cameras.keys()

        logger.info(f"loading {len(cameras)} cameras")
        self.cameras = {c: all_cameras[c] for c in cameras}

        # these are ids
        self.cond_cameras = cond_cameras

        for camera in self.cameras.values():
            camera["cam_pos"] = -np.dot(camera["Rt"][:3, :3].T, camera["Rt"][:3, 3])
            camera["Rt"][:, -1] *= 1000.0
        self.sample_cameras = sample_cameras
        self.camera_id = camera_id

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):

        seq, frame = self.frame_list[idx]
        camera_id = (
            np.random.choice(list(self.cameras.keys()))
            if self.sample_cameras
            else self.camera_id
        )
        fmts = dict(frame=int(frame), seq=seq, camera=camera_id)

        sample = {"index": idx, **fmts}

        with open(self.smplx_path.format(**fmts), "r") as fh:
            sample.update(load_smplx_params(json.load(fh)))

        sample["image"] = np.transpose(
            cv2.imread(self.image_path.format(**fmts))[..., ::-1].astype(np.float32),
            axes=(2, 0, 1),
        )

        # reading all the cond images
        if self.cond_cameras:
            sample["cond_image"] = []
            sample["cond_Rt"] = []
            sample["cond_K"] = []
            for cond_camera_id in self.cond_cameras:
                cond_image = np.transpose(
                    cv2.imread(
                        self.image_path.format(
                            frame=int(frame), seq=seq, camera=cond_camera_id
                        )
                    )[..., ::-1].astype(np.float32),
                    axes=(2, 0, 1),
                )
                sample["cond_image"].append(cond_image)
                sample["cond_Rt"].append(self.cameras[cond_camera_id]["Rt"])
                sample["cond_K"].append(self.cameras[cond_camera_id]["K"])

            for key in ["image", "K", "Rt"]:
                sample[f"cond_{key}"] = np.stack(sample[f"cond_{key}"], axis=0)

            sample["cond_cameras"] = self.cond_cameras[:]

        sample["image"] = np.transpose(
            cv2.imread(self.image_path.format(**fmts))[..., ::-1].astype(np.float32),
            axes=(2, 0, 1),
        )

        sample["image_mask"] = (
            cv2.imread(self.image_mask_path.format(**fmts))[np.newaxis, ..., 0].astype(
                np.float32
            )
            != 0
        ).astype(np.float32)

        sample["image_part_mask"] = cv2.imread(
            self.image_part_mask_path.format(**fmts)
        )[np.newaxis, ..., 0]

        sample["image_bg"] = sample["image"] * ~(sample["image_part_mask"] != 0)

        sample.update(self.cameras[camera_id])

        return sample
