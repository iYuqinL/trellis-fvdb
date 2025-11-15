# -*- coding:utf-8 -*-
###
# File: mesh.py
# Created Date: Friday, November 14th 2025, 7:54:42 pm
# Author: iYuqinL
# -----
# Last Modified: 
# Modified By: 
# -----
# Copyright © 2025 iYuqinL Holding Limited
# 
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###
from typing import Union, Literal
import numpy as np
import trimesh


class RandomRotate(object):
    def __init__(self,
                 x_angle_deg: Union[float, tuple[float, float]],
                 y_angle_deg: Union[float, tuple[float, float]],
                 z_angle_deg: Union[float, tuple[float, float]],
                 inplace: bool = True):
        super(RandomRotate, self).__init__()
        self.inplace = inplace
        assert isinstance(x_angle_deg, (float, int, tuple, list)), (
            f"not support x_angle_deg type {type(x_angle_deg)}")
        assert isinstance(y_angle_deg, (float, int, tuple, list)), (
            f"not support y_angle_deg type {type(y_angle_deg)}")
        assert isinstance(z_angle_deg, (float, int, tuple, list)), (
            f"not support z_angle_deg type {type(z_angle_deg)}")

        self.x_angle_deg = (
            [-x_angle_deg, x_angle_deg]
            if isinstance(x_angle_deg, (float, int)) else x_angle_deg)
        assert len(self.x_angle_deg) == 2
        self.y_angle_deg = (
            [-y_angle_deg, y_angle_deg]
            if isinstance(y_angle_deg, (float, int)) else y_angle_deg)
        assert len(self.y_angle_deg) == 2
        self.z_angle_deg = (
            [-z_angle_deg, z_angle_deg]
            if isinstance(z_angle_deg, (float, int)) else z_angle_deg)
        assert len(self.z_angle_deg) == 2

    def __call__(self, mesh: trimesh.Trimesh):
        trans = trimesh.transformations
        xrotang = np.random.uniform(self.x_angle_deg[0], self.x_angle_deg[1])
        xrotmat = trans.rotation_matrix(np.deg2rad(xrotang), direction=[1, 0, 0])
        yrotang = np.random.uniform(self.y_angle_deg[0], self.y_angle_deg[1])
        yrotmat = trans.rotation_matrix(np.deg2rad(yrotang), direction=[1, 0, 0])
        zrotang = np.random.uniform(self.z_angle_deg[0], self.z_angle_deg[1])
        zrotmat = trans.rotation_matrix(np.deg2rad(zrotang), direction=[0, 0, 1])

        rotmat = zrotmat @ xrotmat @ yrotmat
        mesh = mesh.copy() if not self.inplace else mesh
        mesh = mesh.apply_transform(rotmat)

        return mesh



class Normalize(object):
    def __init__(self,
                 norm_len: float = 1.0,
                 norm_type: Literal["axis", "diag"] = "axis",
                 centralize: bool = True,
                 inplace: bool = True):
        self.norm_len = norm_len
        self.norm_type = norm_type
        self.centralize = centralize
        self.inplace = inplace

    def __call__(self, mesh: trimesh.Trimesh):
        mesh = mesh.copy() if not self.inplace else mesh
        mesh_bounds = mesh.bounds
        min_bound, max_bound = mesh_bounds[0], mesh_bounds[1]
        center = (min_bound + max_bound) / 2
        if not self.centralize:
            center[...] = 0.0

        if self.norm_type == "axis":
            mesh_size = max_bound - min_bound
            scale = self.norm_len / np.max(mesh_size)
            mesh.apply_translation(-center)
            mesh.apply_scale(scale)
        elif self.norm_type == "diag":
            diag_len = np.linalg.norm(max_bound - min_bound)
            scale = self.norm_len / diag_len
            mesh.apply_translation(-center)
            mesh.apply_scale(scale)
        else:
            raise ValueError(f"not support norm_type {self.norm_type}")

        return mesh
