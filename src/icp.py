"""
https://github.com/NCALM-UH/CODEM/blob/main/src/codem/registration/icp.py
IcpRegistration.py
Project: CRREL-NEGGS University of Houston Collaboration
Date: February 2021

This module contains a class to co-register two point clouds using a robust
point-to-plane ICP method.

This module contains the following class:

* IcpRegistration: a class for point cloud to point cloud registration
"""
import sys
import os
sys.path.append(os.getcwd())
import open3d as o3d
import numpy as np
from src.config import REG_CONFIG
import logging
import math
import os
import warnings
from typing import Any
from typing import Dict
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np
from scipy import spatial
from scipy.spatial import cKDTree
from scipy.sparse import diags

def cal_reso_pcd(pcd_arr):
   tree=cKDTree(pcd_arr)
   dd,_=tree.query(pcd_arr,k=2)
   mean_dis=np.median(dd[:,1])
   return mean_dis

def is_invertible(a):
     return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

class IcpRegistration:
    """
    A class to solve the transformation between two point clouds. Uses point-to-
    plane ICP with robust weighting.

    Parameters
    ----------
    fnd_obj: DSM object
        the foundation DSM
    aoi_obj: DSM object
        the area of interest DSM
    config: Dictionary
        dictionary of configuration parameters

    Methods
    --------
    register
    _residuals
    _get_weights
    _apply_transform
    _scaled
    _unscaled
    _output
    """

    def __init__(
        self,
        ref_pc,
        src_pc,
        config: REG_CONFIG,
        init_trans: np.ndarray,
        out_name:str
    ) -> None:
        self.config = config
        self.out_name=out_name
        self.logger = logging.getLogger(__name__)
        #fixed pc
        self.fixed = ref_pc
        normals=np.asarray(self.fixed.normals)
        if not normals.shape[0]:
            self.fixed.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=6))
        self.normals=np.asarray(self.fixed.normals)
        self.fixed=np.array(self.fixed.points)
        if self.fixed.shape[0]>self.config.ICP_MAX_REF_PTS:
            index=np.random.permutation(self.fixed.shape[0])[:self.config.ICP_MAX_REF_PTS]
            self.fixed=self.fixed[index]
            self.normals=self.normals[index]
        self.fixed_resolution=cal_reso_pcd(self.fixed)
        #moving pc
        self.moving = np.array(src_pc.points)
        if self.moving.shape[0]>self.config.ICP_MAX_SRC_PTS:
            index=np.random.permutation(self.moving.shape[0])[:self.config.ICP_MAX_SRC_PTS]
            self.moving=self.moving[index]
        self.moving_resolution=cal_reso_pcd(self.moving)

        self.initial_transform = init_trans
        #self.outlier_thresh = 2*max(self.fixed_resolution,self.moving_resolution)
        self.outlier_thresh = 2*max(self.fixed_resolution,self.moving_resolution)

        self.residual_origins: np.ndarray = np.empty((0, 0), np.double)
        self.residual_vectors: np.ndarray = np.empty((0, 0), np.double)

        if not all(
            [
                self.fixed.shape[1] == 3,
                self.moving.shape[1] == 3,
                self.normals.shape[1] == 3,
            ]
        ):
            raise ValueError("Point and Normal Vector Must be 3D")

        if self.fixed.shape[0] < 7 or self.moving.shape[0] < 7:
            raise ValueError(
                "At least 7 points required for hte point to plane ICP algorithm."
            )

        if self.normals.shape != self.fixed.shape:
            raise ValueError(
                "Normal vector array must be same size as fixed points array."
            )

    def register(self) -> None:
        """
        Executes ICP by minimizing point-to-plane distances:
        * Find fixed point closest to each moving point
        * Find the transform that minimizes the projected distance between each
        paired fixed and moving point, where the projection is to the fixed
        point normal direction
        * Apply the transform to the moving points
        * Repeat above steps until a convergence criteria or maximum iteration
        threshold is reached
        * Assign final transformation as attribute
        """
        self.logger.info("Solving ICP registration.")


        # Apply transform from previous feature-matching registration
        moving = self._apply_transform(self.moving, self.initial_transform)

        # Remove fixed mean to decorrelate rotation and translation
        fixed_mean = np.mean(self.fixed, axis=0)
        fixed = self.fixed - fixed_mean
        moving = moving - fixed_mean

        fixed_tree = spatial.cKDTree(fixed)

        cumulative_transform = np.eye(4)
        moving_transformed = moving
        rmse = np.float64(0.0)
        previous_rmse = np.float64(1e-12)

        alpha = 2.0
        beta = (self.moving_resolution) / 2 + 0.5
        tau = 0.2

        for i in range(self.config.ICP_MAX_ITER):
            _, idx = fixed_tree.query(
                moving_transformed, k=1
            )

            include_fixed = idx[idx < fixed.shape[0]]
            include_moving = idx < fixed.shape[0]
            temp_fixed = fixed[include_fixed]
            temp_normals = self.normals[include_fixed]
            temp_moving_transformed = moving_transformed[include_moving]

            if temp_fixed.shape[0] < 7:
                raise RuntimeError(
                    "At least 7 points within the ICP outlier threshold are required."
                )

            weights = self._get_weights(
                temp_fixed, temp_normals, temp_moving_transformed, alpha, beta
            )
            alpha -= tau
            if i==0:
                current_transform, euler, distance = self._unscaled(
                    temp_fixed, temp_normals, temp_moving_transformed, weights
                )
            else:
                if self.config.ICP_SOLVE_SCALE:
                    current_transform, euler, distance = self._scaled(
                        temp_fixed, temp_normals, temp_moving_transformed, weights
                    )
                else:
                    current_transform, euler, distance = self._unscaled(
                        temp_fixed, temp_normals, temp_moving_transformed, weights
                    )

            nan_check=np.sum(np.isnan(current_transform))
            if nan_check:
                self.logger.debug("transform is nan")
                break
            
            cumulative_transform = current_transform @ cumulative_transform
            moving_transformed = self._apply_transform(moving, cumulative_transform)

            temp_moving_transformed = self._apply_transform(
                temp_moving_transformed, current_transform
            )
            squared_error = (temp_fixed - temp_moving_transformed) ** 2
            rmse = np.sqrt(
                np.sum(np.sum(squared_error, axis=1)) / temp_moving_transformed.shape[0]
            )

            relative_change_rmse = np.abs((rmse - previous_rmse) / previous_rmse)
            previous_rmse = rmse

            if relative_change_rmse < self.config.ICP_RMSE_THRESHOLD:
                self.logger.debug("ICP converged via minimum relative change in RMSE.")
                break

            if (
                euler < self.config.ICP_ANGLE_THRESHOLD
                and distance < self.config.ICP_DISTANCE_THRESHOLD
            ):
                self.logger.debug("ICP converged via angle and distance thresholds.")
                break
        print('ICP used #iters: {}'.format(i))
        # The mean removal must be accommodated to generate the actual transform
        pre_transform = np.eye(4)
        pre_transform[:3, 3] = -fixed_mean
        post_transform = np.eye(4)
        post_transform[:3, 3] = fixed_mean
        icp_transform = post_transform @ cumulative_transform @ pre_transform

        T = icp_transform @ self.initial_transform
        c = np.sqrt(T[0, 0] ** 2 + T[1, 0] ** 2 + T[2, 0] ** 2)
        if c < 0.67 or c > 1.5:
            warnings.warn(
                (
                    "Coarse regsistration solved scale between datasets exceeds 50%. "
                    "Registration is likely to fail"
                ),
                category=RuntimeWarning,
                stacklevel=2,
            )


        self.rmse_3d = rmse
        self.rmse_xyz = np.sqrt(
            np.sum((temp_fixed - temp_moving_transformed) ** 2, axis=0)
            / temp_moving_transformed.shape[0]
        )
        self.number_points = temp_moving_transformed.shape[0]
        self.logger.debug(f"ICP number of iterations = {i+1}, RMSE = {rmse}")
        ## compute overlap
        self.moving_overlap=np.mean(np.abs(temp_fixed-temp_moving_transformed) < self.outlier_thresh)
        self.moving_transformed = self._apply_transform(self.moving, T)
        moving_tree=cKDTree(self.moving_transformed)
        dd, _ = moving_tree.query(self.fixed, k=1)
        self.fixed_overlap=np.mean(dd<self.outlier_thresh)

        if self.config.ICP_SAVE_RESIDUALS:
            self.residual_origins = self._apply_transform(self.moving, T)
            self.residual_vectors = self._residuals(
                fixed_tree, fixed, self.normals, moving_transformed
            )

        self.transformation = T
        #self._output()

    def cal_metrics(self) -> None:

        # Apply transform from previous feature-matching registration
        moving = self._apply_transform(self.moving, self.initial_transform)

        # Remove fixed mean to decorrelate rotation and translation
        fixed_mean = np.mean(self.fixed, axis=0)
        fixed = self.fixed - fixed_mean
        moving = moving - fixed_mean

        fixed_tree = cKDTree(fixed)
        moving_tree=cKDTree(moving)

        dd, idx = fixed_tree.query(moving, k=1)

        squared_error = (dd) ** 2
        rmse = np.sqrt(np.mean(squared_error))
        self.rmse_3d = rmse
        self.rmse_xyz = rmse
        self.number_points = moving.shape[0]
        ## compute overlap
        self.moving_overlap=np.mean(dd < self.outlier_thresh)
    
        dd, _ = moving_tree.query(self.fixed, k=1)
        self.fixed_overlap=np.mean(dd<self.outlier_thresh)


    def _residuals(
        self,
        fixed_tree: spatial.cKDTree,
        fixed: np.ndarray,
        normals: np.ndarray,
        moving: np.ndarray,
    ) -> np.ndarray:
        """
        Generates residual vectors for visualization purposes to illustrate the
        approximate orthogonal difference between the foundation and AOI
        surfaces that remains after registration. Note that these residuals will
        always be in meters.
        """
        _, idx = fixed_tree.query(moving, k=1)
        include_fixed = idx[idx < fixed.shape[0]]
        include_moving = idx < fixed.shape[0]
        temp_fixed = fixed[include_fixed]
        temp_normals = normals[include_fixed]
        temp_moving = moving[include_moving]

        residuals = np.sum((temp_moving - temp_fixed) * temp_normals, axis=1)
        residual_vectors: np.ndarray = (temp_normals.T * residuals).T
        return residual_vectors

    def _get_weights(
        self,
        fixed: np.ndarray,
        normals: np.ndarray,
        moving: np.ndarray,
        alpha: float,
        beta: float,
    ) -> diags:
        """
        A dynamic weight function from an as yet unpublished manuscript. Details
        will be inserted once the manuscript is published. Traditional robust
        least squares weight functions rescale the errors on each iteration;
        this method essentially rescales the weight funtion on each iteration
        instead. It appears to work slightly better than traditional methods.

        Parameters
        ----------
        fixed: np.array
            Array of fixed points ordered to correspond to the moving points
        normals: np.array
            Array of normal vectors corresponding to the fixed points
        moving: np.array
            Array of points to be transformed to the fixed point locations
        alpha: float
            Scalar value that controls how the weight function changes
        beta: float
            Scalar value that loosely represents the random noise in the data

        Returns
        -------
        weights: diags
            Sparse matrix of weights
        """
        r = np.sum((moving - fixed) * normals, axis=1)
        if alpha != 0:
            weights = (1 + (r / beta) ** 2) ** (alpha / 2 - 1)
        else:
            weights = beta**2 / (beta**2 + r**2)

        return diags(weights)

    def _apply_transform(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
        Applies a 4x4 homogeneous transformation matrix to an array of 3D point
        coordinates.

        Parameters
        ----------
        points: np.array
            Array of 3D points to be transformed
        transform: np.array
            4x4 transformation matrix to apply to points

        Returns
        -------
        transformed_points: np.array
            Array of transformed 3D points
        """
        if transform.shape != (4, 4):
            raise ValueError(
                f"Transformation matrix is an invalid shape: {transform.shape}"
            )
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points: np.ndarray = np.transpose(transform @ points.T)[:, 0:3]
        return transformed_points

    def _scaled(
        self, fixed: np.ndarray, normals: np.ndarray, moving: np.ndarray, weights: diags
    ) -> Tuple[np.ndarray, float, float]:
        """
        Solves a scaled rigid-body transformation (7-parameter) that minimizes
        the point to plane distances.

        Parameters
        ----------
        fixed: np.array
            Array of fixed points ordered to correspond to the moving points
        normals: np.array
            Array of normal vectors corresponding to the fixed points
        moving: np.array
            Array of points to be transformed to the fixed point locations
        weights: scipy.sparse.diags
            Sparse matrix of weights for robustness against outliers

        Returns
        -------
        transform: np.array
            4x4 transformation matrix
        euler: float
            The rotation angle in terms of a single rotation axis
        distance: float
            The translation distance
        """
        b = np.sum(fixed * normals, axis=1)
        A1 = np.cross(moving, normals)
        A2 = normals
        A3 = np.expand_dims(np.sum(moving * normals, axis=1), axis=1)
        A = np.hstack((A1, A2, A3))

        if self.config.ICP_ROBUST:
            x = np.linalg.inv(A.T @ weights @ A) @ A.T @ weights @ b
        else:
            x = np.linalg.inv(A.T @ A) @ A.T @ b

        x[:3] /= x[6]

        R = np.eye(3)
        T = np.zeros(3)
        R[0, 0] = np.cos(x[2]) * np.cos(x[1])
        R[0, 1] = -np.sin(x[2]) * np.cos(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.sin(
            x[0]
        )
        R[0, 2] = np.sin(x[2]) * np.sin(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.cos(
            x[0]
        )
        R[1, 0] = np.sin(x[2]) * np.cos(x[1])
        R[1, 1] = np.cos(x[2]) * np.cos(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.sin(
            x[0]
        )
        R[1, 2] = -np.cos(x[2]) * np.sin(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.cos(
            x[0]
        )
        R[2, 0] = -np.sin(x[1])
        R[2, 1] = np.cos(x[1]) * np.sin(x[0])
        R[2, 2] = np.cos(x[1]) * np.cos(x[0])
        T[0] = x[3]
        T[1] = x[4]
        T[2] = x[5]
        c = x[6]

        transform = np.eye(4)
        transform[:3, :3] = c * R
        transform[:3, 3] = T

        euler = np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        distance = np.sqrt(np.sum(T**2))

        return transform, euler, distance

    def _unscaled(
        self, fixed: np.ndarray, normals: np.ndarray, moving: np.ndarray, weights: diags
    ) -> Tuple[np.ndarray, float, float]:
        """
        Solves a rigid-body transformation (6-parameter) that minimizes
        the point to plane distances.

        Parameters
        ----------
        fixed: np.array
            Array of fixed points ordered to correspond to the moving points
        normals: np.array
            Array of normal vectors corresponding to the fixed points
        moving: np.array
            Array of points to be transformed to the fixed point locations
        weights: scipy.sparse.diags
            Sparse matrix of weights for robustness against outliers

        Returns
        -------
        transform: np.array
            4x4 transformation matrix
        euler: float
            The rotation angle in terms of a single rotation axis
        distance: float
            The translation distance
        """
        b1 = np.sum(fixed * normals, axis=1)
        b2 = np.sum(moving * normals, axis=1)
        b = np.expand_dims(b1 - b2, axis=1)
        A1 = np.cross(moving, normals)
        A2 = normals
        A = np.hstack((A1, A2))

        if self.config.ICP_ROBUST:
            C=A.T @ weights @ A
            if is_invertible(C):
                x = np.linalg.inv(A.T @ weights @ A) @ A.T @ weights @ b
            else:
                x = np.linalg.inv(A.T @ A) @ A.T @ b
        else:
            x = np.linalg.inv(A.T @ A) @ A.T @ b

        R = np.eye(3)
        T = np.zeros(3)
        R[0, 0] = np.cos(x[2]) * np.cos(x[1])
        R[0, 1] = -np.sin(x[2]) * np.cos(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.sin(
            x[0]
        )
        R[0, 2] = np.sin(x[2]) * np.sin(x[0]) + np.cos(x[2]) * np.sin(x[1]) * np.cos(
            x[0]
        )
        R[1, 0] = np.sin(x[2]) * np.cos(x[1])
        R[1, 1] = np.cos(x[2]) * np.cos(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.sin(
            x[0]
        )
        R[1, 2] = -np.cos(x[2]) * np.sin(x[0]) + np.sin(x[2]) * np.sin(x[1]) * np.cos(
            x[0]
        )
        R[2, 0] = -np.sin(x[1])
        R[2, 1] = np.cos(x[1]) * np.sin(x[0])
        R[2, 2] = np.cos(x[1]) * np.cos(x[0])
        T[0] = x[3]
        T[1] = x[4]
        T[2] = x[5]

        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = T

        euler = np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        distance = np.sqrt(np.sum(T**2))

        return transform, euler, distance

    def _output(self) -> None:
        """
        Stores registration results in a dictionary and writes them to a file
        """
        X = self.transformation
        R = X[0:3, 0:3]
        c = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2 + R[2, 0] ** 2)
        omega = np.rad2deg(np.arctan2(R[2, 1] / c, R[2, 2] / c))
        phi = np.rad2deg(-np.arcsin(R[2, 0] / c))
        kappa = np.rad2deg(np.arctan2(R[1, 0] / c, R[0, 0] / c))
        tx = X[0, 3]
        ty = X[1, 3]
        tz = X[2, 3]

        self.registration_parameters= {
            "matrix": X,
            "omega": omega,
            "phi": phi,
            "kappa": kappa,
            "trans_x": tx,
            "trans_y": ty,
            "trans_z": tz,
            "scale": c,
            "n_pairs": self.number_points,
            "rmse_x": self.rmse_xyz[0],
            "rmse_y": self.rmse_xyz[1],
            "rmse_z": self.rmse_xyz[2],
            "rmse_3d": self.rmse_3d,
        }
        output_file = os.path.join(self.config.out_dir, self.out_name)

        self.logger.info(f"Saving ICP registration parameters to: {output_file}")
        with open(output_file, "a", encoding="utf_8") as f:
            f.write("ICP REGISTRATION")
            f.write("\n----------------")
            f.write(f"\nTransformation matrix: \n {X}")
            f.write(f"\nScale = {c:.6f}")
            f.write(f"\nmoving # Pts = {self.moving.shape[0]}")
            f.write(f"\nmoving resolution = {self.moving_resolution}")
            f.write(f"\nfixed # Pts = {self.fixed.shape[0]}")
            f.write(f"\nfixed resolution = {self.fixed_resolution}")
            f.write(f"\noutlier thresh = {self.outlier_thresh}")
            f.write(f"\nmoving overlap = {self.moving_overlap}")
            f.write(f"\nfixed overlap = {self.fixed_overlap}")
            f.write(f"\nRMSE = {self.rmse_3d}")
            f.write("\n\n")

def point2plane_icp(src_pc,ref_pc,config:REG_CONFIG,init_trans: np.ndarray=np.identity(4),result_name='icp_result.txt'):
    icp_reg=IcpRegistration(ref_pc,src_pc,config,init_trans,result_name)
    icp_reg.register()
    out_trans=icp_reg.transformation
    return icp_reg.transformation, icp_reg.moving_overlap,icp_reg.fixed_overlap,icp_reg.rmse_3d

def point2plane_icp_new(src_path,ref_path,config:REG_CONFIG,init_trans: np.ndarray=np.identity(4),result_name='icp_result.txt'):
    import time
    ref_pc=o3d.io.read_point_cloud(ref_path)
    src_pc=o3d.io.read_point_cloud(src_path)
    start=time.time()
    icp_reg=IcpRegistration(ref_pc,src_pc,config,init_trans,result_name)
    icp_reg.register()
    out_trans=icp_reg.transformation
    end=time.time()
    print("{}".format(end-start))
    return icp_reg.transformation, icp_reg.moving_overlap,icp_reg.fixed_overlap,icp_reg.rmse_3d

def cal_metrics(src_path,ref_path,config:REG_CONFIG,init_trans: np.ndarray=np.identity(4),result_name='icp_result.txt'):
    import copy
    ref_pc=o3d.io.read_point_cloud(ref_path)
    src_pc=o3d.io.read_point_cloud(src_path)
    icp_reg=IcpRegistration(ref_pc,src_pc,config,init_trans,result_name)
    icp_reg.cal_metrics()
    return icp_reg.moving_overlap,icp_reg.fixed_overlap,icp_reg.rmse_3d

if __name__=="__main__": 
    src_path=r'E:\tmp\lidar_msp\cov_2021-04-23_13-18-38_S2223314_DxO_res_new.ply'
    ref_path=r'E:\tmp\lidar_msp\LiDAR - Cloud.ply'
    # src_path=r'J:\xuningli\wriva\data\vary\labeled_ptcld\t04_v01_s02_r03_out1\src_facade.ply'
    # ref_path=r'J:\xuningli\wriva\data\vary\labeled_ptcld\t04_v01_s02_r03_out1\ref_building.ply'
    # src_path=r'E:\data\test1\pair5\src.ply'
    # ref_path=r'E:\data\test1\pair5\ref.ply'
    # src_pc=o3d.io.read_point_cloud(src_path)
    # ref_pc=o3d.io.read_point_cloud(ref_path)
    config=REG_CONFIG()
    config.ICP_SOLVE_SCALE=False
    config.ICP_MAX_ITER=100
    config.ICP_ROBUST=True
    config.out_dir=r'E:\tmp\lidar_msp'
    # init_trans=np.array([[-0.2379713441975079,
	# 			0.033631318445817567,
	# 			-0.4767218210823418,
	# 			1.5855017840252983],
    #             [-0.4330288259953807,
	# 			0.25439395054297456,
	# 			0.23417312860772328,
	# 			0.8323667881983745],
    #             [0.20310450644183435,
	# 			0.6203706390341398,
	# 			-0.05745246685861662,
	# 			2.616387103608388],
    #             [0.0,
	# 			0.0,
	# 			0.0,
	# 			1.0]])
    init_trans=np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0.0,0.0,0.0,1.0]])
    #init_trans=np.identity(4)
    out=point2plane_icp_new(src_path,ref_path,config,init_trans)
    np.savetxt(os.path.join(config.out_dir,'reg.txt'),out[0])
    print(list(out))

#test()
