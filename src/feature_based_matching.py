import open3d as o3d
import numpy as np
import os
os.environ['OMP_NUM_THREADS']='12'
import sys
sys.path.append(os.getcwd())
import copy 
from scipy.spatial import cKDTree
from src.preprocess import *
from src.config import REG_CONFIG


def cal_reso_pcd(pcd_arr):
   tree=cKDTree(pcd_arr)
   dd,_=tree.query(pcd_arr,k=2)
   mean_dis=np.median(dd[:,1])
   return mean_dis

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size * 2
  pcd.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

  radius_feature = voxel_size * 5
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
  return np.array(fpfh.data).T

def extract_fpfh_knn(pcd):
  pcd.estimate_normals(
      o3d.geometry.KDTreeSearchParamKNN(knn=6))

  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamKNN(knn=100))
  return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True,max_num_corr=10000):
  nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
  corres01_idx0 = np.arange(len(nns01))
  corres01_idx1 = nns01

  if not mutual_filter:
    return corres01_idx0, corres01_idx1

  nns10, dist = find_knn_cpu(feats1, feats0, knn=1, return_distance=True)
  corres10_idx1 = np.arange(len(nns10))
  corres10_idx0 = nns10

  mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
  corres_idx0 = corres01_idx0[mutual_filter]
  corres_idx1 = corres01_idx1[mutual_filter]
  if corres_idx0.shape[0]>max_num_corr:
   index=np.random.permutation(corres01_idx0.shape[0])[:max_num_corr]
   corres_idx0=corres_idx0[index]
   corres_idx1=corres_idx1[index]
  return corres_idx0, corres_idx1

def get_teaser_solver(noise_bound):
    import teaserpp_python
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = True
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver

def Rt2T(R,t,s):
    T = np.identity(4)
    T[:3,:3] = R*s
    T[:3,3] = t
    return T 

def feature_based_matching_origin(src_pc,ref_pc,config:REG_CONFIG,VISUALIZE=False):
    print("#### Start Feature-based Matching ####")
    A_pcd_raw = copy.deepcopy(src_pc)
    B_pcd_raw = copy.deepcopy(ref_pc)
    A_reso_raw=cal_reso_pcd(np.array(A_pcd_raw.points))
    B_reso_raw=cal_reso_pcd(np.array(B_pcd_raw.points))
    print("SRC reso: {}, TGT reso: {}".format(A_reso_raw,B_reso_raw))

    if VISUALIZE:
        A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
        B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red
        o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw]) # plot A and B 

    # voxel downsample both clouds
    A_reso=A_reso_raw*20*2
    B_reso=B_reso_raw*20
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=A_reso)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=B_reso)
    # A_pcd = A_pcd_raw.uniform_down_sample(10)
    # B_pcd = B_pcd_raw.uniform_down_sample(10)
    # A_reso=cal_reso_pcd(np.array(A_pcd.points))
    # B_reso=cal_reso_pcd(np.array(B_pcd.points))
    print("SRC downsample reso: {}, TGT downsample reso: {}".format(A_reso,B_reso))
    VOXEL_SIZE=A_reso if A_reso>B_reso else B_reso
    print("VOXEL SIZE: {}".format(VOXEL_SIZE))
    # if VISUALIZE:
    #     o3d.visualization.draw_geometries([A_pcd,B_pcd]) # plot downsampled A and B 

    A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M

    # extract FPFH features
    A_feats = extract_fpfh(A_pcd,A_reso)
    B_feats = extract_fpfh(B_pcd,B_reso)

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(
        A_feats, B_feats, mutual_filter=True)
    A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
    B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs

    num_corrs = A_corr.shape[1]
    print(f'FPFH generates {num_corrs} putative correspondences.')


    if VISUALIZE:
        # visualize the point clouds together with feature correspondences
        points = np.concatenate((A_corr.T,B_corr.T),axis=0)
        lines = []
        for i in range(num_corrs):
            lines.append([i,i+num_corrs])
        colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([A_pcd,B_pcd,line_set])

    # robust global registration using TEASER++
    NOISE_BOUND = 1.5*VOXEL_SIZE
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(A_corr,B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    s_teaser=solution.scale
    T_teaser = Rt2T(R_teaser,t_teaser,s_teaser)

    if VISUALIZE:
        A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_teaser)
        o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd])
    return T_teaser

def feature_based_matching(config:REG_CONFIG,VISUALIZE=False):
    import math
    ## STEP0: preprocess, 1. downsample if too big, 2. remove outlier
    REF_PC=o3d.io.read_point_cloud(config.ref_path)
    num_ref_pts=np.array(REF_PC.points).shape[0]
    if num_ref_pts>config.max_pts:
        sample_every=math.ceil(num_ref_pts/config.max_pts)
        REF_PC=REF_PC.uniform_down_sample(every_k_points=sample_every)
    REF_PC, _ = REF_PC.remove_statistical_outlier(nb_neighbors=5,std_ratio=1)
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'ref_preprocessed_feature.ply'),REF_PC)
    SRC_PC=o3d.io.read_point_cloud(config.src_path)
    num_src_pts=np.array(SRC_PC.points).shape[0]
    if num_src_pts>config.max_pts:
        sample_every=math.ceil(num_src_pts/config.max_pts)
        SRC_PC=SRC_PC.uniform_down_sample(every_k_points=sample_every)
    SRC_PC, _ = SRC_PC.remove_statistical_outlier(nb_neighbors=5,std_ratio=0.01)
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_preprocessed_feature.ply'),SRC_PC)

    ## STEP1: extract building facade, for ref data, first interpolate, and then combine
    ref_x_len,ref_y_len=get_bound_along_plane(REF_PC)
    REF_FACADE_PC,_=extract_facade_part(REF_PC,config)
    # #REF_INTERPOLATED_PC=interpolate_facade_points(REF_PC,REF_FACADE_PC,config)
    # #REF_FACADE_PC=MERGE_PC(REF_FACADE_PC,REF_INTERPOLATED_PC)
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'ref_facade_feature.ply'),REF_FACADE_PC)
    # #REF_PC=MERGE_PC(REF_PC,REF_INTERPOLATED_PC)
    # o3d.io.write_point_cloud(os.path.join(config.out_dir,'ref_preprocessed_feature.ply'),REF_PC)
    # SRC_FACADE_PC,_=extract_facade_part(SRC_PC,config)
    # o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_facade_feature.ply'),SRC_FACADE_PC)


    print("#### Start Feature-based Matching ####")
    A_pcd_raw = copy.deepcopy(SRC_PC)
    B_pcd_raw = copy.deepcopy(REF_FACADE_PC)
    A_reso_raw=cal_reso_pcd(np.array(A_pcd_raw.points))
    B_reso_raw=cal_reso_pcd(np.array(B_pcd_raw.points))
    print("SRC reso: {}, TGT reso: {}".format(A_reso_raw,B_reso_raw))

    if VISUALIZE:
        A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
        B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red
        o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw]) # plot A and B 

    # voxel downsample both clouds
    A_reso=A_reso_raw*10
    B_reso=B_reso_raw*10
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=A_reso)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=B_reso)

    print("SRC downsample reso: {}, TGT downsample reso: {}".format(A_reso,B_reso))
    VOXEL_SIZE=A_reso if A_reso>B_reso else B_reso
    print("VOXEL SIZE: {}".format(VOXEL_SIZE))

    A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M

    # extract FPFH features
    A_feats = extract_fpfh(A_pcd,A_reso)
    B_feats = extract_fpfh(B_pcd,B_reso)

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(
        A_feats, B_feats, mutual_filter=True)
    A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
    B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs

    num_corrs = A_corr.shape[1]
    print(f'FPFH generates {num_corrs} putative correspondences.')


    if VISUALIZE:
        # visualize the point clouds together with feature correspondences
        points = np.concatenate((A_corr.T,B_corr.T),axis=0)
        lines = []
        for i in range(num_corrs):
            lines.append([i,i+num_corrs])
        colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([A_pcd,B_pcd,line_set])

    # robust global registration using TEASER++
    NOISE_BOUND = 2*VOXEL_SIZE
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(A_corr,B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    s_teaser=solution.scale
    T_teaser = Rt2T(R_teaser,t_teaser,s_teaser)

    if VISUALIZE:
        A_pcd_T_teaser = copy.deepcopy(A_pcd).transform(T_teaser)
        o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd])
    return T_teaser

def keypoint(pc_path):
    pcd=o3d.io.read_point_cloud(pc_path)
    pcd_arr=np.array(pcd.points)
    reso=cal_reso_pcd(pcd_arr)
    kpts = o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                        salient_radius=reso*5,
                                                        non_max_radius=reso*5,
                                                        gamma_21=0.975,
                                                        gamma_32=0.975)
    o3d.io.write_point_cloud(pc_path[:-4]+"_kpts.ply",kpts)

   

if __name__=="__main__": 

    # src_path='/research/GDA/xuningli/wriva/data/vary/labeled_ptcld/t04_v01_s03_r03_out2/src_facade.ply'
    # ref_path='/research/GDA/xuningli/wriva/data/vary/labeled_ptcld/t04_v01_s03_r03_out2/ref_building.ply'
    # config=REG_CONFIG()
    # config.ref_path=ref_path
    # config.src_path=src_path
    # config.out_dir='/research/GDA/xuningli/wriva/data/vary/labeled_ptcld/t04_v01_s03_r03_out2/'
    # result=feature_based_matching(config)
    # print(result)
    keypoint(r'E:\data\wriva\cross-view\pair7_vary\ref_preprocessed_line.ply')

#test()