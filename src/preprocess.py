import os
import sys
sys.path.append(os.getcwd())
import open3d as o3d
import numpy as np
import os
import glob
from scipy.spatial import KDTree
from src.config import REG_CONFIG
import itertools
from scipy.spatial.transform import Rotation as R

def numpy2open3d(pts_arr,normal_arr,rgb_arr):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_arr)
    pcd.normals=o3d.utility.Vector3dVector(normal_arr)
    if rgb_arr:
        pcd.colors = o3d.utility.Vector3dVector(rgb_arr)
    
    return pcd

def plane_normal(ptcloud):
    arr=ptcloud
    #pca
    mean=np.mean(arr,axis=0)
    arr_demean=arr-mean
    cov=np.transpose(arr_demean)@arr_demean
    w,v=np.linalg.eig(cov)
    min_value_ind=np.argmin(w)
    normal=v[:,min_value_ind]
    return normal

def remove_ele_origin(neighbor_ids,seed_ids):
    for seed_id in seed_ids:
        if seed_id in neighbor_ids:
            neighbor_ids.remove(seed_id)
    return neighbor_ids

def remove_ele(neighbor_ids,seed_ids):
    out_neighbors=[]
    for neighbor_id in neighbor_ids:
        if neighbor_id not in seed_ids:
            out_neighbors.append(neighbor_id)
    # for seed_id in seed_ids:
    #     if seed_id in neighbor_ids:
    #         neighbor_ids.remove(seed_id)
    return out_neighbors

#input: points Nx3
#output: labels N, region grow in 2D
def region_grow_2d(points,radius=0.5):
    points_2d=points[:,:2]
    kdtree=KDTree(points_2d)
    labels=np.ones(points.shape[0],dtype=np.int16)*-1 #-1 means unvisited

    #start region grow
    visited_ids=[]
    seg_id=0
    seed_ids=[0]
    labels[seed_ids]=seg_id
    visited_ids+=seed_ids
    visited_ids=list(np.unique(np.array(visited_ids)))
    ids=list(kdtree.query_ball_point(points_2d[seed_ids],radius))
    neighbor_ids=list(np.unique(np.array(list(itertools.chain.from_iterable(ids)))))
    neighbor_ids=remove_ele(neighbor_ids,seed_ids)
    if len(neighbor_ids)!=0:
        labels[neighbor_ids]=seg_id
        seed_ids=neighbor_ids
        visited_ids+=seed_ids
        visited_ids=list(np.unique(np.array(visited_ids)))
    else:
        seed_ids=[]
        seg_id+=1
    while (labels!=-1).all()==False: ##stop when all labels != -1
        if len(seed_ids)!=0:
            ids=list(kdtree.query_ball_point(points_2d[seed_ids],radius))
            neighbor_ids=list(np.unique(np.array(list(itertools.chain.from_iterable(ids)))))
            neighbor_ids=remove_ele(neighbor_ids,visited_ids)
            if len(neighbor_ids)!=0:
                labels[neighbor_ids]=seg_id
                seed_ids=neighbor_ids
                visited_ids+=seed_ids
                visited_ids=list(np.unique(np.array(visited_ids)))
            else:
                print("seg_id: {} finished, #visited points: {}".format(seg_id,len(visited_ids)))
                seed_ids=[]
                seg_id+=1
        else:
            seed_ids=[np.where(labels==-1)[0][0]]
            labels[seed_ids]=seg_id
            ids=list(kdtree.query_ball_point(points_2d[seed_ids],radius))
            neighbor_ids=list(np.unique(np.array(list(itertools.chain.from_iterable(ids)))))
            neighbor_ids=remove_ele(neighbor_ids,visited_ids)
            if len(neighbor_ids)!=0:
                labels[neighbor_ids]=seg_id
                seed_ids=neighbor_ids
                visited_ids+=seed_ids
                visited_ids=list(np.unique(np.array(visited_ids)))
            else:
                print("seg_id: {} finished, #visited points: {}".format(seg_id,len(visited_ids)))
                seed_ids=[]
                seg_id+=1
    return labels

def filter_regiongrow_results(points,labels,pts_thresh=200):
    max_id=np.max(labels)
    min_id=np.min(labels)
    valid_ids=[]
    for id in range(min_id,max_id+1):
        ids=np.where(labels==id)[0]
        if ids.shape[0]>pts_thresh:
            valid_ids+=list(ids)
    points_filter=points[valid_ids,:]
    labels_filter=labels[valid_ids]
    return points_filter,labels_filter

def get_bound_along_plane(pc):
    pc_pts=np.array(pc.points)
    normal_v=plane_normal(pc_pts)
    z_axis=np.array([0,0,1])
    rot_axis=np.cross(normal_v,z_axis)
    angle=np.arccos(normal_v@z_axis)
    rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()
    pc_pts_corrected=(rot_plane@pc_pts.T).T
    bound_min=np.min(pc_pts_corrected[:,:3],axis=0)
    bound_max=np.max(pc_pts_corrected[:,:3],axis=0)
    x_len=bound_max[0]-bound_min[0]
    y_len=bound_max[1]-bound_min[1]
    return x_len,y_len

def extract_facade_part(pc, config:REG_CONFIG, z_axis=None, normal_thresh=75):
    ## compute boundary length for searching radius
    pc_pts=np.array(pc.points)
    pc_colors=np.array(pc.colors)
    pc_normals=np.array(pc.normals)
    if z_axis is not None:
        normal_v=z_axis
    else:
        normal_v=plane_normal(pc_pts)
    #rot_axis=np.cross(normal_v,z_axis)
    #angle=np.arccos(normal_v@z_axis)
    #rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()

    # search_radius=0.005*min(x_len,y_len)
    # pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=6))
    # pc_normals=np.asarray(pc.normals)

    ### STEP1: building facade extarction (filter out normal angle)
    ### con1: angle > 75
    cos_angle_thresh=np.cos(normal_thresh*np.pi/180)
    cos_angles=np.abs(pc_normals@normal_v) #nx1
    building_indice1=cos_angles<cos_angle_thresh
    ### con2: neighbor > 75%
    points=np.asarray(pc.points)
    kdtree=KDTree(points)
    dd,ii=kdtree.query(points,k=10)
    neighbor=np.mean(building_indice1[ii],axis=1) #nx1
    building_indice2=neighbor>0.6
    #conbine con1 and con2
    building_indice=building_indice1 * building_indice2
    building_indice=np.squeeze(building_indice)
    points_building=points[building_indice,:]
    normals_building=pc_normals[building_indice,:]
    #colors_building=pc_colors[building_indice,:]
    ## region grow
    # if points_building.shape[0]>config.max_facade_pts:
    #     index=np.random.permutation(points_building.shape[0])[:config.max_facade_pts]
    #     points_building=points_building[index,:]
    # labels=region_grow_2d(points_building,search_radius)
    # points_building,labels=filter_regiongrow_results(points_building,labels,100)

    pc_facade=numpy2open3d(points_building,normals_building,None)
    #pc_facade, _ = pc_facade.remove_statistical_outlier(nb_neighbors=20,std_ratio=0.1)
    return pc_facade,normal_v

def MERGE_PC(pc1,pc2):
    pc_pts_merge=np.concatenate([np.array(pc1.points),np.array(pc2.points)])
    pc_normals_merge=np.concatenate([np.array(pc1.normals),np.array(pc2.normals)])
    pc_colors_merge=np.concatenate([np.array(pc1.colors),np.array(pc2.colors)])
    pc_merge = o3d.geometry.PointCloud()
    pc_merge.points = o3d.utility.Vector3dVector(pc_pts_merge)
    pc_merge.normals = o3d.utility.Vector3dVector(pc_normals_merge)
    pc_merge.colors = o3d.utility.Vector3dVector(pc_colors_merge)
    return pc_merge

def interpolate_facade_points(pc,pc_facade, config:REG_CONFIG):
    pc_pts=np.array(pc.points)
    num_pts=pc_pts.shape[0]
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=10)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    pc_sampled=mesh.sample_points_uniformly(number_of_points=num_pts)
    #o3d.io.write_point_cloud(r'/research/GDA/xuningli/wriva/data/cross_view/pair2/out2/ref_interpolated.ply',pc_sampled)
    ## the sampled points should first extarct facade, and then inpaint to those no points area
    pc_sampled_facade,_=extract_facade_part(pc_sampled,pc,25)
    #o3d.io.write_point_cloud(r'/research/GDA/xuningli/wriva/data/cross_view/pair2/out2/ref_interpolated_facade.ply',pc_sampled_facade)
    pc_sampled_facade_pts=np.array(pc_sampled_facade.points)
    pc_sampled_facade_normals=np.array(pc_sampled_facade.normals)
    pc_sampled_facade_rgb=np.array(pc_sampled_facade.colors)

    tree=KDTree(pc_pts)
    dd,_ =tree.query(pc_pts,k=2)
    reso=np.mean(dd)
    dd,ii=tree.query(pc_sampled_facade_pts,k=1)

    pcd_interpolated = o3d.geometry.PointCloud()
    pcd_interpolated.points = o3d.utility.Vector3dVector(pc_sampled_facade_pts[dd>5*reso])
    pcd_interpolated.normals=o3d.utility.Vector3dVector(pc_sampled_facade_normals[dd>5*reso])
    pcd_interpolated.colors=o3d.utility.Vector3dVector(pc_sampled_facade_rgb[dd>5*reso])
    return pcd_interpolated

def interpolate_and_merge_facade_points(pc,pc_facade, merge_thresh, config:REG_CONFIG):
    pc_pts=np.array(pc.points)
    pc_facade_pts=np.array(pc_facade.points)
    num_pts=pc_pts.shape[0]
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=6))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=10)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    pc_sampled=mesh.sample_points_uniformly(number_of_points=num_pts)
    pc_sampled_facade,_=extract_facade_part(pc_sampled,pc)
    pc_sampled_facade_pts=np.array(pc_sampled_facade.points)
    #pc_sampled_facade_rgb=np.array(pc_sampled_facade.colors)

    tree=KDTree(pc_facade_pts)
    dd,ii=tree.query(pc_sampled_facade_pts,k=1)
    pc_sampled_facade_pts_keep=pc_sampled_facade_pts[dd<merge_thresh]

    pc_pts_merge=np.concatenate([pc_facade_pts,pc_sampled_facade_pts_keep])
    pcd_interpolated = o3d.geometry.PointCloud()
    pcd_interpolated.points = o3d.utility.Vector3dVector(pc_pts_merge)
    return pcd_interpolated

def extract_building_part_zaxis(pc,sem_arr,ground_indice,building_indice):
    #determine z-axis
    ground_idx=[]
    for id in ground_indice:
        ground_idx=ground_idx+list(np.where(sem_arr==id)[0])
    ground_idx=np.array(ground_idx)
    ground_pc=np.array(pc.points)[ground_idx,:]
    ground_normals=np.array(pc.normals)[ground_idx,:]
    ground_normal1=plane_normal(ground_pc)
    ground_normal2=-ground_normal1
    sim1=np.mean(ground_normals@ground_normal1)
    sim2=np.mean(ground_normals@ground_normal2)
    if sim1>sim2:
        ground_normal=ground_normal1
    else:
        ground_normal=ground_normal2
    
    # extract building
    building_idx=[]
    for id in building_indice:
        building_idx=building_idx+list(np.where(sem_arr==id)[0])
    building_idx=np.array(building_idx)
    building_pc_raw=np.array(pc.points)[building_idx,:]
    buiiding_normals_raw=np.array(pc.normals)[building_idx,:]
    projected_heights=building_pc_raw@ground_normal
    height_min=np.quantile(projected_heights,0.1)
    height_max=np.quantile(projected_heights,0.98)
    height=height_max-height_min
    building_idx_mainpart=np.logical_and(projected_heights<height_max,projected_heights>height_min)

    #include ground part
    # ground_min=np.quantile(projected_heights,0.00)
    # ground_max=np.quantile(projected_heights,0.1)
    # building_idx_groundpart=np.logical_and(projected_heights<ground_max,projected_heights>ground_min)
    # cos_angle_thresh=np.cos(75*np.pi/180)
    # facade_idx=abs(buiiding_normals_raw@ground_normal)<cos_angle_thresh
    # building_idx_groundpart=np.logical_and(building_idx_groundpart,facade_idx)
    # building_idx_refine=np.logical_or(building_idx_mainpart,building_idx_groundpart)
    #np.savetxt(r'J:\xuningli\papers\g2a\data\ISPRS_Multi\result\zeche\uavsony_verwaltungsony1\ground_building_raw.txt',building_pc_raw)
    building_idx_refine=building_idx_mainpart

    building_pc=building_pc_raw[building_idx_refine,:]
    building_normals=buiiding_normals_raw[building_idx_refine,:]
    building_o3d=o3d.geometry.PointCloud()
    building_o3d.points = o3d.utility.Vector3dVector(building_pc)
    building_o3d.normals = o3d.utility.Vector3dVector(building_normals)

    return building_o3d, height, ground_normal





if __name__=="__main__": 
    # drone_path=r'/research/GDA/xuningli/wriva/data/cross_view/pair1/out1/ref_facade.ply'
    # drone_pc=o3d.io.read_point_cloud(drone_path)
    # pc_facade, _ = drone_pc.remove_statistical_outlier(nb_neighbors=50,std_ratio=0.01)
    # o3d.io.write_point_cloud(r'/research/GDA/xuningli/wriva/data/cross_view/pair1/out1/ref_facade_filter.ply',pc_facade)
    print('start')
    ground_path=r'/research/GDA/xuningli/wriva/data/cross_view/pair1/out1/src_preprocessed.ply'
    drone_path=r'/research/GDA/xuningli/wriva/data/cross_view/pair1/out1/ref_preprocessed.ply'
    config=REG_CONFIG()
    ground_pc=o3d.io.read_point_cloud(ground_path)
    drone_pc=o3d.io.read_point_cloud(drone_path)
    # ground_pc, _ = ground_pc.remove_statistical_outlier(nb_neighbors=5,std_ratio=0.1)
    # drone_pc, _ = drone_pc.remove_statistical_outlier(nb_neighbors=5,std_ratio=2)
    #ground_facade,_=extract_facade_part(ground_pc,config)
    drone_facade,_=extract_facade_part(drone_pc,config)
    #o3d.io.write_point_cloud(r'/research/GDA/xuningli/wriva/data/cross_view/pair1/out1/src_facade.ply',ground_facade)
    o3d.io.write_point_cloud(r'/research/GDA/xuningli/wriva/data/cross_view/pair1/out1/ref_facade.ply',drone_facade)
    #o3d.io.write_point_cloud(r'J:\xuningli\wriva\data\cross_view\apl_ground_uav\ws1\drone_facade_180.ply',drone_facade)

#test()