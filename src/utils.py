from scipy.spatial import cKDTree
import open3d as o3d
import numpy as np
                                                       
def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    num=feat1tree.n
    dim=feat1tree.m
    #dists, nn_inds = feat1tree.query(feat0, k=knn)
    dists, nn_inds = feat1tree.query(feat0, k=knn)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds

def FPFH_ISS(pcd_path,keypts_path,radius_feature=3):
    pcd=o3d.io.read_point_cloud(pcd_path)
    keypts=o3d.io.read_point_cloud(keypts_path)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=1024))
    key_indice=find_knn_cpu(np.array(keypts.points),np.array(pcd.points))
    pcd_arr=np.array(pcd.points)

    pcd_fpfh_arr=np.array(pcd_fpfh.data).T
    pcd_keypts_arr=pcd_arr[key_indice,:]
    pcd_keypts_fpfh_arr=pcd_fpfh_arr[key_indice,:]
    np.save(keypts_path[:-4]+"_fpfh",pcd_keypts_fpfh_arr)
    return pcd_keypts_arr,pcd_keypts_fpfh_arr

def eval_inliers(src_path,ref_path,thresh):
    src_pc=o3d.io.read_point_cloud(src_path)
    ref_pc=o3d.io.read_point_cloud(ref_path)
    src_arr=np.array(src_pc.points)
    ref_arr=np.array(ref_pc.points)
    ref_tree=cKDTree(ref_arr)
    dd,ii=ref_tree.query(src_arr,k=1)
    for i in range(1,5):
        print("inlier ratio (<{} m): {}".format(i,np.mean(dd<i)))

def eval_matches(src_kpts_path,src_path,ref_kpts_path,ref_path):
    src_pc=o3d.io.read_point_cloud(src_kpts_path)
    src_arr=np.array(src_pc.points)
    ref_pc=o3d.io.read_point_cloud(ref_kpts_path)
    ref_arr=np.array(ref_pc.points)
    src_feat=np.load(src_path)
    ref_feat=np.load(ref_path)
    ref_tree=cKDTree(ref_feat)
    dd,ii=ref_tree.query(src_feat,k=2)
    ratios=dd[:,0]/dd[:,1]
    threshs=[0.5,0.7,0.9,0.95,0.97]
    for i in threshs:
        print("valid match ratio (<{} ): {}, #matches: {}".format(i,np.mean(ratios<i),np.sum(ratios<i)))
        inliers=ratios<i
        match=ii[inliers,0]
        match_src=src_arr[inliers,:]
        match_ref=ref_arr[match,:]
        dis=np.linalg.norm(match_ref-match_src,axis=1)
        for j in range(1,5):
            print("inliers match ratio (<{} m): {}".format(j,np.mean(dis<j)))




if __name__=="__main__": 
    #eval_inliers(r'E:\data\wriva\cross-view\pair7_vary\src_preprocessed_line_kpts.ply',r'E:\data\wriva\cross-view\pair7_vary\ref_preprocessed_line_kpts.ply',1)
    eval_inliers(r'E:\data\wriva\cross-view\pair7_vary\ref_preprocessed_line_kpts.ply',r'E:\data\wriva\cross-view\pair7_vary\src_preprocessed_line_kpts.ply',1)
    # FPFH_ISS(r'E:\data\wriva\cross-view\pair7_vary\src_preprocessed_line.ply',r'E:\data\wriva\cross-view\pair7_vary\src_preprocessed_line_kpts.ply')
    # FPFH_ISS(r'E:\data\wriva\cross-view\pair7_vary\ref_preprocessed_line.ply',r'E:\data\wriva\cross-view\pair7_vary\ref_preprocessed_line_kpts.ply')
    # eval_matches(r'E:\data\wriva\cross-view\pair7_vary\src_preprocessed_line_kpts.ply',
    #             r'E:\data\wriva\cross-view\pair7_vary\src_preprocessed_line_kpts_fpfh.npy',
    #             r'E:\data\wriva\cross-view\pair7_vary\ref_preprocessed_line_kpts.ply',
    #             r'E:\data\wriva\cross-view\pair7_vary\ref_preprocessed_line_kpts_fpfh.npy')