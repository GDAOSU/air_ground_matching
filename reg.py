import os
os.environ['OMP_NUM_THREADS']='12'
import glob
import open3d as o3d
import numpy as np
import math
from src.icp import point2plane_icp_new,cal_metrics,cal_reso_pcd
from src.config import REG_CONFIG
import argparse

def cal_metrics_structural(src_path,ref_path,init_trans):
    from src.icp import point2plane_icp_new,cal_metrics,cal_reso_pcd
    from scipy.spatial import KDTree
    src_o3d=o3d.io.read_point_cloud(src_path)
    src_o3d=src_o3d.uniform_down_sample(every_k_points=100)
    src_normals=np.array(src_o3d.normals)
    src_pts=np.array(src_o3d.points)
    ref_o3d=o3d.io.read_point_cloud(ref_path)
    ref_o3d=ref_o3d.uniform_down_sample(every_k_points=100)
    ref_normals=np.array(ref_o3d.normals)
    ref_pts=np.array(ref_o3d.points)
    
    R=init_trans[:3,:3]
    src_pts = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
    src_pts_trans = np.transpose(init_trans @ src_pts.T)[:, 0:3]
    src_normals_trans=(R@src_normals.T).T

    #determine threshold
    ref_reso=cal_reso_pcd(ref_pts)
    src_reso=cal_reso_pcd(src_pts_trans)
    outlier_thresh = 2*max(ref_reso,src_reso)

    ref_tree=KDTree(ref_pts)
    dd,ii=ref_tree.query(src_pts_trans,k=1)
    src_overlap=np.mean(dd<outlier_thresh)
    
    match_dirs=(src_pts_trans-ref_pts[ii,:])/np.expand_dims(np.linalg.norm(src_pts_trans-ref_pts[ii,:],axis=1),axis=1)
    match_src=np.mean(abs(np.sum(match_dirs*src_normals_trans,axis=1)))
    match_ref=np.mean(abs(np.sum(match_dirs*ref_normals[ii],axis=1)))
    normal_similarity=np.mean(np.sum(ref_normals[ii,:]*src_normals_trans,axis=1))
    structure_similarity=(match_src+match_ref+normal_similarity)/3

    return src_overlap, structure_similarity


def Register(config:REG_CONFIG):
    print("Perform air-ground registration")
    #from src.preprocess import *
    from src.line_based_matching import line_based_matching
    from src.feature_based_matching import feature_based_matching
    ## Line-based matching
    trans_line=line_based_matching(config)

    ## feature-based matching
    trans_feature=feature_based_matching(config)

    ## STEP2: line-based and feature-based matching
    rough_trans_list=[]
    rough_trans_list.append(trans_line)
    rough_trans_list.append(trans_feature)
    rough_trans_list.append(config.init_trans)

    ## STEP3: icp-based refine on three rough trans, choose the best overlap one
    final_trans_results_list=[]
    #line -based
    fine_trans,_,_,_=point2plane_icp_new(os.path.join(config.out_dir,"src_preprocessed_line.ply"),os.path.join(config.out_dir,"ref_preprocessed_line.ply"),config,rough_trans_list[0],"icp_result_{}.txt".format("line_based"))
    rough_overlap, rough_structural_similarity=cal_metrics_structural(os.path.join(config.out_dir,"src_preprocessed_line.ply"),os.path.join(config.out_dir,"ref_preprocessed_line.ply"),rough_trans_list[0])
    fine_overlap, fine_structural_similarity=cal_metrics_structural(os.path.join(config.out_dir,"src_preprocessed_line.ply"),os.path.join(config.out_dir,"ref_preprocessed_line.ply"),fine_trans)
    final_trans_results_list.append({'trans':rough_trans_list[0].tolist(),'overlap':rough_overlap,'structural_sim':rough_structural_similarity})
    final_trans_results_list.append({'trans':fine_trans.tolist(),'overlap':fine_overlap,'structural_sim':fine_structural_similarity})

    #feature -based
    fine_trans,_,_,_=point2plane_icp_new(os.path.join(config.out_dir,"src_preprocessed_line.ply"),os.path.join(config.out_dir,"ref_preprocessed_line.ply"),config,rough_trans_list[1],"icp_result_{}.txt".format("feature_based"))
    rough_overlap, rough_structural_similarity=cal_metrics_structural(os.path.join(config.out_dir,"src_preprocessed_line.ply"),os.path.join(config.out_dir,"ref_preprocessed_line.ply"),rough_trans_list[1])
    fine_overlap, fine_structural_similarity=cal_metrics_structural(os.path.join(config.out_dir,"src_preprocessed_line.ply"),os.path.join(config.out_dir,"ref_preprocessed_line.ply"),fine_trans)
    final_trans_results_list.append({'trans':rough_trans_list[1].tolist(),'overlap':rough_overlap,'structural_sim':rough_structural_similarity})
    final_trans_results_list.append({'trans':fine_trans.tolist(),'overlap':fine_overlap,'structural_sim':fine_structural_similarity})

    #init -based
    fine_trans,_,_,_=point2plane_icp_new(os.path.join(config.out_dir,"src_preprocessed_line.ply"),os.path.join(config.out_dir,"ref_preprocessed_line.ply"),config,rough_trans_list[2],"icp_result_{}.txt".format("init"))
    rough_overlap, rough_structural_similarity=cal_metrics_structural(os.path.join(config.out_dir,"src_preprocessed_line.ply"),os.path.join(config.out_dir,"ref_preprocessed_line.ply"),rough_trans_list[2])
    fine_overlap, fine_structural_similarity=cal_metrics_structural(os.path.join(config.out_dir,"src_preprocessed_line.ply"),os.path.join(config.out_dir,"ref_preprocessed_line.ply"),fine_trans)
    final_trans_results_list.append({'trans':rough_trans_list[2].tolist(),'overlap':rough_overlap,'structural_sim':rough_structural_similarity})
    final_trans_results_list.append({'trans':fine_trans.tolist(),'overlap':fine_overlap,'structural_sim':fine_structural_similarity})
    
    import json
    result_json={}
    result_json['line_rough']=final_trans_results_list[0]
    result_json['line_fine']=final_trans_results_list[1]
    result_json['feature_rough']=final_trans_results_list[2]
    result_json['feature_fine']=final_trans_results_list[3]
    result_json['init_rough']=final_trans_results_list[4]
    result_json['init_fine']=final_trans_results_list[5]
    with open(os.path.join(config.out_dir,'results.json'),'w') as outfile:
        json.dump(result_json,outfile)
    
    best_overlap=0
    best_id=0
    for id,result in enumerate(final_trans_results_list):
        trans=result['trans']
        overlap=result['overlap']
        if overlap>best_overlap:
            best_overlap=overlap
            best_id=id

    print("Finally overlap: {}".format(final_trans_results_list[best_id]['overlap']))
    print("Finally structural sim: {}".format(final_trans_results_list[best_id]['structural_sim']))
    print("Finally transformation: \n{}".format(final_trans_results_list[best_id]['trans']))
    return final_trans_results_list[best_id]['trans']

def Register_semantic(config:REG_CONFIG):
    print("Perform air-ground registration with semantics")
    #from src.preprocess import *
    from src.line_based_matching import line_based_matching_sem
    ## Line-based matching
    trans_line=line_based_matching_sem(config)

    ## STEP2: line-based and feature-based matching
    rough_trans_list=[]
    rough_trans_list.append(trans_line)
    rough_trans_list.append(config.init_trans)

    ## STEP3: icp-based refine on three rough trans, choose the best overlap one
    final_trans_results_list=[]
    #line -based
    fine_trans,_,_,_=point2plane_icp_new(os.path.join(config.out_dir,"ground_facade.ply"),os.path.join(config.out_dir,"air_building.ply"),config,rough_trans_list[0],"icp_result_{}.txt".format("line_based"))
    rough_overlap, rough_structural_similarity=cal_metrics_structural(os.path.join(config.out_dir,"ground_downsampled.ply"),os.path.join(config.out_dir,"air_downsampled.ply"),rough_trans_list[0])
    fine_overlap, fine_structural_similarity=cal_metrics_structural(os.path.join(config.out_dir,"ground_downsampled.ply"),os.path.join(config.out_dir,"air_downsampled.ply"),fine_trans)
    final_trans_results_list.append({'trans':rough_trans_list[0].tolist(),'overlap':rough_overlap,'structural_sim':rough_structural_similarity})
    final_trans_results_list.append({'trans':fine_trans.tolist(),'overlap':fine_overlap,'structural_sim':fine_structural_similarity})

    #init -based
    fine_trans,_,_,_=point2plane_icp_new(os.path.join(config.out_dir,"ground_facade.ply"),os.path.join(config.out_dir,"air_building.ply"),config,rough_trans_list[1],"icp_result_{}.txt".format("init"))
    rough_overlap, rough_structural_similarity=cal_metrics_structural(os.path.join(config.out_dir,"ground_downsampled.ply"),os.path.join(config.out_dir,"air_downsampled.ply"),rough_trans_list[1])
    fine_overlap, fine_structural_similarity=cal_metrics_structural(os.path.join(config.out_dir,"ground_downsampled.ply"),os.path.join(config.out_dir,"air_downsampled.ply"),fine_trans)
    final_trans_results_list.append({'trans':rough_trans_list[1].tolist(),'overlap':rough_overlap,'structural_sim':rough_structural_similarity})
    final_trans_results_list.append({'trans':fine_trans.tolist(),'overlap':fine_overlap,'structural_sim':fine_structural_similarity})
    
    import json
    result_json={}
    result_json['line_rough']=final_trans_results_list[0]
    result_json['line_fine']=final_trans_results_list[1]
    result_json['init_rough']=final_trans_results_list[2]
    result_json['init_fine']=final_trans_results_list[3]
    with open(os.path.join(config.out_dir,'results.json'),'w') as outfile:
        json.dump(result_json,outfile)
    
    best_overlap=0
    best_id=0
    for id,result in enumerate(final_trans_results_list):
        trans=result['trans']
        overlap=result['overlap']
        if overlap>best_overlap:
            best_overlap=overlap
            best_id=id

    print("Finally overlap: {}".format(final_trans_results_list[best_id]['overlap']))
    print("Finally structural sim: {}".format(final_trans_results_list[best_id]['structural_sim']))
    print("Finally transformation: \n{}".format(final_trans_results_list[best_id]['trans']))
    return final_trans_results_list[best_id]['trans']

def CROSS_VIEW_MATCHING(src_path:str,
                        ref_path:str,
                        out_dir:str,
                        src_sem_path:str=None,
                        ref_sem_path:str=None,
                        footprint_path:str=None,
                        gsd_ratio:float=2,
                        init_trans:np.ndarray=np.identity(4)):
    
    config=REG_CONFIG
    config.src_path=src_path
    config.ref_path=ref_path
    config.out_dir=out_dir
    config.footprint_path=footprint_path
    config.gsd_ratio=gsd_ratio
    config.ICP_SOLVE_SCALE=True
    config.init_trans=init_trans
    config.ICP_MAX_ITER=500
    config.ICP_ROBUST=True

    T=Register_semantic(config)

    return T


parser = argparse.ArgumentParser(
                    prog='CROSS_VIEW MATCHING',
                    description='CROSS_VIEW MATCHING')
parser.add_argument('-air_pc_path',help="air point cloud",type=str)  
parser.add_argument('-ground_pc_path',help="ground point cloud",type=str,required=True)   
parser.add_argument('-air_sem_path',help="air point cloud semantics, in txt file, each line corresponds to each point",type=str)  
parser.add_argument('-ground_sem_path',help="ground point cloud semantics, in txt file, each line corresponds to each point",type=str)   
parser.add_argument('-outdir',type=str,required=True)  
parser.add_argument('-gsd_ratio',type=float,default=2.0)  
parser.add_argument('-init_trans',type=float,nargs='+')  
parser.add_argument('-footprint_path',type=str)  

if __name__ == "__main__":



    args=parser.parse_args()
    air_path=args.air_pc_path
    ground_path=args.ground_pc_path
    air_sem_path=args.air_sem_path
    ground_sem_path=args.ground_sem_path
    outdir=args.outdir
    gsd_ratio=args.gsd_ratio
    footprint_path=args.footprint_path

    init_trans=args.init_trans
    if init_trans is None:
        init_trans_arr=np.identity(4)
    elif len(init_trans)!=16:
        print("Initial transofrmaiton should be 16 elements")
        exit()
    else:
        init_trans_arr=np.array([[init_trans[0],init_trans[1],init_trans[2],init_trans[3]],
                                [init_trans[4],init_trans[5],init_trans[6],init_trans[7]],
                                [init_trans[8],init_trans[9],init_trans[10],init_trans[11]],
                                [init_trans[12],init_trans[13],init_trans[14],init_trans[15]]])


    CROSS_VIEW_MATCHING(ground_path,air_path,outdir,ground_sem_path,air_sem_path,gsd_ratio,footprint_path,init_trans_arr)

    # CROSS_VIEW_MATCHING(r'J:\xuningli\wriva\data\vary\labeled_ptcld\t04_v01_s02_r03_ground\ascii_with_labels.ply',
    #                     r'J:\xuningli\wriva\data\vary\labeled_ptcld\t04_v01_s02_r03_air\ascii_with_labels.ply',
    #                     r'J:\xuningli\wriva\data\vary\labeled_ptcld\t04_v01_s02_r03_out1',
    #                     r'J:\xuningli\wriva\data\vary\labeled_ptcld\t04_v01_s02_r03_ground\ascii_with_labels_semantic.txt',
    #                     r'J:\xuningli\wriva\data\vary\labeled_ptcld\t04_v01_s02_r03_air\ascii_with_labels_semantic.txt',)

    # CROSS_VIEW_MATCHING(r'/research/GDA/xuningli/wriva/data/cross_view/pair6/ground1.ply',
    #                     r'/research/GDA/xuningli/wriva/data/cross_view/pair6/drone_down.ply',
    #                     r'/research/GDA/xuningli/wriva/data/cross_view/pair6/out1/')

    # CROSS_VIEW_MATCHING(r'/research/GDA/xuningli/wriva/data/cross_view/pair1/ground.ply',
    #                     r'/research/GDA/xuningli/wriva/data/cross_view/pair1/drone.ply',
    #                     r'/research/GDA/xuningli/wriva/data/cross_view/pair1/out4/')

    # CROSS_VIEW_MATCHING(r'/research/GDA/xuningli/wriva/data/cross_view/pair2/ground.ply',
    #                     r'/research/GDA/xuningli/wriva/data/cross_view/pair2/drone.ply',
    #                     r'/research/GDA/xuningli/wriva/data/cross_view/pair2/out4')

        
        













