import os
import glob
import numpy as np
import sys
sys.path.append(os.getcwd())
import cv2
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.config import REG_CONFIG
from scipy.spatial import cKDTree
from src.preprocess import *
from scipy import stats
from src.mask2polygon import run_fit_main
import copy
import math
from src.lsd_deep import predict_lsd
from src.ply import read_ply
from src.line_based_matching import line_based_matching_sem, line_based_matching_sem_vis
from src.icp import point2plane_icp_new


def single_test(src,ref,outdir,init_trans,gt_trans,sem_label='coco',refine=False):
    ground_path=src
    drone_path=ref
    config=REG_CONFIG()
    config.use_sem=False
    config.sem_label_type=sem_label
    config.ref_path=drone_path
    config.src_path=ground_path
    config.out_dir=outdir
    config.init_trans=init_trans
    T=line_based_matching_sem(config)
    print(T)
    if refine:
        config.ICP_SOLVE_SCALE=True
        config.ICP_MAX_ITER=100
        config.ICP_ROBUST=True
        config.out_dir=r'E:\tmp'
        T,_,_,_=point2plane_icp_new(ground_path,drone_path,config,T)

    np.savetxt(os.path.join(outdir,'reg.txt'),T)
    # src_pc=o3d.io.read_point_cloud(src)
    # src_pc1=copy.deepcopy(src_pc)
    # src_reg=src_pc.transform(T)
    # src_gt=src_pc1.transform(gt_trans)
    # src_reg=np.array(src_reg.points)
    # src_gt=np.array(src_gt.points)
    # err=np.mean(np.linalg.norm(src_reg-src_gt,axis=1))
    return 

def single_vis(src,ref,outdir,init_trans,gt_trans,sem_label='coco'):
    ground_path=src
    drone_path=ref
    config=REG_CONFIG()
    config.use_sem=False
    config.sem_label_type=sem_label
    config.ref_path=drone_path
    config.src_path=ground_path
    config.out_dir=outdir
    config.init_trans=init_trans
    line_based_matching_sem_vis(config)
    return 

def batch_test(src,ref,outdir):
    gt_scales=[0.5,0.75,1,1.5,2]
    results_err=[]
    for id,scale in enumerate(gt_scales):
        init_trans=np.identity(4)
        init_scale=1/scale
        init_trans[:3,:3]=init_trans[:3,:3]*init_scale
        gt=np.identity(4)
        err=single_test(src,ref,os.path.join(outdir,'{}'.format(scale)),init_trans,gt)
        results_err.append(err)
    for id,scale in enumerate(gt_scales):
        print("gt scale: {}, registration error: {}".format(gt_scales[id],results_err[id]))

def dataset_enumerate():
    DATA_DIR=r'J:\xuningli\papers\g2a\data\results\data'
    isprs_center_uav=os.path.join(DATA_DIR,'isprs_center_uav.ply')
    isprs_zeche_uav=os.path.join(DATA_DIR,'isprs_zeche_uav.ply')
    apl_uav=os.path.join(DATA_DIR,'apl_uav.ply')

    grounds=glob.glob(os.path.join(DATA_DIR,"*.ply"))
    grounds=[file for file in grounds if 'uav' not in file]

    paired_list=[]
    for ground in grounds:
        if 'center' in ground:
            paired_list.append([isprs_center_uav,ground])
        elif 'zeche' in ground:
            paired_list.append([isprs_zeche_uav,ground])
        else:
            paired_list.append([apl_uav,ground])
    return paired_list

def dataset_scale_enumerate():
    import glob
    DATA_DIR=r'J:\xuningli\papers\g2a\data\results\data'
    #DATA_DIR=r'/local/storage/ssd_a/xu.3961_local/results/data'
    isprs_center_uav=os.path.join(DATA_DIR,'isprs_center_uav.ply')
    isprs_zeche_uav=os.path.join(DATA_DIR,'isprs_zeche_uav.ply')
    apl_uav=os.path.join(DATA_DIR,'apl_uav.ply')

    grounds=glob.glob(os.path.join(DATA_DIR,"*.ply"))
    grounds=[file for file in grounds if 'uav' not in file]

    paired_list=[]
    for ground in grounds:
        if 'center' in ground:
            paired_list.append([isprs_center_uav,ground])
        elif 'zeche' in ground:
            paired_list.append([isprs_zeche_uav,ground])
        else:
            paired_list.append([apl_uav,ground])
    res=[]
    for pair in paired_list:
        ground=pair[1]
        if "_1.ply" in ground or "_2.ply" in ground or "_3.ply" in ground or "_4.ply" in ground or "_5.ply" in ground:
            continue
        res.append(pair)
    return res

def accuracy_batch(outdir):
    paired_list=dataset_enumerate()
    init_scale=1.5
    gt=np.identity(4)
    init_trans=np.identity(4)
    rot=R.from_euler('xyz',[90,0,90],degrees=True).as_matrix()
    init_trans[:3,:3]=rot*init_scale
    init_trans[2,3]=5
    print(init_trans)
    results_err=[]
    for pair in paired_list:
        uav=pair[0]
        ground=pair[1]
        sem_label='coco'
        uav_name=os.path.basename(uav)
        ground_name=os.path.basename(ground)
        outname="{}_{}".format(uav_name,ground_name)
        out_dir_cur=os.path.join(outdir,outname)
        os.makedirs(out_dir_cur,exist_ok=True)
        err=single_test(ground,uav,out_dir_cur,init_trans,gt,sem_label)
        results_err.append(err)
        result_trans=np.loadtxt(os.path.join(out_dir_cur,"reg.txt"))
        np.savetxt(os.path.join(outdir,"{}_{}_our.txt".format(ground_name,uav_name)),result_trans)
        

    for id,err in enumerate(results_err):
        uav_name=os.path.basename(paired_list[id][0])
        ground_name=os.path.basename(paired_list[id][1])
        print("uav: {}, ground: {}, registration error: {}".format(uav_name,ground_name,err))

def scale_batch(outdir):
    paired_list=dataset_scale_enumerate()
    #scale_list=[1,1.25,1.5,1.75,2,3,5,10]
    scale_list=[2,3,5,7.5,10,15]
    os.makedirs(outdir,exist_ok=True)
    for pair in paired_list:
        for scale in scale_list:
            gt=np.identity(4)
            init_trans=np.identity(4)
            rot=R.from_euler('xyz',[90,0,90],degrees=True).as_matrix()
            init_trans[:3,:3]=rot*scale
            init_trans[2,3]=5
            #gt=np.linalg.inv(init_trans)

            uav=pair[0]
            ground=pair[1]
            sem_label='coco'
            uav_name=os.path.basename(uav)
            ground_name=os.path.basename(ground)
            outname="{}_{}_{}".format(uav_name,ground_name,scale)
            out_file=os.path.join(outdir,"{}_{}_{:.2f}_our.txt".format(ground_name,uav_name,scale))

            if os.path.exists(out_file):
                continue
            out_dir_cur=os.path.join(outdir,outname)
            os.makedirs(out_dir_cur,exist_ok=True)
            single_test(ground,uav,out_dir_cur,init_trans,gt,sem_label)
            result_trans=np.loadtxt(os.path.join(out_dir_cur,"reg.txt"))
            np.savetxt(out_file,result_trans)

def scale_print():
    paired_list=dataset_scale_enumerate()
    scale_list=[1,1.25,1.5,1.75,2,3,5,10]
    #scale_list=[2,3,5,7.5,10,15]

    for scale in scale_list:
        gt=np.identity(4)
        init_trans=np.identity(4)
        rot=R.from_euler('xyz',[90,0,90],degrees=True).as_matrix()
        init_trans[:3,:3]=rot*scale
        init_trans[2,3]=5
        print(init_trans)

def vis_batch(outdir):
    paired_list=dataset_scale_enumerate()
    gt=np.identity(4)
    init_trans=np.identity(4)
    results_err=[]
    for pair in paired_list:
        if 'apl' not in pair[0]:
            continue
        uav=pair[0]
        ground=pair[1]
        sem_label='coco'
        uav_name=os.path.basename(uav)
        ground_name=os.path.basename(ground)
        outname="{}_{}".format(uav_name,ground_name)
        out_dir_cur=os.path.join(outdir,outname)
        os.makedirs(out_dir_cur,exist_ok=True)
        err=single_vis(ground,uav,out_dir_cur,init_trans,gt,sem_label)
        
def batch_refine(in_dir,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    paired_list=dataset_enumerate()
    gt=np.identity(4)
    init_trans=np.identity(4)
    rot=R.from_euler('xyz',[90,0,90],degrees=True).as_matrix()
    init_trans[:3,:3]=rot*2
    init_trans[2,3]=5
    print(init_trans)
    for pair in paired_list:
        uav=pair[0]
        ground=pair[1]

        sem_label='coco'
        uav_name=os.path.basename(uav)
        ground_name=os.path.basename(ground)

        init_txt=glob.glob(os.path.join(in_dir,"{}_{}*.txt".format(ground_name,uav_name)))[0]
        init_trans=np.loadtxt(init_txt)
        out_txt=os.path.join(out_dir,"{}_{}.txt".format(ground_name,uav_name))
        if os.path.exists(out_txt):
            continue
        if "mp5_2" in ground:
            np.savetxt(out_txt,init_trans)
            
        config=REG_CONFIG()
        config.use_sem=False
        config.sem_label_type=sem_label
        config.ref_path=uav
        config.src_path=ground
        config.out_dir=out_dir
        config.init_trans=init_trans
        config.ICP_SOLVE_SCALE=False
        config.ICP_MAX_ITER=100
        config.ICP_ROBUST=True
        config.out_dir=r'E:\tmp'
        T_fine,_,_,_=point2plane_icp_new(ground,uav,config,init_trans)
        
        np.savetxt(out_txt,T_fine)
        



if __name__=='__main__':
    # init_trans=np.identity(4)
    # rot=R.from_euler('xyz',[90,0,90],degrees=True).as_matrix()
    # init_trans[:3,:3]=rot*2
    # init_trans[2,3]=5
    # single_test(r'E:\tmp\cross\isprs_center_obelisk_1_tmp.ply',r'J:\xuningli\papers\g2a\data\results\data\isprs_center_uav.ply',r'E:\data\isprs_a2g_1\isprs_center_uav.ply_isprs_center_obelisk_1.ply',init_trans,np.identity(4))
    accuracy_batch(r'E:\data\isprs_a2g_2')
    #vis_batch(r'E:\data\isprs_vis')
    #scale_batch(r'E:\data\isprs_a2g_scale_new1')
    #scale_print()
    # batch_refine(r'J:\xuningli\papers\g2a\data\results\acc_result\our',r'J:\xuningli\papers\g2a\data\results\acc_result\our_refine')
    # batch_refine(r'J:\xuningli\papers\g2a\data\results\acc_result\yoho_opransac',r'J:\xuningli\papers\g2a\data\results\acc_result\yoho_opransac_refine')
    # batch_refine(r'J:\xuningli\papers\g2a\data\results\acc_result\yoho_star_opransac',r'J:\xuningli\papers\g2a\data\results\acc_result\yoho_star_opransac_refine')
    # batch_refine(r'J:\xuningli\papers\g2a\data\results\acc_result\yoho_star_teaser',r'J:\xuningli\papers\g2a\data\results\acc_result\yoho_star_teaser_refine')
    # batch_refine(r'J:\xuningli\papers\g2a\data\results\acc_result\yoho_teaser',r'J:\xuningli\papers\g2a\data\results\acc_result\yoho_teaser_refine')
