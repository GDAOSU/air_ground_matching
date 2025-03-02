import os
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

coco_ground_idx=[90,98,100,123,125]
coco_building_idx=[86,91,110,111,112,114,115,129,131]
apl_ground_idx=[1]
apl_building_idx=[2]

def read_ply_to_o3d(src_path):
    cloud=read_ply(src_path)
    cloud = np.vstack((cloud['x'], cloud['y'], cloud['z'],cloud['nx'], cloud['ny'], cloud['nz'],cloud['scalar_label'])).T
    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:,:3])
    pcd.normals=o3d.utility.Vector3dVector(cloud[:,3:6])
    semantic=cloud[:,6]
    return pcd,semantic

class GeoData:
    def __init__(self,config:REG_CONFIG,data_type:str) -> None:
        self.type=data_type
        self.config=config
        if config.sem_label_type=='coco':
            self.ground_indice=coco_ground_idx
            self.building_indice=coco_building_idx
        elif config.sem_label_type=='apl':
            self.ground_indice=apl_ground_idx
            self.building_indice=apl_building_idx
        
        if data_type=='ground':
            ## STEP0: preprocess, 1. downsample if too big, 2. remove outlier
            self.PC,semantic=read_ply_to_o3d(self.config.src_path)
            self.SEMANTIC=semantic.astype(np.int32)
            normal1=np.array(self.PC.normals)[0,:]
            self.PC=self.PC.transform(self.config.init_trans)
            normals=np.array(self.PC.normals)
            normals=normals/np.expand_dims(np.linalg.norm(normals,axis=1),axis=1)
            self.PC.normals=o3d.utility.Vector3dVector(normals)
            
            #o3d.io.write_point_cloud(os.path.join(self.config.out_dir,'ground.ply'),self.PC)
            normal2=np.array(self.PC.normals)[0,:]
            # self.PC=o3d.io.read_point_cloud(self.config.src_path)
            # self.PC=self.PC.transform(self.config.init_trans)
            # self.SEMANTIC=np.loadtxt(self.config.src_sem_path).astype(np.int32)
            # num_pts=np.array(self.PC.points).shape[0]
            # inliers=np.zeros(num_pts)
            # if num_pts>config.max_pts:
            #     sample_every=math.ceil(num_pts/self.config.max_pts)
            #     select_arr=np.arange(0,num_pts,sample_every)
            #     inliers[select_arr]=True
            # else:
            #     select_arr=np.arange(0,num_pts)
            #     inliers[select_arr]=True
            # _, inliers_ind1 = self.PC.remove_statistical_outlier(nb_neighbors=5,std_ratio=2)
            # inliers1=np.zeros(num_pts)
            # inliers1[inliers_ind1]=True
            # inliers=np.logical_and(inliers,inliers1)
            # self.PC=self.PC.select_by_index(np.where(inliers==True)[0])
            # self.SEMANTIC=self.SEMANTIC[inliers]
            #o3d.io.write_point_cloud(os.path.join(self.config.out_dir,'ground_downsampled.ply'),self.PC)

            ## STEP1: extract building or facade data
            self.BUILDING_PC,self.BUILDING_HEIGHT,self.GROUND_NORMAL=extract_building_part_zaxis(self.PC,self.SEMANTIC,self.ground_indice,self.building_indice)
            self.FACADE_PC,_=extract_facade_part(self.BUILDING_PC,self.config,self.GROUND_NORMAL)
            # o3d.io.write_point_cloud(os.path.join(self.config.out_dir,'ground_building.ply'),self.BUILDING_PC)
            # o3d.io.write_point_cloud(os.path.join(self.config.out_dir,'ground_facade.ply'),self.FACADE_PC)
        elif data_type=='air':
            ## STEP0: preprocess, 1. downsample if too big, 2. remove outlier
            self.PC,semantic=read_ply_to_o3d(self.config.ref_path)
            self.SEMANTIC=semantic.astype(np.int32)
            # self.PC=o3d.io.read_point_cloud(self.config.ref_path)
            # self.SEMANTIC=np.loadtxt(self.config.ref_sem_path).astype(np.int32)
            num_ref_pts=np.array(self.PC.points).shape[0]
            ref_inliers=np.zeros(num_ref_pts)
            if num_ref_pts>self.config.max_pts:
                sample_every=math.ceil(num_ref_pts/self.config.max_pts)
                select_arr=np.arange(0,num_ref_pts,sample_every)
                ref_inliers[select_arr]=True
            else:
                select_arr=np.arange(0,num_ref_pts)
                ref_inliers[select_arr]=True
            _, ref_inliers_ind1 = self.PC.remove_statistical_outlier(nb_neighbors=5,std_ratio=1)
            ref_inliers1=np.zeros(num_ref_pts)
            ref_inliers1[ref_inliers_ind1]=True
            ref_inliers=np.logical_and(ref_inliers,ref_inliers1)
            self.PC=self.PC.select_by_index(np.where(ref_inliers==True)[0])
            self.SEMANTIC=self.SEMANTIC[ref_inliers]

            ## STEP1: extract building or facade data
            self.BUILDING_PC,self.BUILDING_HEIGHT,self.GROUND_NORMAL=extract_building_part_zaxis(self.PC,self.SEMANTIC,self.ground_indice,self.building_indice)
            ref_x_len,ref_y_len=get_bound_along_plane(self.PC)
            self.FACADE_PC,_=extract_facade_part(self.BUILDING_PC,self.config,self.GROUND_NORMAL)
            #o3d.io.write_point_cloud(os.path.join(self.config.out_dir,'air_building.ply'),self.BUILDING_PC)
            #o3d.io.write_point_cloud(os.path.join(self.config.out_dir,'air_facade.ply'),self.FACADE_PC)

            # o3d.io.write_point_cloud(os.path.join(self.config.out_dir,'air_downsampled.ply'),self.PC)
        elif data_type=='footprint':
            self.reso=self.config.footprint_gsd
            self.rot_plane=np.identity(3)
            self.z_height=0
            self.scale_init=1

    def plane_rot(self,ref_data=None)->None:

        if self.type=='ground':           
            # STEP2: start line-based matching
            print("#### Start Line-based Matching ####")
            z_axis=np.array([0,0,1])
            src_facade_pts=np.array(self.FACADE_PC.points)
            src_facade_normals=np.array(self.FACADE_PC.normals)
            src_z_axis=self.GROUND_NORMAL

            # rotate src to z_axis, scale to ref
            rot_axis=np.cross(src_z_axis,z_axis)
            rot_axis_norm=np.linalg.norm(rot_axis)
            if rot_axis_norm==0:
                rot_axis_norm=1
            rot_axis=rot_axis/rot_axis_norm
            angle=np.arccos(src_z_axis@z_axis)
            src_rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()
            src_facade_pts_z_corrected=(src_rot_plane@src_facade_pts.T).T
            src_facade_normals_z_corrected=(src_rot_plane@src_facade_normals.T).T
            self.rot_plane=src_rot_plane

            scale_init=1
            # if ref_data is not None:
            #     scale_init= ref_data.BUILDING_HEIGHT/self.BUILDING_HEIGHT
            src_facade_pts_z_corrected=src_facade_pts_z_corrected*scale_init

            ground_idx=[]
            for id in self.ground_indice:
                ground_idx=ground_idx+list(np.where(self.SEMANTIC==id)[0])
            ground_idx=np.array(ground_idx)
            src_ground_pts=np.array(self.PC.points)[ground_idx]
            src_ground_pts_z_corrected=(src_rot_plane@src_ground_pts.T).T*scale_init
            src_pts_z_corrected=(src_rot_plane@np.array(self.PC.points).T).T*scale_init
            src_z_height=np.quantile(src_ground_pts_z_corrected[:,2],0.2)
            self.z_height=src_z_height
            self.scale_init=scale_init

            src_bound_min=np.min(src_facade_pts_z_corrected[:,:3],axis=0)
            src_bound_max=np.max(src_facade_pts_z_corrected[:,:3],axis=0)
            src_x_len=src_bound_max[0]-src_bound_min[0]
            src_y_len=src_bound_max[1]-src_bound_min[1]
            max_len=max(src_x_len,src_y_len)
            reso=max_len/self.config.ground_max_pixel_length
            self.reso=max(self.config.ground_gsd,reso)
            if ref_data is not None:
                self.reso=ref_data.reso

            self.img,self.pixels,self.pixels_normals,self.bound_min,self.bound_max=plot_boundary(src_facade_pts_z_corrected,src_facade_normals_z_corrected,self.reso)
            cv2.imwrite(os.path.join(self.config.out_dir,"ground_facade_img.png"),self.img)
            # src_sem_img,src_sem_color_sem=plot_boundary_sem(src_pts_z_corrected,self.SEMANTIC,self.reso,src_bound_min,src_bound_max)
            # cv2.imwrite(os.path.join(self.config.out_dir,"ground_sem_img.png"),src_sem_img)
            # cv2.imwrite(os.path.join(self.config.out_dir,"ground_sem_color_img.png"),src_sem_color_sem)
        elif self.type=='air':
            # rotate ref to z_axis & generate img
            ref_building_pts=np.array(self.BUILDING_PC.points)
            ref_building_normals=np.array(self.BUILDING_PC.normals)
            ref_facade_pts=np.array(self.FACADE_PC.points)
            ref_facade_normals=np.array(self.FACADE_PC.normals)
            ref_z_axis=self.GROUND_NORMAL
            z_axis=np.array([0,0,1])
            rot_axis=np.cross(ref_z_axis,z_axis)
            rot_axis=rot_axis/np.linalg.norm(rot_axis)
            angle=np.arccos(ref_z_axis@z_axis)
            ref_rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()
            ref_building_pts_z_corrected=(ref_rot_plane@ref_building_pts.T).T
            ref_building_normals_z_corrected=(ref_rot_plane@ref_building_normals.T).T
            ref_facade_pts_z_corrected=(ref_rot_plane@ref_facade_pts.T).T
            ref_facade_normals_z_corrected=(ref_rot_plane@ref_facade_normals.T).T

            ground_idx=[]
            for id in self.ground_indice:
                ground_idx=ground_idx+list(np.where(self.SEMANTIC==id)[0])
            ground_idx=np.array(ground_idx)
            ref_ground_pts=np.array(self.PC.points)[ground_idx]
            ref_ground_pts_z_corrected=(ref_rot_plane@ref_ground_pts.T).T
            ref_z_height=np.quantile(ref_ground_pts_z_corrected[:,2],0.2)
            ref_pts_z_corrected=(ref_rot_plane@np.array(self.PC.points).T).T
            self.rot_plane=ref_rot_plane
            self.z_height=ref_z_height

            ##determine the resolution
            ref_bound_min=np.min(ref_building_pts_z_corrected[:,:3],axis=0)
            ref_bound_max=np.max(ref_building_pts_z_corrected[:,:3],axis=0)
            ref_x_len=ref_bound_max[0]-ref_bound_min[0]
            ref_y_len=ref_bound_max[1]-ref_bound_min[1]
            max_len=max(ref_x_len,ref_y_len)
            self.reso=max(self.config.air_gsd,max_len/self.config.air_max_pixel_length)

            self.img,self.pixels,self.pixels_normals,self.bound_min,self.bound_max=plot_boundary(ref_building_pts_z_corrected,ref_building_normals_z_corrected,self.reso)
            self.facade_img,self.facade_pixels,self.facade_pixels_normals,self.facade_bound_min,self.facade_bound_max=plot_boundary(ref_facade_pts_z_corrected,ref_facade_normals_z_corrected,self.reso,self.bound_min,self.bound_max)
            cv2.imwrite(os.path.join(self.config.out_dir,"air_building_img.png"),self.img)
            cv2.imwrite(os.path.join(self.config.out_dir,"air_facade_img.png"),self.facade_img)

            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            # self.img = cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
            # cv2.imwrite(os.path.join(self.config.out_dir,'air_building_img_denoise.png'),self.img)
            # self.facade_img = cv2.morphologyEx(self.facade_img, cv2.MORPH_OPEN, kernel)
            # cv2.imwrite(os.path.join(self.config.out_dir,'air_facade_img_denoise.png'),self.facade_img)
            # ref_sem_img,ref_sem_color_img=plot_boundary_sem(ref_pts_z_corrected,self.SEMANTIC,self.reso,ref_bound_min,ref_bound_max)
            # cv2.imwrite(os.path.join(self.config.out_dir,"air_sem_img.png"),ref_sem_img)
            # cv2.imwrite(os.path.join(self.config.out_dir,"air_sem_color_img.png"),ref_sem_color_img)

    def line_extraction(self)->None:
        if self.type=='ground':
            lsd = cv2.createLineSegmentDetector(1)
            #lines_raw = lsd.detect(self.img)[0]
            lines_raw=predict_lsd(self.img)
            lines=lines_raw
            #lines=merge_line_seg(lines_raw)
            src_lines=lines.copy()
            drawn_img_raw = lsd.drawSegments(self.img,lines_raw)
            drawn_img = lsd.drawSegments(self.img,lines)
            for i in range(lines_raw.shape[0]):
                line=lines_raw[i,0,:]
                pt1=line[:2].astype(np.int32)
                pt2=line[2:].astype(np.int32)
                cv2.circle(drawn_img_raw,pt1,1,(255,0,0),-1)
                cv2.circle(drawn_img_raw,pt2,1,(255,0,0),-1)
            for i in range(lines.shape[0]):
                line=lines[i,0,:]
                pt1=line[:2].astype(np.int32)
                pt2=line[2:].astype(np.int32)
                cv2.circle(drawn_img,pt1,1,(255,0,0),-1)
                cv2.circle(drawn_img,pt2,1,(255,0,0),-1)
            cv2.imwrite(os.path.join(self.config.out_dir,'ground_line_refined.png'),drawn_img)
            cv2.imwrite(os.path.join(self.config.out_dir,'ground_line_raw.png'),drawn_img_raw)
            
            #vis 3D lines in input coordinate system
            lines3d=[]
            line_len=5
            line_margin=3
            # line_len=0.75
            # line_margin=0.3
            for i in range(lines.shape[0]):
                line=lines[i,0,:]
                pt1=line[:2].astype(np.int32)
                pt2=line[2:].astype(np.int32)
                line1_x=self.bound_min[0]+self.reso*pt1[0]
                line1_y=self.bound_max[1]-self.reso*pt1[1]
                line1_z_top=self.z_height+line_len*3
                line1_z_mid=line1_z_top-line_margin
                line1_z_bot=self.z_height-line_len
                line1_pt1=np.array([line1_x,line1_y,line1_z_top])
                line1_pt2=np.array([line1_x,line1_y,line1_z_mid])
                line1_pt3=np.array([line1_x,line1_y,line1_z_bot])
                line1_pt1=np.linalg.inv(self.rot_plane)@line1_pt1/self.scale_init
                line1_pt2=np.linalg.inv(self.rot_plane)@line1_pt2/self.scale_init
                line1_pt3=np.linalg.inv(self.rot_plane)@line1_pt3/self.scale_init
                lines3d.append([line1_pt1[0],line1_pt1[1],line1_pt1[2],line1_pt2[0],line1_pt2[1],line1_pt2[2],line1_pt3[0],line1_pt3[1],line1_pt3[2]])

                line2_x=self.bound_min[0]+self.reso*pt2[0]
                line2_y=self.bound_max[1]-self.reso*pt2[1]
                line2_z_top=self.z_height+line_len*3
                line2_z_mid=line2_z_top-line_margin
                line2_z_bot=self.z_height-line_len
                line2_pt1=np.array([line2_x,line2_y,line2_z_top])
                line2_pt2=np.array([line2_x,line2_y,line2_z_mid])
                line2_pt3=np.array([line2_x,line2_y,line2_z_bot])
                line2_pt1=np.linalg.inv(self.rot_plane)@line2_pt1/self.scale_init
                line2_pt2=np.linalg.inv(self.rot_plane)@line2_pt2/self.scale_init
                line2_pt3=np.linalg.inv(self.rot_plane)@line2_pt3/self.scale_init
                lines3d.append([line2_pt1[0],line2_pt1[1],line2_pt1[2],line2_pt2[0],line2_pt2[1],line2_pt2[2],line2_pt3[0],line2_pt3[1],line2_pt3[2]])
            write_line3d(lines3d,os.path.join(self.config.out_dir,'ground_line3d.obj'))

            mat=np.zeros(self.img.shape,dtype=self.img.dtype)
            src_drawn_img = lsd.drawSegments(mat,src_lines)
            for i in range(src_lines.shape[0]):
                line=src_lines[i,0,:]
                pt1=line[:2].astype(np.int32)
                pt2=line[2:].astype(np.int32)
                cv2.circle(src_drawn_img,pt1,1,(255,0,0),-1)
                cv2.circle(src_drawn_img,pt2,1,(255,0,0),-1)
                cv2.line(src_drawn_img,pt1,pt2,(0,255,0),3,-1)
            cv2.imwrite(os.path.join(self.config.out_dir,'ground_lines_only.png'),src_drawn_img)
            self.lines=src_lines
            self.lines_normals,self.lines_pixels_set,self.lines_pixels_normals,self.lines_pixels_weights=get_line_pixel_info(self.lines,self.pixels,self.pixels_normals,tag='src')
        elif self.type=='air':
            lsd = cv2.createLineSegmentDetector(1)
            #lines_raw = lsd.detect(self.facade_img)[0]
            lines_raw=predict_lsd(self.facade_img)
            lines=lines_raw
            #lines=run_fit_main(self.img,self.config.out_dir,90)
            ref_lines=lines.copy()
            drawn_img = lsd.drawSegments(self.facade_img,lines)
            for i in range(lines.shape[0]):
                line=lines[i,0,:]
                pt1=line[:2].astype(np.int32)
                pt2=line[2:].astype(np.int32)
                cv2.circle(drawn_img,pt1,1,(255,0,0),-1)
                cv2.circle(drawn_img,pt2,1,(255,0,0),-1)
            cv2.imwrite(os.path.join(self.config.out_dir,'air_line.png'),drawn_img)
            mat=np.zeros(self.img.shape,dtype=self.img.dtype)

            #vis 3D lines in input coordinate system
            lines3d=[]
            line_len=5
            line_margin=3
            # line_len=0.75
            # line_margin=0.3
            for i in range(lines.shape[0]):
                line=lines[i,0,:]
                pt1=line[:2].astype(np.int32)
                pt2=line[2:].astype(np.int32)
                line1_x=self.bound_min[0]+self.reso*pt1[0]
                line1_y=self.bound_max[1]-self.reso*pt1[1]
                line1_z_top=self.z_height+line_len*3
                line1_z_mid=line1_z_top-line_margin
                line1_z_bot=self.z_height-line_len
                line1_pt1=np.array([line1_x,line1_y,line1_z_top])
                line1_pt2=np.array([line1_x,line1_y,line1_z_mid])
                line1_pt3=np.array([line1_x,line1_y,line1_z_bot])
                line1_pt1=np.linalg.inv(self.rot_plane)@line1_pt1
                line1_pt2=np.linalg.inv(self.rot_plane)@line1_pt2
                line1_pt3=np.linalg.inv(self.rot_plane)@line1_pt3
                lines3d.append([line1_pt1[0],line1_pt1[1],line1_pt1[2],line1_pt2[0],line1_pt2[1],line1_pt2[2],line1_pt3[0],line1_pt3[1],line1_pt3[2]])

                line2_x=self.bound_min[0]+self.reso*pt2[0]
                line2_y=self.bound_max[1]-self.reso*pt2[1]
                line2_z_top=self.z_height+line_len*3
                line2_z_mid=line2_z_top-line_margin
                line2_z_bot=self.z_height-line_len
                line2_pt1=np.array([line2_x,line2_y,line2_z_top])
                line2_pt2=np.array([line2_x,line2_y,line2_z_mid])
                line2_pt3=np.array([line2_x,line2_y,line2_z_bot])
                line2_pt1=np.linalg.inv(self.rot_plane)@line2_pt1
                line2_pt2=np.linalg.inv(self.rot_plane)@line2_pt2
                line2_pt3=np.linalg.inv(self.rot_plane)@line2_pt3
                lines3d.append([line2_pt1[0],line2_pt1[1],line2_pt1[2],line2_pt2[0],line2_pt2[1],line2_pt2[2],line2_pt3[0],line2_pt3[1],line2_pt3[2]])
            write_line3d(lines3d,os.path.join(self.config.out_dir,'air_line3d.obj'))

            self.lines=ref_lines
            self.lines_normals,self.lines_pixels_set,self.lines_pixels_normals,self.lines_pixels_weights=get_line_pixel_info(self.lines,self.facade_pixels,self.facade_pixels_normals,tag='ref')
        elif self.type=='footprint':
            #self.polygons_utm=parse_kml(self.config.footprint_path)
            self.polygons_utm=parse_kml_noutm(self.config.footprint_path)
            self.img,self.lines_pixels_set,self.lines_pixels_normals,self.lines,self.lines_normals,self.lines_pixels_weights,self.bound_min,self.bound_max=plot_boundary_footprint(self.polygons_utm,self.reso)
            cv2.imwrite(os.path.join(self.config.out_dir,"footprint_img.png"),self.img)

class Line_Matching:
    def __init__(self,ground_data:GeoData,air_data:GeoData,footprint_data:GeoData,config:GeoData) -> None:
        self.ground=ground_data
        self.air=air_data
        self.footprint=footprint_data
        self.config=config

    def align_g2a(self):
        src_line_pixels=self.ground.lines_pixels_set.astype(np.int32)
        ref_line_pixels=self.air.lines_pixels_set.astype(np.int32)

        src_line_img=np.zeros((self.ground.img.shape[0],self.ground.img.shape[1]))
        ref_line_img=np.zeros((self.air.img.shape[0],self.air.img.shape[1]))
        for i in range(src_line_pixels.shape[0]):
            src_line_img[src_line_pixels[i][1],src_line_pixels[i][0]]=1
        for i in range(ref_line_pixels.shape[0]):
            ref_line_img[ref_line_pixels[i][1],ref_line_pixels[i][0]]=1
        src_line_img*=255
        ref_line_img*=255
        for i in range(self.ground.lines.shape[0]):
            pt1=self.ground.lines[i][0][:2]
            pt2=self.ground.lines[i][0][2:]
            mid_pt=(pt1+pt2)/2
            normal_end_pt=mid_pt+self.ground.lines_normals[i]*5
            cv2.line(src_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
        for i in range(self.air.lines.shape[0]):
            pt1=self.air.lines[i][0][:2]
            pt2=self.air.lines[i][0][2:]
            mid_pt=(pt1+pt2)/2
            normal_end_pt=mid_pt+self.air.lines_normals[i]*5
            cv2.line(ref_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
        cv2.imwrite(os.path.join(self.config.out_dir,"ground_line_pixel.png"),src_line_img)
        cv2.imwrite(os.path.join(self.config.out_dir,"air_line_pixel.png"),ref_line_img)

        ref_tree=KDTree(ref_line_pixels)

        src_end_pts=[]
        src_end_pts_normals=[]
        for i in range(self.ground.lines.shape[0]):
            src_end_pts.append(self.ground.lines[i,0,:2])
            src_end_pts.append(self.ground.lines[i,0,2:])
            src_end_pts_normals.append(self.ground.lines_normals[i])
            src_end_pts_normals.append(self.ground.lines_normals[i])
        src_end_pts=np.array(src_end_pts)
        num_src_pts=src_end_pts.shape[0]
        num_ref_lines=self.air.lines.shape[0]
        best_sim=-999
        best_src_line=None
        best_ref_line=None
        best_T=np.identity(4)
        best_T_list=[]
        final_score_list=[]
        corr_normal_consis_list=[]
        normal_consis_list=[]
        dis_list=[]
        
        best_src_line_list=[]
        best_ref_line_list=[]

        #print("Total # line seg matches: {}".format(num_src_lines*num_ref_lines))
        ref_line_len_list=[]
        for j in range(num_ref_lines):
            ref_line_len=np.linalg.norm(self.air.lines[j,0,:2]-self.air.lines[j,0,2:])
            ref_line_len_list.append(ref_line_len)
        ref_line_len_arr=np.array(ref_line_len_list)
        print("Ref line: max: {}, min: {}, mean: {}, 90%: {}, 70%:{}, 50%:{},30%:{}".format(np.max(ref_line_len_arr),np.min(ref_line_len_arr),np.mean(ref_line_len_arr),
                                                                                            np.quantile(ref_line_len_arr,0.9),
                                                                                            np.quantile(ref_line_len_arr,0.7),
                                                                                            np.quantile(ref_line_len_arr,0.5),
                                                                                            np.quantile(ref_line_len_arr,0.3)
                                                                                            ))
        cnc_weight=0.1
        dis_weight=0.5
        nc_weight=0.4
        for j in range(num_ref_lines):
            for ii1 in range(num_src_pts):
                for ii2 in range(ii1,num_src_pts):
                    pt1=src_end_pts[ii1]
                    pt2=src_end_pts[ii2]
                    pt1_normal=src_end_pts_normals[ii1]
                    pt2_normal=src_end_pts_normals[ii2]
                    src_line_len=np.linalg.norm(pt1-pt2)
                    ref_line_len=np.linalg.norm(self.air.lines[j,0,:2]-self.air.lines[j,0,2:])
                    if src_line_len/ref_line_len<0.2 or src_line_len/ref_line_len>5 or ref_line_len<np.quantile(ref_line_len_arr,0.5):
                        continue
                    src_line=np.array([pt1[0],pt1[1],pt2[0],pt2[1]])
                    src_line_dir=(pt2-pt1)/np.linalg.norm(pt2-pt1)
                    angle1=abs(np.sum(src_line_dir*pt1_normal))
                    angle2=abs(np.sum(src_line_dir*pt2_normal))
                    #filter out the two points that are not consitute a building facade
                    if angle1>np.cos(80*math.pi/180) or angle2>np.cos(80*math.pi/180):
                        continue
                    R_90=np.array([[0,-1],[1,0]])
                    R_90_neg=np.array([[0,1],[-1,0]])
                    src_normal1=R_90@src_line_dir
                    src_normal2=R_90_neg@src_line_dir
                    T1,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_normal1,self.air.lines[j,0,:],self.air.lines_normals[j])
                    T2,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_normal2,self.air.lines[j,0,:],self.air.lines_normals[j])
                    if np.sum(np.isnan(T1))>0 or np.sum(np.isnan(T2))>0:
                        continue
                    src_set_trans1=(T1[:2,:2]@src_line_pixels.T+T1[:2,2].reshape(2,1)).T
                    src_set_trans2=(T2[:2,:2]@src_line_pixels.T+T2[:2,2].reshape(2,1)).T
                    R1=T1[:2,:2]/scale
                    R2=T2[:2,:2]/scale
                    src_normals_trans1=(R1@self.ground.lines_pixels_normals.T).T
                    src_normals_trans2=(R2@self.ground.lines_pixels_normals.T).T

                    d1,i1=ref_tree.query(src_set_trans1,k=1)
                    d2,i2=ref_tree.query(src_set_trans2,k=1)

                    #normal difference
                    ref_pixels1=ref_line_pixels[i1]
                    mask=np.linalg.norm(src_set_trans1-ref_pixels1,axis=1)
                    mask=mask<3
                    match_direction=(src_set_trans1-ref_pixels1)/np.expand_dims(np.linalg.norm(src_set_trans1-ref_pixels1,axis=1),axis=1)
                    sim1_match_src=abs(np.sum(match_direction*src_normals_trans1,axis=1))
                    sim1_match_ref=abs(np.sum(match_direction*self.air.lines_pixels_normals[i1],axis=1))
                    sim1_match_src[mask]=1
                    sim1_match_ref[mask]=1
                    sim1_match_src=np.sum(sim1_match_src*self.ground.lines_pixels_weights)
                    sim1_match_ref=np.sum(sim1_match_ref*self.ground.lines_pixels_weights)
                    sim1_src_ref=np.sum(np.sum((self.air.lines_pixels_normals[i1]*src_normals_trans1),axis=1)*self.ground.lines_pixels_weights)
                    dis1=np.sum(d1<12)/d1.shape[0]
                    sim1=nc_weight*sim1_src_ref+cnc_weight*(sim1_match_ref+sim1_match_src)/2+dis_weight*dis1
                    

                    ref_pixels2=ref_line_pixels[i2]
                    mask=np.linalg.norm(src_set_trans2-ref_pixels2,axis=1)
                    mask=mask<3
                    match_direction=(src_set_trans2-ref_pixels2)/np.expand_dims(np.linalg.norm(src_set_trans2-ref_pixels2,axis=1),axis=1)
                    sim2_match_src=abs(np.sum(match_direction*src_normals_trans2,axis=1))
                    sim2_match_ref=abs(np.sum(match_direction*self.air.lines_pixels_normals[i2],axis=1))
                    sim2_match_src[mask]=1
                    sim2_match_ref[mask]=1
                    sim2_match_src=np.sum(sim2_match_src*self.ground.lines_pixels_weights)
                    sim2_match_ref=np.sum(sim2_match_ref*self.ground.lines_pixels_weights)
                    sim2_src_ref=np.sum(np.sum((self.air.lines_pixels_normals[i2]*src_normals_trans2),axis=1)*self.ground.lines_pixels_weights)
                    dis2=np.sum(d2<12)/d2.shape[0]
                    sim2=nc_weight*sim2_src_ref+cnc_weight*(sim2_match_ref+sim2_match_src)/2+dis_weight*dis2
                    

                    src_pair=src_line
                    ref_pair=self.air.lines[j,0,:]
                    #best_sim_list.append(sim1)
                    final_score_list.append(sim1)
                    normal_consis_list.append(sim1_src_ref)
                    corr_normal_consis_list.append(0.5*(sim1_match_ref+sim1_match_src))
                    dis_list.append(dis1)
                    best_T_list.append(T1)
                    best_src_line_list.append(src_pair)
                    best_ref_line_list.append(ref_pair)

                    #best_sim_list.append(sim2)
                    final_score_list.append(sim2)
                    normal_consis_list.append(sim2_src_ref)
                    corr_normal_consis_list.append(0.5*(sim2_match_ref+sim2_match_src))
                    dis_list.append(dis2)
                    best_T_list.append(T2)
                    best_src_line_list.append(src_pair)
                    best_ref_line_list.append(ref_pair)
                    
        best_sim_list=sorted(final_score_list,reverse=True)
        sim=best_sim_list[0]
        index=final_score_list.index(sim)
        best_T_final=best_T_list[index]
        if self.config.use_sem:
            ref_sem_arr=cv2.imread(os.path.join(self.config.out_dir,"air_sem_img.png"))
            src_sem_arr=cv2.imread(os.path.join(self.config.out_dir,"ground_sem_img.png"))
            src_sem_color_arr=cv2.imread(os.path.join(self.config.out_dir,"ground_sem_color_img.png"))
        for i in range(40):
            sim=best_sim_list[i]
            index=final_score_list.index(sim)
            normal_consis=normal_consis_list[index]
            corr_normal_consis=corr_normal_consis_list[index]
            dis=dis_list[index]
            best_T=best_T_list[index]
            best_src_line=best_src_line_list[index]
            best_ref_line=best_ref_line_list[index]

            warped_lines=self.ground.lines.copy()
            warped_lines[:,0,:2]=(best_T[:2,:2]@self.ground.lines[:,0,:2].T+best_T[:2,2].reshape(2,1)).T
            warped_lines[:,0,2:]=(best_T[:2,:2]@self.ground.lines[:,0,2:].T+best_T[:2,2].reshape(2,1)).T
            warped_line_img=np.zeros((self.air.img.shape[0],self.air.img.shape[1],3),dtype=self.air.img.dtype)
            for ii in range(self.ground.lines.shape[0]):
                cv2.line(warped_line_img,warped_lines[ii,0,:2].astype(np.int32),warped_lines[ii,0,2:].astype(np.int32),(0,0,255),1)
            cv2.imwrite(os.path.join(self.config.out_dir,'warp_ground_line_'+str(i)+'.png'),warped_line_img)

            match_img=drawmatch_line(self.ground.img,self.air.img,best_src_line,best_ref_line)
            cv2.imwrite(os.path.join(self.config.out_dir,"match_{}_score_{:5.4f}_nc_{:5.4f}_cnc_{:5.4f}_dis_{:5.4f}.png".format(i,sim,normal_consis,corr_normal_consis,dis)),match_img)

            if self.config.use_sem:
                warped_srcimg = cv2.warpAffine(src_sem_arr,best_T[:2,:],(ref_sem_arr.shape[1],ref_sem_arr.shape[0]))  
                warped_srccolorimg = cv2.warpAffine(src_sem_color_arr,best_T[:2,:],(ref_sem_arr.shape[1],ref_sem_arr.shape[0]))    
                valid_ind=warped_srcimg!=0
                same_ind=warped_srcimg==ref_sem_arr
                ind=same_ind*valid_ind
                sem_consis=np.sum(ind)/np.sum(valid_ind)
                cv2.imwrite(os.path.join(self.config.out_dir,'warp_ground_sem_{}_score_{:5.4f}.png'.format(i,sem_consis)),warped_srccolorimg)
        # change to 3D transformation, src_pts -> rotate plane to Z axis -> scale init -> center to lefttop bbox -> object to image space -> Line Transformation 3D -> image to object space-> back to ref lefttop bbox -> rotate back to ref
        Trans_line_2D=best_T_final
        scale=np.sqrt(Trans_line_2D[0,0]**2+Trans_line_2D[0,1]**2)
        Rot_3D=np.identity(3)*scale
        Rot_3D[:2,:2]=Trans_line_2D[:2,:2]
        Trans_3D=np.zeros(3)
        Trans_3D[:2]=Trans_line_2D[:2,2]*self.air.reso
        Transformation_Line_3D=np.identity(4)
        Transformation_Line_3D[:3,:3]=Rot_3D
        Transformation_Line_3D[:3,3]=Trans_3D
        Trans_object2ortho=np.identity(4)
        Trans_object2ortho[1,1]=-1
        Trans_SRC_LEFTTOP=np.identity(4)
        Trans_SRC_LEFTTOP[0,3]=-self.ground.bound_min[0]
        Trans_SRC_LEFTTOP[1,3]=-self.ground.bound_max[1]
        Trans_REF_LEFTTOP=np.identity(4)
        Trans_REF_LEFTTOP[0,3]=-self.air.bound_min[0]
        Trans_REF_LEFTTOP[1,3]=-self.air.bound_max[1]
        Trans_SRC_ROTPLANE=np.identity(4)
        Trans_SRC_ROTPLANE[:3,:3]=self.ground.rot_plane
        Trans_REF_ROTPLANE=np.identity(4)
        Trans_REF_ROTPLANE[:3,:3]=self.air.rot_plane
        HEIGHT_SHIFT=np.identity(4)
        HEIGHT_SHIFT[2,3]=self.air.z_height-self.ground.z_height*scale
        SCALE_INIT=np.identity(4)
        SCALE_INIT[:3,:3]*=self.ground.scale_init
        out_transformation=np.linalg.inv(Trans_REF_ROTPLANE)@HEIGHT_SHIFT@np.linalg.inv(Trans_REF_LEFTTOP)@np.linalg.inv(Trans_object2ortho)@Transformation_Line_3D@Trans_object2ortho@Trans_SRC_LEFTTOP@SCALE_INIT@Trans_SRC_ROTPLANE@self.config.init_trans
        self.T_g2a=out_transformation
    
    def align_g2a_simple(self):
        src_line_pixels=self.ground.lines_pixels_set.astype(np.int32)
        ref_line_pixels=self.air.lines_pixels_set.astype(np.int32)

        src_line_img=np.zeros((self.ground.img.shape[0],self.ground.img.shape[1]))
        ref_line_img=np.zeros((self.air.img.shape[0],self.air.img.shape[1]))
        for i in range(src_line_pixels.shape[0]):
            y=src_line_pixels[i][1]
            x=src_line_pixels[i][0]
            y=min(y,src_line_img.shape[0]-1)
            x=min(x,src_line_img.shape[1]-1)
            src_line_img[y,x]=1
        for i in range(ref_line_pixels.shape[0]):
            y=ref_line_pixels[i][1]
            x=ref_line_pixels[i][0]
            y=min(y,ref_line_img.shape[0]-1)
            x=min(x,ref_line_img.shape[1]-1)
            ref_line_img[y,x]=1
        src_line_img*=255
        ref_line_img*=255
        for i in range(self.ground.lines.shape[0]):
            pt1=self.ground.lines[i][0][:2]
            pt2=self.ground.lines[i][0][2:]
            mid_pt=(pt1+pt2)/2
            normal_end_pt=mid_pt+self.ground.lines_normals[i]*5
            cv2.line(src_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
        for i in range(self.air.lines.shape[0]):
            pt1=self.air.lines[i][0][:2]
            pt2=self.air.lines[i][0][2:]
            mid_pt=(pt1+pt2)/2
            normal_end_pt=mid_pt+self.air.lines_normals[i]*5
            cv2.line(ref_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
        cv2.imwrite(os.path.join(self.config.out_dir,"ground_line_pixel.png"),src_line_img)
        cv2.imwrite(os.path.join(self.config.out_dir,"air_line_pixel.png"),ref_line_img)

        ref_tree=KDTree(ref_line_pixels)

        src_end_pts=[]
        src_end_pts_normals=[]
        for i in range(self.ground.lines.shape[0]):
            src_end_pts.append(self.ground.lines[i,0,:2])
            src_end_pts.append(self.ground.lines[i,0,2:])
            src_end_pts_normals.append(self.ground.lines_normals[i])
            src_end_pts_normals.append(self.ground.lines_normals[i])
        src_end_pts=np.array(src_end_pts)
        num_src_pts=src_end_pts.shape[0]
        num_ref_lines=self.air.lines.shape[0]
        num_src_lines=self.ground.lines.shape[0]
        best_sim=-999
        best_src_line=None
        best_ref_line=None
        best_T=np.identity(4)
        best_T_list=[]
        final_score_list=[]
        corr_normal_consis_list=[]
        normal_consis_list=[]
        dis_list=[]
        
        best_src_line_list=[]
        best_ref_line_list=[]

        #print("Total # line seg matches: {}".format(num_src_lines*num_ref_lines))
        ref_line_len_list=[]
        for j in range(num_ref_lines):
            ref_line_len=np.linalg.norm(self.air.lines[j,0,:2]-self.air.lines[j,0,2:])
            ref_line_len_list.append(ref_line_len)
        ref_line_len_arr=np.array(ref_line_len_list)
        print("Ref line: max: {}, min: {}, mean: {}, 90%: {}, 70%:{}, 50%:{},30%:{}".format(np.max(ref_line_len_arr),np.min(ref_line_len_arr),np.mean(ref_line_len_arr),
                                                                                            np.quantile(ref_line_len_arr,0.9),
                                                                                            np.quantile(ref_line_len_arr,0.7),
                                                                                            np.quantile(ref_line_len_arr,0.5),
                                                                                            np.quantile(ref_line_len_arr,0.3)
                                                                                            ))
        cnc_weight=0.1
        dis_weight=0.5
        nc_weight=0.4
        for j in range(num_ref_lines):
            for i in range(num_src_lines):
                pt1=self.ground.lines[i,0,:2]
                pt2=self.ground.lines[i,0,2:]
                src_line_len=np.linalg.norm(pt1-pt2)
                ref_line_len=np.linalg.norm(self.air.lines[j,0,:2]-self.air.lines[j,0,2:])
                if src_line_len/ref_line_len<0.2 or src_line_len/ref_line_len>5 or ref_line_len<np.quantile(ref_line_len_arr,0.5):
                    continue
                src_line=np.array([pt1[0],pt1[1],pt2[0],pt2[1]])
                src_line_normal=self.ground.lines_normals[i]

                T1,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_line_normal,self.air.lines[j,0,:],self.air.lines_normals[j])
                if np.sum(np.isnan(T1))>0:
                    continue
                src_set_trans1=(T1[:2,:2]@src_line_pixels.T+T1[:2,2].reshape(2,1)).T
                R1=T1[:2,:2]/scale
                src_normals_trans1=(R1@self.ground.lines_pixels_normals.T).T

                d1,i1=ref_tree.query(src_set_trans1,k=1)

                #normal difference
                ref_pixels1=ref_line_pixels[i1]
                mask=np.linalg.norm(src_set_trans1-ref_pixels1,axis=1)
                mask=mask<3
                match_direction=(src_set_trans1-ref_pixels1)/np.expand_dims(np.linalg.norm(src_set_trans1-ref_pixels1,axis=1),axis=1)
                sim1_match_src=abs(np.sum(match_direction*src_normals_trans1,axis=1))
                sim1_match_ref=abs(np.sum(match_direction*self.air.lines_pixels_normals[i1],axis=1))
                sim1_match_src[mask]=1
                sim1_match_ref[mask]=1
                sim1_match_src=np.sum(sim1_match_src*self.ground.lines_pixels_weights)
                sim1_match_ref=np.sum(sim1_match_ref*self.ground.lines_pixels_weights)
                sim1_src_ref=np.sum(np.sum((self.air.lines_pixels_normals[i1]*src_normals_trans1),axis=1)*self.ground.lines_pixels_weights)
                dis1=np.sum(d1<12)/d1.shape[0]
                sim1=nc_weight*sim1_src_ref+cnc_weight*(sim1_match_ref+sim1_match_src)/2
                
                if sim1_src_ref>1 or 0.5*(sim1_match_ref+sim1_match_src)>1:
                    continue
                src_pair=src_line
                ref_pair=self.air.lines[j,0,:]
                #best_sim_list.append(sim1)
                final_score_list.append(sim1)
                normal_consis_list.append(sim1_src_ref)
                corr_normal_consis_list.append(0.5*(sim1_match_ref+sim1_match_src))
                dis_list.append(dis1)
                best_T_list.append(T1)
                best_src_line_list.append(src_pair)
                best_ref_line_list.append(ref_pair)
                    
        best_sim_list=sorted(final_score_list,reverse=True)
        sim=best_sim_list[0]
        index=final_score_list.index(sim)
        best_T_final=best_T_list[index]
        if self.config.use_sem:
            ref_sem_arr=cv2.imread(os.path.join(self.config.out_dir,"air_sem_img.png"))
            src_sem_arr=cv2.imread(os.path.join(self.config.out_dir,"ground_sem_img.png"))
            src_sem_color_arr=cv2.imread(os.path.join(self.config.out_dir,"ground_sem_color_img.png"))
        for i in range(20):
            sim=best_sim_list[i]
            index=final_score_list.index(sim)
            normal_consis=normal_consis_list[index]
            corr_normal_consis=corr_normal_consis_list[index]
            dis=dis_list[index]
            best_T=best_T_list[index]
            best_src_line=best_src_line_list[index]
            best_ref_line=best_ref_line_list[index]

            warped_lines=self.ground.lines.copy()
            warped_lines[:,0,:2]=(best_T[:2,:2]@self.ground.lines[:,0,:2].T+best_T[:2,2].reshape(2,1)).T
            warped_lines[:,0,2:]=(best_T[:2,:2]@self.ground.lines[:,0,2:].T+best_T[:2,2].reshape(2,1)).T
            warped_line_img=np.zeros((self.air.img.shape[0],self.air.img.shape[1],3),dtype=self.air.img.dtype)
            for ii in range(self.ground.lines.shape[0]):
                cv2.line(warped_line_img,warped_lines[ii,0,:2].astype(np.int32),warped_lines[ii,0,2:].astype(np.int32),(0,0,255),1)
            cv2.imwrite(os.path.join(self.config.out_dir,'warp_ground_line_'+str(i)+'.png'),warped_line_img)

            match_img=drawmatch_line(self.ground.img,self.air.facade_img,best_src_line,best_ref_line)
            cv2.imwrite(os.path.join(self.config.out_dir,"match_{}_score_{:5.4f}_nc_{:5.4f}_cnc_{:5.4f}.png".format(i,sim,normal_consis,corr_normal_consis)),match_img)

            if self.config.use_sem:
                warped_srcimg = cv2.warpAffine(src_sem_arr,best_T[:2,:],(ref_sem_arr.shape[1],ref_sem_arr.shape[0]))  
                warped_srccolorimg = cv2.warpAffine(src_sem_color_arr,best_T[:2,:],(ref_sem_arr.shape[1],ref_sem_arr.shape[0]))    
                valid_ind=warped_srcimg!=0
                same_ind=warped_srcimg==ref_sem_arr
                ind=same_ind*valid_ind
                sem_consis=np.sum(ind)/np.sum(valid_ind)
                cv2.imwrite(os.path.join(self.config.out_dir,'warp_ground_sem_{}_score_{:5.4f}.png'.format(i,sem_consis)),warped_srccolorimg)
        # change to 3D transformation, src_pts -> rotate plane to Z axis -> scale init -> center to lefttop bbox -> object to image space -> Line Transformation 3D -> image to object space-> back to ref lefttop bbox -> rotate back to ref
        Trans_line_2D=best_T_final
        scale=np.sqrt(Trans_line_2D[0,0]**2+Trans_line_2D[0,1]**2)
        Rot_3D=np.identity(3)*scale
        Rot_3D[:2,:2]=Trans_line_2D[:2,:2]
        Trans_3D=np.zeros(3)
        Trans_3D[:2]=Trans_line_2D[:2,2]*self.air.reso
        Transformation_Line_3D=np.identity(4)
        Transformation_Line_3D[:3,:3]=Rot_3D
        Transformation_Line_3D[:3,3]=Trans_3D
        Trans_object2ortho=np.identity(4)
        Trans_object2ortho[1,1]=-1
        Trans_SRC_LEFTTOP=np.identity(4)
        Trans_SRC_LEFTTOP[0,3]=-self.ground.bound_min[0]
        Trans_SRC_LEFTTOP[1,3]=-self.ground.bound_max[1]
        Trans_REF_LEFTTOP=np.identity(4)
        Trans_REF_LEFTTOP[0,3]=-self.air.bound_min[0]
        Trans_REF_LEFTTOP[1,3]=-self.air.bound_max[1]
        Trans_SRC_ROTPLANE=np.identity(4)
        Trans_SRC_ROTPLANE[:3,:3]=self.ground.rot_plane
        Trans_REF_ROTPLANE=np.identity(4)
        Trans_REF_ROTPLANE[:3,:3]=self.air.rot_plane
        HEIGHT_SHIFT=np.identity(4)
        HEIGHT_SHIFT[2,3]=self.air.z_height-self.ground.z_height*scale
        SCALE_INIT=np.identity(4)
        SCALE_INIT[:3,:3]*=self.ground.scale_init
        out_transformation=np.linalg.inv(Trans_REF_ROTPLANE)@HEIGHT_SHIFT@np.linalg.inv(Trans_REF_LEFTTOP)@np.linalg.inv(Trans_object2ortho)@Transformation_Line_3D@Trans_object2ortho@Trans_SRC_LEFTTOP@SCALE_INIT@Trans_SRC_ROTPLANE@self.config.init_trans
        self.T_g2a=out_transformation

    def align_g2a_full(self):
        src_line_pixels=self.ground.lines_pixels_set.astype(np.int32)
        ref_line_pixels=self.air.lines_pixels_set.astype(np.int32)

        src_line_img=np.zeros((self.ground.img.shape[0],self.ground.img.shape[1]))
        ref_line_img=np.zeros((self.air.img.shape[0],self.air.img.shape[1]))
        for i in range(src_line_pixels.shape[0]):
            src_line_img[src_line_pixels[i][1],src_line_pixels[i][0]]=1
        for i in range(ref_line_pixels.shape[0]):
            ref_line_img[ref_line_pixels[i][1],ref_line_pixels[i][0]]=1
        src_line_img*=255
        ref_line_img*=255
        for i in range(self.ground.lines.shape[0]):
            pt1=self.ground.lines[i][0][:2]
            pt2=self.ground.lines[i][0][2:]
            mid_pt=(pt1+pt2)/2
            normal_end_pt=mid_pt+self.ground.lines_normals[i]*5
            cv2.line(src_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
        for i in range(self.air.lines.shape[0]):
            pt1=self.air.lines[i][0][:2]
            pt2=self.air.lines[i][0][2:]
            mid_pt=(pt1+pt2)/2
            normal_end_pt=mid_pt+self.air.lines_normals[i]*5
            cv2.line(ref_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
        cv2.imwrite(os.path.join(self.config.out_dir,"ground_line_pixel.png"),src_line_img)
        cv2.imwrite(os.path.join(self.config.out_dir,"air_line_pixel.png"),ref_line_img)

        ref_tree=KDTree(ref_line_pixels)

        src_end_pts=[]
        src_end_pts_normals=[]
        for i in range(self.ground.lines.shape[0]):
            src_end_pts.append(self.ground.lines[i,0,:2])
            src_end_pts.append(self.ground.lines[i,0,2:])
            src_end_pts_normals.append(self.ground.lines_normals[i])
            src_end_pts_normals.append(self.ground.lines_normals[i])
        src_end_pts=np.array(src_end_pts)
        num_src_pts=src_end_pts.shape[0]

        ref_end_pts=[]
        ref_end_pts_normals=[]
        for i in range(self.air.lines.shape[0]):
            ref_end_pts.append(self.air.lines[i,0,:2])
            ref_end_pts.append(self.air.lines[i,0,2:])
            ref_end_pts_normals.append(self.air.lines_normals[i])
            ref_end_pts_normals.append(self.air.lines_normals[i])
        ref_end_pts=np.array(ref_end_pts)
        num_ref_pts=ref_end_pts.shape[0]

        num_ref_lines=self.air.lines.shape[0]
        best_sim=-999
        best_src_line=None
        best_ref_line=None
        best_T=np.identity(4)
        best_T_list=[]
        final_score_list=[]
        corr_normal_consis_list=[]
        normal_consis_list=[]
        dis_list=[]
        
        best_src_line_list=[]
        best_ref_line_list=[]

        #print("Total # line seg matches: {}".format(num_src_lines*num_ref_lines))
        ref_line_len_list=[]
        for j in range(num_ref_lines):
            ref_line_len=np.linalg.norm(self.air.lines[j,0,:2]-self.air.lines[j,0,2:])
            ref_line_len_list.append(ref_line_len)
        ref_line_len_arr=np.array(ref_line_len_list)
        print("Ref line: max: {}, min: {}, mean: {}, 90%: {}, 70%:{}, 50%:{},30%:{}".format(np.max(ref_line_len_arr),np.min(ref_line_len_arr),np.mean(ref_line_len_arr),
                                                                                            np.quantile(ref_line_len_arr,0.9),
                                                                                            np.quantile(ref_line_len_arr,0.7),
                                                                                            np.quantile(ref_line_len_arr,0.5),
                                                                                            np.quantile(ref_line_len_arr,0.3)
                                                                                            ))
        cnc_weight=0.1
        dis_weight=0.5
        nc_weight=0.4
        for jj1 in range(num_ref_pts):
            for jj2 in range(jj1,num_ref_pts):
                ref_pt1=ref_end_pts[jj1]
                ref_pt2=ref_end_pts[jj2]
                ref_pt1_normal=ref_end_pts_normals[jj1]
                ref_pt2_normal=ref_end_pts_normals[jj2]
                ref_pt_normal_mean=(ref_pt1_normal+ref_pt2_normal)/2
                ref_line=np.array([ref_pt1[0],ref_pt1[1],ref_pt2[0],ref_pt2[1]])
                ref_line_dir=(ref_pt2-ref_pt1)/np.linalg.norm(ref_pt2-ref_pt1)
                ref_line_normal=None
                R_90=np.array([[0,-1],[1,0]])
                R_90_neg=np.array([[0,1],[-1,0]])
                ref_line_normal1=R_90@ref_line_dir
                ref_line_normal2=R_90_neg@ref_line_dir
                if np.sum(ref_line_normal1*ref_pt_normal_mean)>0:
                    ref_line_normal=ref_line_normal1
                else:
                    ref_line_normal=ref_line_normal2
                # filter out large normal deviation line
                ref_normal_diff=np.sum(ref_pt1_normal*ref_pt2_normal)
                if ref_normal_diff<np.cos(np.pi/4):
                    continue
                for ii1 in range(num_src_pts):
                    for ii2 in range(ii1,num_src_pts):
                        pt1=src_end_pts[ii1]
                        pt2=src_end_pts[ii2]
                        pt1_normal=src_end_pts_normals[ii1]
                        pt2_normal=src_end_pts_normals[ii2]
                        pt_normal_mean=(pt1_normal+pt2_normal)/2
                        src_line_len=np.linalg.norm(pt1-pt2)
                        ref_line_len=np.linalg.norm(ref_pt1-ref_pt2)
                        if src_line_len/ref_line_len<0.2 or src_line_len/ref_line_len>5 or ref_line_len<np.quantile(ref_line_len_arr,0.5):
                            continue
                        src_line=np.array([pt1[0],pt1[1],pt2[0],pt2[1]])
                        src_line_dir=(pt2-pt1)/np.linalg.norm(pt2-pt1)
                        src_line_normal=None
                        
                        #filter out the two points that are not consitute a building facade
                        src_normal_diff=np.sum(pt1_normal*pt2_normal)
                        if src_normal_diff<np.cos(np.pi/4):
                            continue
                        R_90=np.array([[0,-1],[1,0]])
                        R_90_neg=np.array([[0,1],[-1,0]])
                        src_normal1=R_90@src_line_dir
                        src_normal2=R_90_neg@src_line_dir
                        if np.sum(src_normal1*pt_normal_mean)>0:
                            src_line_normal=src_normal1
                        else:
                            src_line_normal=src_normal2
                        T1,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_line_normal,ref_line,ref_line_normal)
                        if np.sum(np.isnan(T1))>0 :
                            continue
                        src_set_trans1=(T1[:2,:2]@src_line_pixels.T+T1[:2,2].reshape(2,1)).T
                        R1=T1[:2,:2]/scale
                        src_normals_trans1=(R1@self.ground.lines_pixels_normals.T).T
                        d1,i1=ref_tree.query(src_set_trans1,k=1)

                        #normal difference
                        ref_pixels1=ref_line_pixels[i1]
                        mask=np.linalg.norm(src_set_trans1-ref_pixels1,axis=1)
                        mask=mask<3
                        match_direction=(src_set_trans1-ref_pixels1)/np.expand_dims(np.linalg.norm(src_set_trans1-ref_pixels1,axis=1),axis=1)
                        sim1_match_src=abs(np.sum(match_direction*src_normals_trans1,axis=1))
                        sim1_match_ref=abs(np.sum(match_direction*self.air.lines_pixels_normals[i1],axis=1))
                        sim1_match_src[mask]=1
                        sim1_match_ref[mask]=1
                        sim1_match_src=np.sum(sim1_match_src*self.ground.lines_pixels_weights)
                        sim1_match_ref=np.sum(sim1_match_ref*self.ground.lines_pixels_weights)
                        sim1_src_ref=np.sum(np.sum((self.air.lines_pixels_normals[i1]*src_normals_trans1),axis=1)*self.ground.lines_pixels_weights)
                        dis1=np.sum(d1<12)/d1.shape[0]
                        sim1=nc_weight*sim1_src_ref+cnc_weight*(sim1_match_ref+sim1_match_src)/2+dis_weight*dis1
                        

                        src_pair=src_line
                        ref_pair=self.air.lines[j,0,:]
                        #best_sim_list.append(sim1)
                        final_score_list.append(sim1)
                        normal_consis_list.append(sim1_src_ref)
                        corr_normal_consis_list.append(0.5*(sim1_match_ref+sim1_match_src))
                        dis_list.append(dis1)
                        best_T_list.append(T1)
                        best_src_line_list.append(src_pair)
                        best_ref_line_list.append(ref_pair)

                    
        best_sim_list=sorted(final_score_list,reverse=True)
        sim=best_sim_list[0]
        index=final_score_list.index(sim)
        best_T_final=best_T_list[index]
        if self.config.use_sem:
            ref_sem_arr=cv2.imread(os.path.join(self.config.out_dir,"air_sem_img.png"))
            src_sem_arr=cv2.imread(os.path.join(self.config.out_dir,"ground_sem_img.png"))
            src_sem_color_arr=cv2.imread(os.path.join(self.config.out_dir,"ground_sem_color_img.png"))
        for i in range(40):
            sim=best_sim_list[i]
            index=final_score_list.index(sim)
            normal_consis=normal_consis_list[index]
            corr_normal_consis=corr_normal_consis_list[index]
            dis=dis_list[index]
            best_T=best_T_list[index]
            best_src_line=best_src_line_list[index]
            best_ref_line=best_ref_line_list[index]

            warped_lines=self.ground.lines.copy()
            warped_lines[:,0,:2]=(best_T[:2,:2]@self.ground.lines[:,0,:2].T+best_T[:2,2].reshape(2,1)).T
            warped_lines[:,0,2:]=(best_T[:2,:2]@self.ground.lines[:,0,2:].T+best_T[:2,2].reshape(2,1)).T
            warped_line_img=np.zeros((self.air.img.shape[0],self.air.img.shape[1],3),dtype=self.air.img.dtype)
            for ii in range(self.ground.lines.shape[0]):
                cv2.line(warped_line_img,warped_lines[ii,0,:2].astype(np.int32),warped_lines[ii,0,2:].astype(np.int32),(0,0,255),1)
            cv2.imwrite(os.path.join(self.config.out_dir,'warp_ground_line_'+str(i)+'.png'),warped_line_img)

            match_img=drawmatch_line(self.ground.img,self.air.img,best_src_line,best_ref_line)
            cv2.imwrite(os.path.join(self.config.out_dir,"match_{}_score_{:5.4f}_nc_{:5.4f}_cnc_{:5.4f}_dis_{:5.4f}.png".format(i,sim,normal_consis,corr_normal_consis,dis)),match_img)

            if self.config.use_sem:
                warped_srcimg = cv2.warpAffine(src_sem_arr,best_T[:2,:],(ref_sem_arr.shape[1],ref_sem_arr.shape[0]))  
                warped_srccolorimg = cv2.warpAffine(src_sem_color_arr,best_T[:2,:],(ref_sem_arr.shape[1],ref_sem_arr.shape[0]))    
                valid_ind=warped_srcimg!=0
                same_ind=warped_srcimg==ref_sem_arr
                ind=same_ind*valid_ind
                sem_consis=np.sum(ind)/np.sum(valid_ind)
                cv2.imwrite(os.path.join(self.config.out_dir,'warp_ground_sem_{}_score_{:5.4f}.png'.format(i,sem_consis)),warped_srccolorimg)
        # change to 3D transformation, src_pts -> rotate plane to Z axis -> scale init -> center to lefttop bbox -> object to image space -> Line Transformation 3D -> image to object space-> back to ref lefttop bbox -> rotate back to ref
        Trans_line_2D=best_T_final
        scale=np.sqrt(Trans_line_2D[0,0]**2+Trans_line_2D[0,1]**2)
        Rot_3D=np.identity(3)*scale
        Rot_3D[:2,:2]=Trans_line_2D[:2,:2]
        Trans_3D=np.zeros(3)
        Trans_3D[:2]=Trans_line_2D[:2,2]*self.air.reso
        Transformation_Line_3D=np.identity(4)
        Transformation_Line_3D[:3,:3]=Rot_3D
        Transformation_Line_3D[:3,3]=Trans_3D
        Trans_object2ortho=np.identity(4)
        Trans_object2ortho[1,1]=-1
        Trans_SRC_LEFTTOP=np.identity(4)
        Trans_SRC_LEFTTOP[0,3]=-self.ground.bound_min[0]
        Trans_SRC_LEFTTOP[1,3]=-self.ground.bound_max[1]
        Trans_REF_LEFTTOP=np.identity(4)
        Trans_REF_LEFTTOP[0,3]=-self.air.bound_min[0]
        Trans_REF_LEFTTOP[1,3]=-self.air.bound_max[1]
        Trans_SRC_ROTPLANE=np.identity(4)
        Trans_SRC_ROTPLANE[:3,:3]=self.ground.rot_plane
        Trans_REF_ROTPLANE=np.identity(4)
        Trans_REF_ROTPLANE[:3,:3]=self.air.rot_plane
        HEIGHT_SHIFT=np.identity(4)
        HEIGHT_SHIFT[2,3]=self.air.z_height-self.ground.z_height*scale
        SCALE_INIT=np.identity(4)
        SCALE_INIT[:3,:3]*=self.ground.scale_init
        out_transformation=np.linalg.inv(Trans_REF_ROTPLANE)@HEIGHT_SHIFT@np.linalg.inv(Trans_REF_LEFTTOP)@np.linalg.inv(Trans_object2ortho)@Transformation_Line_3D@Trans_object2ortho@Trans_SRC_LEFTTOP@SCALE_INIT@Trans_SRC_ROTPLANE@self.config.init_trans
        self.T_g2a=out_transformation

    def align_g2a_rawpixels(self):
        src_line_pixels=self.ground.pixels.astype(np.int32)
        ref_line_pixels=self.air.lines_pixels_set.astype(np.int32)

        src_line_img=np.zeros((self.ground.img.shape[0],self.ground.img.shape[1]))
        src_line_normal_img=np.zeros((self.ground.img.shape[0],self.ground.img.shape[1],2))
        ref_line_img=np.zeros((self.air.img.shape[0],self.air.img.shape[1]))
        for i in range(src_line_pixels.shape[0]):
            src_line_img[src_line_pixels[i][1],src_line_pixels[i][0]]=1
            src_line_normal_img[src_line_pixels[i][1],src_line_pixels[i][0],:]=self.ground.pixels_normals[i]
        for i in range(ref_line_pixels.shape[0]):
            ref_line_img[ref_line_pixels[i][1],ref_line_pixels[i][0]]=1
        src_line_img*=255
        ref_line_img*=255
        rows,cols=np.where(src_line_img!=0)
        cols=np.expand_dims(cols,axis=1)
        rows=np.expand_dims(rows,axis=1)
        src_line_pixels=np.concatenate([cols,rows],axis=1)
        src_line_pixels_normals=[]
        for i in range(src_line_pixels.shape[0]):
            col=src_line_pixels[i,0]
            row=src_line_pixels[i,1]
            src_line_pixels_normals.append(src_line_normal_img[row,col,:])
        src_line_pixels_normals=np.array(src_line_pixels_normals)
        for i in range(self.air.lines.shape[0]):
            pt1=self.air.lines[i][0][:2]
            pt2=self.air.lines[i][0][2:]
            mid_pt=(pt1+pt2)/2
            normal_end_pt=mid_pt+self.air.lines_normals[i]*5
            cv2.line(ref_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
        cv2.imwrite(os.path.join(self.config.out_dir,"ground_facade_pixel.png"),src_line_img)
        cv2.imwrite(os.path.join(self.config.out_dir,"air_line_pixel.png"),ref_line_img)

        ref_tree=KDTree(ref_line_pixels)

        src_end_pts=[]
        src_end_pts_normals=[]
        for i in range(self.ground.lines.shape[0]):
            src_end_pts.append(self.ground.lines[i,0,:2])
            src_end_pts.append(self.ground.lines[i,0,2:])
            src_end_pts_normals.append(self.ground.lines_normals[i])
            src_end_pts_normals.append(self.ground.lines_normals[i])
        src_end_pts=np.array(src_end_pts)
        num_src_pts=src_end_pts.shape[0]
        num_ref_lines=self.air.lines.shape[0]
        best_sim=-999
        best_src_line=None
        best_ref_line=None
        best_T=np.identity(4)
        best_T_list=[]
        final_score_list=[]
        corr_normal_consis_list=[]
        normal_consis_list=[]
        dis_list=[]
        
        best_src_line_list=[]
        best_ref_line_list=[]

        #print("Total # line seg matches: {}".format(num_src_lines*num_ref_lines))
        ref_line_len_list=[]
        for j in range(num_ref_lines):
            ref_line_len=np.linalg.norm(self.air.lines[j,0,:2]-self.air.lines[j,0,2:])
            ref_line_len_list.append(ref_line_len)
        ref_line_len_arr=np.array(ref_line_len_list)
        print("Ref line: max: {}, min: {}, mean: {}, 90%: {}, 70%:{}, 50%:{},30%:{}".format(np.max(ref_line_len_arr),np.min(ref_line_len_arr),np.mean(ref_line_len_arr),
                                                                                            np.quantile(ref_line_len_arr,0.9),
                                                                                            np.quantile(ref_line_len_arr,0.7),
                                                                                            np.quantile(ref_line_len_arr,0.5),
                                                                                            np.quantile(ref_line_len_arr,0.3)
                                                                                            ))
        cnc_weight=0.1
        dis_weight=0.5
        nc_weight=0.4
        for j in range(num_ref_lines):
            for ii1 in range(num_src_pts):
                for ii2 in range(ii1,num_src_pts):
                    pt1=src_end_pts[ii1]
                    pt2=src_end_pts[ii2]
                    pt1_normal=src_end_pts_normals[ii1]
                    pt2_normal=src_end_pts_normals[ii2]
                    src_line_len=np.linalg.norm(pt1-pt2)
                    ref_line_len=np.linalg.norm(self.air.lines[j,0,:2]-self.air.lines[j,0,2:])
                    if src_line_len/ref_line_len<0.2 or src_line_len/ref_line_len>5 or ref_line_len<np.quantile(ref_line_len_arr,0.5):
                        continue
                    src_line=np.array([pt1[0],pt1[1],pt2[0],pt2[1]])
                    src_line_dir=(pt2-pt1)/np.linalg.norm(pt2-pt1)
                    angle1=abs(np.sum(src_line_dir*pt1_normal))
                    angle2=abs(np.sum(src_line_dir*pt2_normal))
                    #filter out the two points that are not consitute a building facade
                    if angle1>np.cos(80*math.pi/180) or angle2>np.cos(80*math.pi/180):
                        continue
                    R_90=np.array([[0,-1],[1,0]])
                    R_90_neg=np.array([[0,1],[-1,0]])
                    src_normal1=R_90@src_line_dir
                    src_normal2=R_90_neg@src_line_dir
                    T1,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_normal1,self.air.lines[j,0,:],self.air.lines_normals[j])
                    T2,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_normal2,self.air.lines[j,0,:],self.air.lines_normals[j])
                    if np.sum(np.isnan(T1))>0 or np.sum(np.isnan(T2))>0:
                        continue
                    src_set_trans1=(T1[:2,:2]@src_line_pixels.T+T1[:2,2].reshape(2,1)).T
                    src_set_trans2=(T2[:2,:2]@src_line_pixels.T+T2[:2,2].reshape(2,1)).T
                    R1=T1[:2,:2]/scale
                    R2=T2[:2,:2]/scale
                    src_normals_trans1=(R1@src_line_pixels_normals.T).T
                    src_normals_trans2=(R2@src_line_pixels_normals.T).T

                    d1,i1=ref_tree.query(src_set_trans1,k=1)
                    d2,i2=ref_tree.query(src_set_trans2,k=1)

                    #normal difference
                    ref_pixels1=ref_line_pixels[i1]
                    mask=np.linalg.norm(src_set_trans1-ref_pixels1,axis=1)
                    mask=mask<3
                    match_direction=(src_set_trans1-ref_pixels1)/np.expand_dims(np.linalg.norm(src_set_trans1-ref_pixels1,axis=1),axis=1)
                    sim1_match_src=abs(np.sum(match_direction*src_normals_trans1,axis=1))
                    sim1_match_ref=abs(np.sum(match_direction*self.air.lines_pixels_normals[i1],axis=1))
                    sim1_match_src[mask]=1
                    sim1_match_ref[mask]=1
                    sim1_match_src=np.sum(sim1_match_src)
                    sim1_match_ref=np.sum(sim1_match_ref)
                    sim1_src_ref=np.sum(np.sum((self.air.lines_pixels_normals[i1]*src_normals_trans1),axis=1))
                    dis1=np.sum(d1<12)/d1.shape[0]
                    sim1=nc_weight*sim1_src_ref+cnc_weight*(sim1_match_ref+sim1_match_src)/2+dis_weight*dis1
                    

                    ref_pixels2=ref_line_pixels[i2]
                    mask=np.linalg.norm(src_set_trans2-ref_pixels2,axis=1)
                    mask=mask<3
                    match_direction=(src_set_trans2-ref_pixels2)/np.expand_dims(np.linalg.norm(src_set_trans2-ref_pixels2,axis=1),axis=1)
                    sim2_match_src=abs(np.sum(match_direction*src_normals_trans2,axis=1))
                    sim2_match_ref=abs(np.sum(match_direction*self.air.lines_pixels_normals[i2],axis=1))
                    sim2_match_src[mask]=1
                    sim2_match_ref[mask]=1
                    sim2_match_src=np.sum(sim2_match_src)
                    sim2_match_ref=np.sum(sim2_match_ref)
                    sim2_src_ref=np.sum(np.sum((self.air.lines_pixels_normals[i2]*src_normals_trans2),axis=1))
                    dis2=np.sum(d2<12)/d2.shape[0]
                    sim2=nc_weight*sim2_src_ref+cnc_weight*(sim2_match_ref+sim2_match_src)/2+dis_weight*dis2
                    

                    src_pair=src_line
                    ref_pair=self.air.lines[j,0,:]
                    #best_sim_list.append(sim1)
                    final_score_list.append(sim1)
                    normal_consis_list.append(sim1_src_ref)
                    corr_normal_consis_list.append(0.5*(sim1_match_ref+sim1_match_src))
                    dis_list.append(dis1)
                    best_T_list.append(T1)
                    best_src_line_list.append(src_pair)
                    best_ref_line_list.append(ref_pair)

                    #best_sim_list.append(sim2)
                    final_score_list.append(sim2)
                    normal_consis_list.append(sim2_src_ref)
                    corr_normal_consis_list.append(0.5*(sim2_match_ref+sim2_match_src))
                    dis_list.append(dis2)
                    best_T_list.append(T2)
                    best_src_line_list.append(src_pair)
                    best_ref_line_list.append(ref_pair)
                    
        best_sim_list=sorted(final_score_list,reverse=True)
        sim=best_sim_list[0]
        index=final_score_list.index(sim)
        best_T_final=best_T_list[index]
        if self.config.use_sem:
            ref_sem_arr=cv2.imread(os.path.join(self.config.out_dir,"air_sem_img.png"))
            src_sem_arr=cv2.imread(os.path.join(self.config.out_dir,"ground_sem_img.png"))
            src_sem_color_arr=cv2.imread(os.path.join(self.config.out_dir,"ground_sem_color_img.png"))
        for i in range(40):
            sim=best_sim_list[i]
            index=final_score_list.index(sim)
            normal_consis=normal_consis_list[index]
            corr_normal_consis=corr_normal_consis_list[index]
            dis=dis_list[index]
            best_T=best_T_list[index]
            best_src_line=best_src_line_list[index]
            best_ref_line=best_ref_line_list[index]

            warped_lines=self.ground.lines.copy()
            warped_lines[:,0,:2]=(best_T[:2,:2]@self.ground.lines[:,0,:2].T+best_T[:2,2].reshape(2,1)).T
            warped_lines[:,0,2:]=(best_T[:2,:2]@self.ground.lines[:,0,2:].T+best_T[:2,2].reshape(2,1)).T
            warped_line_img=np.zeros((self.air.img.shape[0],self.air.img.shape[1],3),dtype=self.air.img.dtype)
            for ii in range(self.ground.lines.shape[0]):
                cv2.line(warped_line_img,warped_lines[ii,0,:2].astype(np.int32),warped_lines[ii,0,2:].astype(np.int32),(0,0,255),1)
            cv2.imwrite(os.path.join(self.config.out_dir,'warp_ground_line_'+str(i)+'.png'),warped_line_img)

            match_img=drawmatch_line(self.ground.img,self.air.img,best_src_line,best_ref_line)
            cv2.imwrite(os.path.join(self.config.out_dir,"match_{}_score_{:5.4f}_nc_{:5.4f}_cnc_{:5.4f}_dis_{:5.4f}.png".format(i,sim,normal_consis,corr_normal_consis,dis)),match_img)

            if self.config.use_sem:
                warped_srcimg = cv2.warpAffine(src_sem_arr,best_T[:2,:],(ref_sem_arr.shape[1],ref_sem_arr.shape[0]))  
                warped_srccolorimg = cv2.warpAffine(src_sem_color_arr,best_T[:2,:],(ref_sem_arr.shape[1],ref_sem_arr.shape[0]))    
                valid_ind=warped_srcimg!=0
                same_ind=warped_srcimg==ref_sem_arr
                ind=same_ind*valid_ind
                sem_consis=np.sum(ind)/np.sum(valid_ind)
                cv2.imwrite(os.path.join(self.config.out_dir,'warp_ground_sem_{}_score_{:5.4f}.png'.format(i,sem_consis)),warped_srccolorimg)
        # change to 3D transformation, src_pts -> rotate plane to Z axis -> scale init -> center to lefttop bbox -> object to image space -> Line Transformation 3D -> image to object space-> back to ref lefttop bbox -> rotate back to ref
        Trans_line_2D=best_T_final
        scale=np.sqrt(Trans_line_2D[0,0]**2+Trans_line_2D[0,1]**2)
        Rot_3D=np.identity(3)*scale
        Rot_3D[:2,:2]=Trans_line_2D[:2,:2]
        Trans_3D=np.zeros(3)
        Trans_3D[:2]=Trans_line_2D[:2,2]*self.air.reso
        Transformation_Line_3D=np.identity(4)
        Transformation_Line_3D[:3,:3]=Rot_3D
        Transformation_Line_3D[:3,3]=Trans_3D
        Trans_object2ortho=np.identity(4)
        Trans_object2ortho[1,1]=-1
        Trans_SRC_LEFTTOP=np.identity(4)
        Trans_SRC_LEFTTOP[0,3]=-self.ground.bound_min[0]
        Trans_SRC_LEFTTOP[1,3]=-self.ground.bound_max[1]
        Trans_REF_LEFTTOP=np.identity(4)
        Trans_REF_LEFTTOP[0,3]=-self.air.bound_min[0]
        Trans_REF_LEFTTOP[1,3]=-self.air.bound_max[1]
        Trans_SRC_ROTPLANE=np.identity(4)
        Trans_SRC_ROTPLANE[:3,:3]=self.ground.rot_plane
        Trans_REF_ROTPLANE=np.identity(4)
        Trans_REF_ROTPLANE[:3,:3]=self.air.rot_plane
        HEIGHT_SHIFT=np.identity(4)
        HEIGHT_SHIFT[2,3]=self.air.z_height-self.ground.z_height*scale
        SCALE_INIT=np.identity(4)
        SCALE_INIT[:3,:3]*=self.ground.scale_init
        out_transformation=np.linalg.inv(Trans_REF_ROTPLANE)@HEIGHT_SHIFT@np.linalg.inv(Trans_REF_LEFTTOP)@np.linalg.inv(Trans_object2ortho)@Transformation_Line_3D@Trans_object2ortho@Trans_SRC_LEFTTOP@SCALE_INIT@Trans_SRC_ROTPLANE@self.config.init_trans
        self.T_g2a=out_transformation

    def align_g2f(self):
        import math
        self.config.use_sem=False
        src_line_pixels=self.ground.lines_pixels_set.astype(np.int32)
        ref_line_pixels=self.footprint.lines_pixels_set.astype(np.int32)

        src_line_img=np.zeros((self.ground.img.shape[0],self.ground.img.shape[1]))
        ref_line_img=np.zeros((self.footprint.img.shape[0],self.footprint.img.shape[1]))
        for i in range(src_line_pixels.shape[0]):
            src_line_img[src_line_pixels[i][1],src_line_pixels[i][0]]=1
        for i in range(ref_line_pixels.shape[0]):
            ref_line_img[ref_line_pixels[i][1],ref_line_pixels[i][0]]=1
        src_line_img*=255
        ref_line_img*=255
        for i in range(self.ground.lines.shape[0]):
            pt1=self.ground.lines[i][0][:2]
            pt2=self.ground.lines[i][0][2:]
            mid_pt=(pt1+pt2)/2
            normal_end_pt=mid_pt+self.ground.lines_normals[i]*5
            cv2.line(src_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
        for i in range(self.footprint.lines.shape[0]):
            pt1=self.footprint.lines[i][0][:2]
            pt2=self.footprint.lines[i][0][2:]
            mid_pt=(pt1+pt2)/2
            normal_end_pt=mid_pt+self.footprint.lines_normals[i]*5
            cv2.line(ref_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
        cv2.imwrite(os.path.join(self.config.out_dir,"ground_line_pixel.png"),src_line_img)
        cv2.imwrite(os.path.join(self.config.out_dir,"footprint_line_pixel.png"),ref_line_img)

        ref_tree=KDTree(ref_line_pixels)

        src_end_pts=[]
        src_end_pts_normals=[]
        for i in range(self.ground.lines.shape[0]):
            src_end_pts.append(self.ground.lines[i,0,:2])
            src_end_pts.append(self.ground.lines[i,0,2:])
            src_end_pts_normals.append(self.ground.lines_normals[i])
            src_end_pts_normals.append(self.ground.lines_normals[i])

        src_end_pts=np.array(src_end_pts)
        src_end_pts_normals=np.array(src_end_pts_normals)
        num_src_pts=src_end_pts.shape[0]
        num_ref_lines=self.footprint.lines.shape[0]
        best_sim=-999
        best_src_line=None
        best_ref_line=None
        best_T=np.identity(4)
        best_T_list=[]
        final_score_list=[]
        corr_normal_consis_list=[]
        normal_consis_list=[]
        dis_list=[]
        
        best_src_line_list=[]
        best_ref_line_list=[]

        #print("Total # line seg matches: {}".format(num_src_lines*num_ref_lines))
        cnc_weight=0.1
        dis_weight=0.5
        nc_weight=0.4
        for j in range(num_ref_lines):
           for ii1 in range(num_src_pts):
                for ii2 in range(ii1,num_src_pts):
                    pt1=src_end_pts[ii1]
                    pt2=src_end_pts[ii2]
                    pt1_normal=src_end_pts_normals[ii1]
                    pt2_normal=src_end_pts_normals[ii2]
                    src_line_len=np.linalg.norm(pt1-pt2)
                    ref_line_len=np.linalg.norm(self.footprint.lines[j,0,:2]-self.footprint.lines[j,0,2:])
                    if src_line_len/ref_line_len<0.2 or src_line_len/ref_line_len>5:
                        continue
                    
                    src_line=np.array([pt1[0],pt1[1],pt2[0],pt2[1]])
                    src_line_dir=(pt2-pt1)/np.linalg.norm(pt2-pt1)
                    angle1=abs(np.sum(src_line_dir*pt1_normal))
                    angle2=abs(np.sum(src_line_dir*pt2_normal))
                    #filter out the two points that are not consitute a building facade
                    if angle1>np.cos(80*math.pi/180) or angle2>np.cos(80*math.pi/180):
                        continue
                    R_90=np.array([[0,-1],[1,0]])
                    R_90_neg=np.array([[0,1],[-1,0]])
                    src_normal1=R_90@src_line_dir
                    src_normal2=R_90_neg@src_line_dir
                    T1,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_normal1,self.footprint.lines[j,0,:],self.footprint.lines_normals[j])
                    T2,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_normal2,self.footprint.lines[j,0,:],self.footprint.lines_normals[j])
                    if np.sum(np.isnan(T1))>0 or np.sum(np.isnan(T2))>0:
                        continue
                    src_set_trans1=(T1[:2,:2]@src_line_pixels.T+T1[:2,2].reshape(2,1)).T
                    src_set_trans2=(T2[:2,:2]@src_line_pixels.T+T2[:2,2].reshape(2,1)).T
                    R1=T1[:2,:2]/scale
                    R2=T2[:2,:2]/scale
                    src_normals_trans1=(R1@self.ground.lines_pixels_normals.T).T
                    src_normals_trans2=(R2@self.ground.lines_pixels_normals.T).T

                    d1,i1=ref_tree.query(src_set_trans1,k=1)
                    d2,i2=ref_tree.query(src_set_trans2,k=1)

                    #normal difference
                    ref_pixels1=ref_line_pixels[i1]
                    mask=np.linalg.norm(src_set_trans1-ref_pixels1,axis=1)
                    mask=mask<3
                    match_direction=(src_set_trans1-ref_pixels1)/np.expand_dims(np.linalg.norm(src_set_trans1-ref_pixels1,axis=1),axis=1)
                    sim1_match_src=abs(np.sum(match_direction*src_normals_trans1,axis=1))
                    sim1_match_ref=abs(np.sum(match_direction*self.footprint.lines_pixels_normals[i1],axis=1))
                    sim1_match_src[mask]=1
                    sim1_match_ref[mask]=1
                    sim1_match_src=np.sum(sim1_match_src*self.ground.lines_pixels_weights)
                    sim1_match_ref=np.sum(sim1_match_ref*self.ground.lines_pixels_weights)
                    sim1_src_ref=np.sum(np.sum((self.footprint.lines_pixels_normals[i1]*src_normals_trans1),axis=1)*self.ground.lines_pixels_weights)
                    dis1=np.sum(d1<12)/d1.shape[0]
                    sim1=nc_weight*sim1_src_ref+cnc_weight*(sim1_match_ref+sim1_match_src)/2+dis_weight*dis1
                    

                    ref_pixels2=ref_line_pixels[i2]
                    mask=np.linalg.norm(src_set_trans2-ref_pixels2,axis=1)
                    mask=mask<3
                    match_direction=(src_set_trans2-ref_pixels2)/np.expand_dims(np.linalg.norm(src_set_trans2-ref_pixels2,axis=1),axis=1)
                    sim2_match_src=abs(np.sum(match_direction*src_normals_trans2,axis=1))
                    sim2_match_ref=abs(np.sum(match_direction*self.footprint.lines_pixels_normals[i2],axis=1))
                    sim2_match_src[mask]=1
                    sim2_match_ref[mask]=1
                    sim2_match_src=np.sum(sim2_match_src*self.ground.lines_pixels_weights)
                    sim2_match_ref=np.sum(sim2_match_ref*self.ground.lines_pixels_weights)
                    sim2_src_ref=np.sum(np.sum((self.footprint.lines_pixels_normals[i2]*src_normals_trans2),axis=1)*self.ground.lines_pixels_weights)
                    dis2=np.sum(d2<12)/d2.shape[0]
                    sim2=nc_weight*sim2_src_ref+cnc_weight*(sim2_match_ref+sim2_match_src)/2+dis_weight*dis2
                    

                    src_pair=src_line
                    ref_pair=self.footprint.lines[j,0,:]
                    #best_sim_list.append(sim1)
                    final_score_list.append(sim1)
                    normal_consis_list.append(sim1_src_ref)
                    corr_normal_consis_list.append(0.5*(sim1_match_ref+sim1_match_src))
                    dis_list.append(dis1)
                    best_T_list.append(T1)
                    best_src_line_list.append(src_pair)
                    best_ref_line_list.append(ref_pair)

                    #best_sim_list.append(sim2)
                    final_score_list.append(sim2)
                    normal_consis_list.append(sim2_src_ref)
                    corr_normal_consis_list.append(0.5*(sim2_match_ref+sim2_match_src))
                    dis_list.append(dis2)
                    best_T_list.append(T2)
                    best_src_line_list.append(src_pair)
                    best_ref_line_list.append(ref_pair)
                    
        best_sim_list=sorted(final_score_list,reverse=True)
        sim=best_sim_list[0]
        index=final_score_list.index(sim)
        best_T_final=best_T_list[index]
        for i in range(20):
            sim=best_sim_list[i]
            index=final_score_list.index(sim)
            normal_consis=normal_consis_list[index]
            corr_normal_consis=corr_normal_consis_list[index]
            dis=dis_list[index]
            best_T=best_T_list[index]
            best_src_line=best_src_line_list[index]
            best_ref_line=best_ref_line_list[index]

            warped_lines=self.ground.lines.copy()
            warped_lines[:,0,:2]=(best_T[:2,:2]@self.ground.lines[:,0,:2].T+best_T[:2,2].reshape(2,1)).T
            warped_lines[:,0,2:]=(best_T[:2,:2]@self.ground.lines[:,0,2:].T+best_T[:2,2].reshape(2,1)).T
            warped_line_img=np.zeros((self.footprint.img.shape[0],self.footprint.img.shape[1],3),dtype=self.footprint.img.dtype)
            for ii in range(self.ground.lines.shape[0]):
                cv2.line(warped_line_img,warped_lines[ii,0,:2].astype(np.int32),warped_lines[ii,0,2:].astype(np.int32),(0,0,255),1)
            cv2.imwrite(os.path.join(self.config.out_dir,'warp_ground_line_'+str(i)+'.png'),warped_line_img)

            match_img=drawmatch_line(self.ground.img,self.footprint.img,best_src_line,best_ref_line)
            cv2.imwrite(os.path.join(self.config.out_dir,"match_{}_score_{:5.4f}_nc_{:5.4f}_cnc_{:5.4f}_dis_{:5.4f}.png".format(i,sim,normal_consis,corr_normal_consis,dis)),match_img)

        # change to 3D transformation, src_pts -> rotate plane to Z axis -> scale init -> center to lefttop bbox -> object to image space -> Line Transformation 3D -> image to object space-> back to ref lefttop bbox -> rotate back to ref
        Trans_line_2D=best_T_final
        scale=np.sqrt(Trans_line_2D[0,0]**2+Trans_line_2D[0,1]**2)
        Rot_3D=np.identity(3)*scale
        Rot_3D[:2,:2]=Trans_line_2D[:2,:2]
        Trans_3D=np.zeros(3)
        Trans_3D[:2]=Trans_line_2D[:2,2]
        Transformation_Line_3D=np.identity(4)
        Transformation_Line_3D[:3,:3]=Rot_3D
        Transformation_Line_3D[:3,3]=Trans_3D
        Trans_object2ortho=np.identity(4)
        Trans_object2ortho[1,1]=-1
        Trans_SRC_LEFTTOP=np.identity(4)
        Trans_SRC_LEFTTOP[0,3]=-self.ground.bound_min[0]
        Trans_SRC_LEFTTOP[1,3]=-self.ground.bound_max[1]
        Trans_REF_LEFTTOP=np.identity(4)
        Trans_REF_LEFTTOP[0,3]=-self.footprint.bound_min[0]
        Trans_REF_LEFTTOP[1,3]=-self.footprint.bound_max[1]
        Trans_SRC_ROTPLANE=np.identity(4)
        Trans_SRC_ROTPLANE[:3,:3]=self.ground.rot_plane
        Trans_REF_ROTPLANE=np.identity(4)
        #Trans_REF_ROTPLANE[:3,:3]=self.footprint.rot_plane
        HEIGHT_SHIFT=np.identity(4)
        HEIGHT_SHIFT[2,3]=self.footprint.z_height-self.ground.z_height*scale
        SRC_SCALE=np.identity(4)
        SRC_SCALE[:3,:3]*=1/self.ground.reso
        REF_SCALE=np.identity(4)
        REF_SCALE[:3,:3]*=self.footprint.reso
        #out_transformation=np.linalg.inv(Trans_REF_ROTPLANE)@HEIGHT_SHIFT@np.linalg.inv(Trans_REF_LEFTTOP)@np.linalg.inv(Trans_object2ortho)@Transformation_Line_3D@Trans_object2ortho@Trans_SRC_LEFTTOP@SCALE_INIT@Trans_SRC_ROTPLANE@self.config.init_trans
        out_transformation=np.linalg.inv(Trans_REF_ROTPLANE)@HEIGHT_SHIFT@np.linalg.inv(Trans_REF_LEFTTOP)@np.linalg.inv(Trans_object2ortho)@REF_SCALE@Transformation_Line_3D@Trans_object2ortho@SRC_SCALE@Trans_SRC_LEFTTOP@Trans_SRC_ROTPLANE@self.config.init_trans
        self.T=out_transformation
        print(self.T)
    
    def align_a2f(self):
        self.config.use_sem=False
        src_line_pixels=self.air.lines_pixels_set.astype(np.int32)
        ref_line_pixels=self.footprint.lines_pixels_set.astype(np.int32)

        src_line_img=np.zeros((self.air.img.shape[0],self.air.img.shape[1]))
        ref_line_img=np.zeros((self.footprint.img.shape[0],self.footprint.img.shape[1]))
        for i in range(src_line_pixels.shape[0]):
            src_line_img[src_line_pixels[i][1],src_line_pixels[i][0]]=1
        for i in range(ref_line_pixels.shape[0]):
            ref_line_img[ref_line_pixels[i][1],ref_line_pixels[i][0]]=1
        src_line_img*=255
        ref_line_img*=255
        for i in range(self.air.lines.shape[0]):
            pt1=self.air.lines[i][0][:2]
            pt2=self.air.lines[i][0][2:]
            mid_pt=(pt1+pt2)/2
            normal_end_pt=mid_pt+self.air.lines_normals[i]*5
            cv2.line(src_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
        for i in range(self.footprint.lines.shape[0]):
            pt1=self.footprint.lines[i][0][:2]
            pt2=self.footprint.lines[i][0][2:]
            mid_pt=(pt1+pt2)/2
            normal_end_pt=mid_pt+self.footprint.lines_normals[i]*5
            cv2.line(ref_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
        cv2.imwrite(os.path.join(self.config.out_dir,"air_line_pixel.png"),src_line_img)
        cv2.imwrite(os.path.join(self.config.out_dir,"footprint_line_pixel.png"),ref_line_img)

        ref_tree=KDTree(ref_line_pixels)

        src_end_pts=[]
        for i in range(self.air.lines.shape[0]):
            src_end_pts.append(self.air.lines[i,0,:2])
            src_end_pts.append(self.air.lines[i,0,2:])
        src_end_pts=np.array(src_end_pts)
        num_src_pts=src_end_pts.shape[0]
        num_ref_lines=self.footprint.lines.shape[0]
        best_sim=-999
        best_src_line=None
        best_ref_line=None
        best_T=np.identity(4)
        best_T_list=[]
        final_score_list=[]
        corr_normal_consis_list=[]
        normal_consis_list=[]
        dis_list=[]
        
        best_src_line_list=[]
        best_ref_line_list=[]

        #print("Total # line seg matches: {}".format(num_src_lines*num_ref_lines))
        cnc_weight=0.1
        dis_weight=0.5
        nc_weight=0.4
        for j in range(num_ref_lines):
            for ii in range(self.air.lines.shape[0]):
                pt1=self.air.lines[ii][0][:2]
                pt2=self.air.lines[ii][0][2:]
                src_line_len=np.linalg.norm(pt1-pt2)
                ref_line_len=np.linalg.norm(self.footprint.lines[j,0,:2]-self.footprint.lines[j,0,2:])
                if src_line_len/ref_line_len<0.3 or src_line_len/ref_line_len>3:
                    continue
                src_line=np.array([pt1[0],pt1[1],pt2[0],pt2[1]])
                src_line_dir=(pt2-pt1)/np.linalg.norm(pt2-pt1)
                R_90=np.array([[0,-1],[1,0]])
                R_90_neg=np.array([[0,1],[-1,0]])
                src_normal1=R_90@src_line_dir
                src_normal2=R_90_neg@src_line_dir
                T1,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_normal1,self.footprint.lines[j,0,:],self.footprint.lines_normals[j])
                T2,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_normal2,self.footprint.lines[j,0,:],self.footprint.lines_normals[j])
                if np.sum(np.isnan(T1))>0 or np.sum(np.isnan(T2))>0:
                    continue
                src_set_trans1=(T1[:2,:2]@src_line_pixels.T+T1[:2,2].reshape(2,1)).T
                src_set_trans2=(T2[:2,:2]@src_line_pixels.T+T2[:2,2].reshape(2,1)).T
                R1=T1[:2,:2]/scale
                R2=T2[:2,:2]/scale
                src_normals_trans1=(R1@self.air.lines_pixels_normals.T).T
                src_normals_trans2=(R2@self.air.lines_pixels_normals.T).T

                d1,i1=ref_tree.query(src_set_trans1,k=1)
                d2,i2=ref_tree.query(src_set_trans2,k=1)

                #normal difference
                ref_pixels1=ref_line_pixels[i1]
                mask=np.linalg.norm(src_set_trans1-ref_pixels1,axis=1)
                mask=mask<3
                match_direction=(src_set_trans1-ref_pixels1)/np.expand_dims(np.linalg.norm(src_set_trans1-ref_pixels1,axis=1),axis=1)
                sim1_match_src=abs(np.sum(match_direction*src_normals_trans1,axis=1))
                sim1_match_ref=abs(np.sum(match_direction*self.footprint.lines_pixels_normals[i1],axis=1))
                sim1_match_src[mask]=1
                sim1_match_ref[mask]=1
                sim1_match_src=np.sum(sim1_match_src*self.air.lines_pixels_weights)
                sim1_match_ref=np.sum(sim1_match_ref*self.air.lines_pixels_weights)
                sim1_src_ref=np.sum(np.sum((self.footprint.lines_pixels_normals[i1]*src_normals_trans1),axis=1)*self.air.lines_pixels_weights)
                dis1=np.sum(d1<12)/d1.shape[0]
                sim1=nc_weight*sim1_src_ref+cnc_weight*(sim1_match_ref+sim1_match_src)/2+dis_weight*dis1
                
                ref_pixels2=ref_line_pixels[i2]
                mask=np.linalg.norm(src_set_trans2-ref_pixels2,axis=1)
                mask=mask<3
                match_direction=(src_set_trans2-ref_pixels2)/np.expand_dims(np.linalg.norm(src_set_trans2-ref_pixels2,axis=1),axis=1)
                sim2_match_src=abs(np.sum(match_direction*src_normals_trans2,axis=1))
                sim2_match_ref=abs(np.sum(match_direction*self.footprint.lines_pixels_normals[i2],axis=1))
                sim2_match_src[mask]=1
                sim2_match_ref[mask]=1
                sim2_match_src=np.sum(sim2_match_src*self.air.lines_pixels_weights)
                sim2_match_ref=np.sum(sim2_match_ref*self.air.lines_pixels_weights)
                sim2_src_ref=np.sum(np.sum((self.footprint.lines_pixels_normals[i2]*src_normals_trans2),axis=1)*self.air.lines_pixels_weights)
                dis2=np.sum(d2<12)/d2.shape[0]
                sim2=nc_weight*sim2_src_ref+cnc_weight*(sim2_match_ref+sim2_match_src)/2+dis_weight*dis2
                
                src_pair=src_line
                ref_pair=self.footprint.lines[j,0,:]
                #best_sim_list.append(sim1)
                final_score_list.append(sim1)
                normal_consis_list.append(sim1_src_ref)
                corr_normal_consis_list.append(0.5*(sim1_match_ref+sim1_match_src))
                dis_list.append(dis1)
                best_T_list.append(T1)
                best_src_line_list.append(src_pair)
                best_ref_line_list.append(ref_pair)

                #best_sim_list.append(sim2)
                final_score_list.append(sim2)
                normal_consis_list.append(sim2_src_ref)
                corr_normal_consis_list.append(0.5*(sim2_match_ref+sim2_match_src))
                dis_list.append(dis2)
                best_T_list.append(T2)
                best_src_line_list.append(src_pair)
                best_ref_line_list.append(ref_pair)
                    
        best_sim_list=sorted(final_score_list,reverse=True)
        sim=best_sim_list[0]
        index=final_score_list.index(sim)
        best_T_final=best_T_list[index]
        for i in range(20):
            sim=best_sim_list[i]
            index=final_score_list.index(sim)
            normal_consis=normal_consis_list[index]
            corr_normal_consis=corr_normal_consis_list[index]
            dis=dis_list[index]
            best_T=best_T_list[index]
            best_src_line=best_src_line_list[index]
            best_ref_line=best_ref_line_list[index]

            warped_lines=self.air.lines.copy()
            warped_lines[:,0,:2]=(best_T[:2,:2]@self.air.lines[:,0,:2].T+best_T[:2,2].reshape(2,1)).T
            warped_lines[:,0,2:]=(best_T[:2,:2]@self.air.lines[:,0,2:].T+best_T[:2,2].reshape(2,1)).T
            warped_line_img=np.zeros((self.footprint.img.shape[0],self.footprint.img.shape[1],3),dtype=self.footprint.img.dtype)
            for ii in range(self.air.lines.shape[0]):
                cv2.line(warped_line_img,warped_lines[ii,0,:2].astype(np.int32),warped_lines[ii,0,2:].astype(np.int32),(0,0,255),1)
            cv2.imwrite(os.path.join(self.config.out_dir,'warp_ground_line_'+str(i)+'.png'),warped_line_img)

            match_img=drawmatch_line(self.air.img,self.footprint.img,best_src_line,best_ref_line)
            cv2.imwrite(os.path.join(self.config.out_dir,"match_{}_score_{:5.4f}_nc_{:5.4f}_cnc_{:5.4f}_dis_{:5.4f}.png".format(i,sim,normal_consis,corr_normal_consis,dis)),match_img)

        # change to 3D transformation, src_pts -> rotate plane to Z axis -> scale init -> center to lefttop bbox -> object to image space -> Line Transformation 3D -> image to object space-> back to ref lefttop bbox -> rotate back to ref
        self.T_a2f_2D=best_T_final
        # Trans_line_2D=best_T_final
        # scale=np.sqrt(Trans_line_2D[0,0]**2+Trans_line_2D[0,1]**2)
        # Rot_3D=np.identity(3)*scale
        # Rot_3D[:2,:2]=Trans_line_2D[:2,:2]
        # Trans_3D=np.zeros(3)
        # Trans_3D[:2]=Trans_line_2D[:2,2]*self.footprint.reso
        # Transformation_Line_3D=np.identity(4)
        # Transformation_Line_3D[:3,:3]=Rot_3D
        # Transformation_Line_3D[:3,3]=Trans_3D
        # Trans_object2ortho=np.identity(4)
        # Trans_object2ortho[1,1]=-1
        # Trans_SRC_LEFTTOP=np.identity(4)
        # Trans_SRC_LEFTTOP[0,3]=-self.air.bound_min[0]
        # Trans_SRC_LEFTTOP[1,3]=-self.air.bound_max[1]
        # Trans_REF_LEFTTOP=np.identity(4)
        # Trans_REF_LEFTTOP[0,3]=-self.footprint.bound_min[0]
        # Trans_REF_LEFTTOP[1,3]=-self.footprint.bound_max[1]
        # Trans_SRC_ROTPLANE=np.identity(4)
        # Trans_SRC_ROTPLANE[:3,:3]=self.air.rot_plane
        # Trans_REF_ROTPLANE=np.identity(4)
        # Trans_REF_ROTPLANE[:3,:3]=self.footprint.rot_plane
        # HEIGHT_SHIFT=np.identity(4)
        # HEIGHT_SHIFT[2,3]=self.footprint.z_height-self.air.z_height*scale
        # SCALE_INIT=np.identity(4)
        # SCALE_INIT[:3,:3]*=self.air.scale_init
        # out_transformation=np.linalg.inv(Trans_REF_ROTPLANE)@HEIGHT_SHIFT@np.linalg.inv(Trans_REF_LEFTTOP)@np.linalg.inv(Trans_object2ortho)@Transformation_Line_3D@Trans_object2ortho@Trans_SRC_LEFTTOP@SCALE_INIT@Trans_SRC_ROTPLANE@self.config.init_trans
        # self.T_a2f=out_transformation

    def align_g2a_w_f(self):
        from skimage.draw import line_aa
        self.align_a2f()
        #self.T_a2f_2D=np.loadtxt(os.path.join(self.config.out_dir,"2d.txt"))

        # find the aoi footprint 
        warped_lines_pixels=(self.T_a2f_2D[:2,:2]@self.air.lines_pixels_set.T+self.T_a2f_2D[:2,2].reshape(2,1)).T
        footprint_aoi_img=np.zeros(self.footprint.img.shape)
        tree=KDTree(warped_lines_pixels)
        lines_init=[]
        lines=[]
        lines_normals=[]
        lines_pixels=[]
        lines_pixels_normals=[]
        pixel_weight_list=[]
        weights=[]
        footprint_aoi_img=np.zeros(self.footprint.img.shape)
        for i in range(self.footprint.lines.shape[0]):
            pt1=self.footprint.lines[i,0,:2]
            pt2=self.footprint.lines[i,0,2:]
            dis1,ind1=tree.query(pt1,k=1)
            dis2,ind2=tree.query(pt2,k=1)
            if dis1<10 or dis2<10:
                cv2.line(footprint_aoi_img,pt1,pt2,255,3,-1)
                lines_init.append(self.footprint.lines[i,0])
                lines_normals.append([self.footprint.lines_normals[i]])
                line_len=np.linalg.norm(np.array(pt1)-np.array(pt2))
                weights.append(line_len)
        weights=np.array(weights)
        weights/=np.sum(weights)
        self.footprint.img=footprint_aoi_img
        cv2.imwrite(os.path.join(self.config.out_dir,"footprint_aoi.png"),footprint_aoi_img)

        for i in range(len(lines_init)):
            pt1=lines_init[i][:2]
            pt2=lines_init[i][2:]
            rr,cc,_=line_aa(pt1[1],pt1[0],pt2[1],pt2[0])
            pts=np.array([cc,rr]).transpose()
            pts_normals=np.ones((rr.shape[0],2))*lines_normals[i]
            pixel_weight=np.ones(pts.shape[0])*weights[i]
            pixel_weight_list.append(pixel_weight)
            lines_pixels.append(pts)
            lines_pixels_normals.append(pts_normals)
            line=np.array([pt1[0],pt1[1],pt2[0],pt2[1]])
            line=np.expand_dims(line,axis=0)
            lines.append(line)
        
        self.footprint.lines_pixels_weights=np.concatenate(pixel_weight_list)
        self.footprint.lines_pixels_set=np.concatenate(lines_pixels,axis=0)
        self.footprint.lines_pixels_normals=np.concatenate(lines_pixels_normals,axis=0)
        lines=np.concatenate(lines,axis=0)
        self.footprint.lines=np.expand_dims(lines,axis=1)
        self.footprint.lines_normals=np.concatenate(lines_normals)

        self.align_g2f()

def write_line3d(lines3d,out_file):
    out=open(out_file,'w',encoding='utf-8')
    for line in lines3d:
        out.write("v {} {} {}\n".format(line[0],line[1],line[2]))
        out.write("v {} {} {}\n".format(line[3],line[4],line[5]))
        out.write("v {} {} {}\n".format(line[6],line[7],line[8]))
    for id,line in enumerate(lines3d):
        out.write("l {} {}\n".format(3*id+1,3*id+3))
    for id in range(0,len(lines3d),2):
        out.write("f {} {} {}\n".format(3*id+2,3*id+3,3*id+5))
        out.write("f {} {} {}\n".format(3*id+5,3*id+6,3*id+3))
    out.close()
             
def cal_reso_pcd(pcd_arr):
   tree=cKDTree(pcd_arr)
   dd,_=tree.query(pcd_arr,k=2)
   mean_dis=np.median(dd[:,1])
   return mean_dis

def plot_boundary(in_pts,in_normals,reso,bound_min=None,bound_max=None):
    if bound_min is None or bound_max is None:
        bound_min=np.min(in_pts[:,:3],axis=0)
        bound_max=np.max(in_pts[:,:3],axis=0)

    x_len_m=bound_max[0]-bound_min[0]
    y_len_m=bound_max[1]-bound_min[1]
    x_len_pix=int(x_len_m/reso)+1
    y_len_pix=int(y_len_m/reso)+1
    img_arr=np.zeros((y_len_pix,x_len_pix),dtype=np.uint8)
    pixel_list=[]
    pixel_normal_list=[]
    for i in range(in_pts.shape[0]):
        pt=in_pts[i,:3]
        pix_x=int((pt[0]-bound_min[0])/reso)
        pix_y=int((bound_max[1]-pt[1])/reso)
        img_arr[pix_y,pix_x]=255
        pixel_list.append([pix_x,pix_y])
        normal_2d=in_normals[i,:2]/np.linalg.norm(in_normals[i,:2])
        normal_2d[1]=-normal_2d[1]
        pixel_normal_list.append(normal_2d)
    pixel_set=np.array(pixel_list)
    pixel_normals=np.array(pixel_normal_list)

    return img_arr,pixel_set,pixel_normals,bound_min,bound_max

def plot_boundary_sem(in_pts,in_sem,reso,bound_min,bound_max,gray=True):
    colormap={0:[0,0,0],
                1:[140,140,140],
              2:[120,120,180],
              3:[200,102,0],
              4:[255,9,112],
              5:[3,200,4],
              6:[126,0,126]}
    x_len_m=bound_max[0]-bound_min[0]
    y_len_m=bound_max[1]-bound_min[1]
    x_len_pix=int(x_len_m/reso)+1
    y_len_pix=int(y_len_m/reso)+1
    img_arr=np.zeros((y_len_pix,x_len_pix),dtype=np.uint8)
    img_color_arr=np.zeros((y_len_pix,x_len_pix,3),dtype=np.uint8)
    for i in range(in_pts.shape[0]):
        pt=in_pts[i,:3]
        pix_x=int((pt[0]-bound_min[0])/reso)
        pix_y=int((bound_max[1]-pt[1])/reso)
        if pix_x<0 or pix_x>=x_len_pix or pix_y<0 or pix_y>=y_len_pix:
            continue
        img_color_arr[pix_y,pix_x,:]=np.array(colormap[int(in_sem[i])])
        img_arr[pix_y,pix_x]=in_sem[i]

    return img_arr,img_color_arr

def plot_boundary_footprint(polygons,gsd=0.5):
    from skimage.draw import line_aa
    from shapely import geometry
    bound_min_x=polygons[0][0][0]
    bound_min_y=polygons[0][0][1]
    bound_max_x=polygons[0][0][0]
    bound_max_y=polygons[0][0][1]

    for polygon in polygons:
        for pt in polygon:
            if pt[0]<bound_min_x:
                bound_min_x=pt[0]
            if pt[0]>bound_max_x:
                bound_max_x=pt[0]
            if pt[1]<bound_min_y:
                bound_min_y=pt[1]
            if pt[1]>bound_max_y:
                bound_max_y=pt[1]
    bound_min=np.array([bound_min_x,bound_min_y,0])
    bound_max=np.array([bound_max_x,bound_max_y,0])
    x_len_m=bound_max_x-bound_min_x
    y_len_m=bound_max_y-bound_min_y
    max_len_m=max(x_len_m,y_len_m)
    x_len_pix=int(x_len_m/gsd)+1
    y_len_pix=int(y_len_m/gsd)+1
    img_arr=np.zeros((y_len_pix,x_len_pix),dtype=np.uint8)

    weights=[]
    for poly in polygons:
        poly_shape=geometry.Polygon(poly[:-1])
        for i in range(len(poly)-1):
            pt0=poly[i]
            pt0_x=int((pt0[0]-bound_min_x)/gsd)
            pt0_y=int((bound_max_y-pt0[1])/gsd)
            pt1=poly[i+1]
            pt1_x=int((pt1[0]-bound_min_x)/gsd)
            pt1_y=int((bound_max_y-pt1[1])/gsd)
            pt0=np.array([pt0_x,pt0_y])
            pt1=np.array([pt1_x,pt1_y])
            line_len=np.linalg.norm(pt0-pt1)
            weights.append(line_len)
    weights=np.array(weights)
    weights/=np.sum(weights)

    pixel_list=[]
    pixel_normal_list=[]
    line_list=[]
    line_normal_list=[]
    pixel_weight_list=[]
    line_id=0
    for poly in polygons:
        poly_shape=geometry.Polygon(poly[:-1])
        for i in range(len(poly)-1):
            pt0=poly[i]
            pt0_x=int((pt0[0]-bound_min_x)/gsd)
            pt0_y=int((bound_max_y-pt0[1])/gsd)
            pt1=poly[i+1]
            pt1_x=int((pt1[0]-bound_min_x)/gsd)
            pt1_y=int((bound_max_y-pt1[1])/gsd)
            rr,cc,val=line_aa(pt0_y,pt0_x,pt1_y,pt1_x)
            img_arr[rr,cc]=255
            pts=np.array([cc,rr]).transpose()
            
            #determine normals
            line_dir=np.array([pt1_x-pt0_x,pt1_y-pt0_y])
            line_dir=line_dir/np.linalg.norm(line_dir)
            R_90=np.array([[0,-1],[1,0]])
            normal1=R_90@line_dir
            normal2=-normal1
            mid_pt=np.array([0.5*(pt0_x+pt1_x),0.5*(pt0_y+pt1_y)])
            test_pt=mid_pt+1*normal1
            test_pt[0]=(test_pt[0]*gsd)+bound_min_x
            test_pt[1]=bound_max_y-(test_pt[1]*gsd)
            test_pt_shape=geometry.Point(test_pt)
            if poly_shape.contains(test_pt_shape):
                line_normal=normal2
                normals=np.ones((rr.shape[0],2))*normal2
            else:
                normals=np.ones((rr.shape[0],2))*normal1
                line_normal=normal1

            #weights
            pixel_weight=np.ones(pts.shape[0])*weights[line_id]
            pixel_weight_list.append(pixel_weight)
            pixel_list.append(pts)
            pixel_normal_list.append(normals)
            line=np.array([pt0_x,pt0_y,pt1_x,pt1_y])
            line=np.expand_dims(line,axis=0)
            line_list.append(line)
            line_normal_list.append([line_normal])
            line_id+=1

    pixel_weights=np.concatenate(pixel_weight_list)
    pixel_set=np.concatenate(pixel_list,axis=0)
    pixel_set_normal=np.concatenate(pixel_normal_list,axis=0)
    lines=np.concatenate(line_list,axis=0)
    lines=np.expand_dims(lines,axis=1)
    line_normals=np.concatenate(line_normal_list)

    return img_arr,pixel_set,pixel_set_normal,lines,line_normals, pixel_weights, bound_min, bound_max

def drawmatch(src_im,ref_im,src_pts,ref_pts):
    if len(src_im.shape)==3:
        src_rgb=src_im
        ref_rgb=ref_im
    else:
        src_rgb=np.zeros((src_im.shape[0],src_im.shape[1],3),np.uint8)
        ref_rgb=np.zeros((ref_im.shape[0],ref_im.shape[1],3),np.uint8)
        src_rgb[src_im!=0]=np.array([0,0,1])*255
        ref_rgb[ref_im!=0]=np.array([0,1,0])*255
    srckp=[cv2.KeyPoint(float(src_pt[0]),float(src_pt[1]),1) for src_pt in src_pts]
    refkp=[cv2.KeyPoint(float(ref_pt[0]),float(ref_pt[1]),1) for ref_pt in ref_pts]
    match=[cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=i,_distance=0) for i in range(len(srckp))]
    outimg=cv2.drawMatches(src_rgb,srckp,ref_rgb,refkp,match,(0,255,0))
    return outimg

def drawmatch_line(src_im,ref_im,src_line,ref_line):
    if len(src_im.shape)==3:
        src_rgb=src_im
        ref_rgb=ref_im
    else:
        src_rgb=np.zeros((src_im.shape[0],src_im.shape[1],3),np.uint8)
        ref_rgb=np.zeros((ref_im.shape[0],ref_im.shape[1],3),np.uint8)
        src_rgb[src_im!=0]=np.array([0,0,1])*255
        ref_rgb[ref_im!=0]=np.array([0,1,0])*255
    src_w=src_im.shape[1]
    src_h=src_im.shape[0]
    ref_w=ref_im.shape[1]
    ref_h=ref_im.shape[0]
    out_h=max(src_h,ref_h)
    out_w=src_w+ref_w
    out_img=np.zeros((out_h,out_w,3))
    out_img[:src_h,:src_w,:]=src_rgb
    out_img[:ref_h,src_w:src_w+ref_w,:]=ref_rgb

    cv2.line(out_img,src_line[:2].astype(np.int32),src_line[2:].astype(np.int32),(255,255,255),3)
    cv2.line(out_img,np.array((ref_line[0]+src_w,ref_line[1])).astype(np.int32),np.array((ref_line[2]+src_w,ref_line[3])).astype(np.int32),(255,255,255),3)

    return out_img

def estimate_smilarity_transform_2pts(src_pt1,src_pt2,ref_pt1,ref_pt2):
    scale=np.linalg.norm(ref_pt2-ref_pt1)/np.linalg.norm(src_pt2-src_pt1)
    src_pt1=scale*src_pt1
    src_pt2=scale*src_pt2
    
    #case1 src_pt1 to ref_pt1
    src_vec=(src_pt2-src_pt1)/np.linalg.norm(src_pt2-src_pt1)
    ref_vec=(ref_pt2-ref_pt1)/np.linalg.norm(ref_pt2-ref_pt1)
    A=np.array([[-src_vec[1],src_vec[0]],[src_vec[0],src_vec[1]]])
    Y=ref_vec
    X=np.linalg.inv(A)@Y # [sin(theta),cos(theta)]
    R1=np.array([[X[1],-X[0]],[X[0],X[1]]])
    t1=ref_pt1-R1@src_pt1
    T1=np.identity(3)
    T1[:2,:2]=R1*scale
    T1[:2,2]=t1

    #case2 src_pt1 to ref_pt2
    src_vec=(src_pt2-src_pt1)/np.linalg.norm(src_pt2-src_pt1)
    ref_vec=(ref_pt1-ref_pt2)/np.linalg.norm(ref_pt1-ref_pt2)
    A=np.array([[-src_vec[1],src_vec[0]],[src_vec[0],src_vec[1]]])
    Y=ref_vec
    X=np.linalg.inv(A)@Y # [sin(theta),cos(theta)]
    R2=np.array([[X[1],-X[0]],[X[0],X[1]]])
    t2=ref_pt2-R2@src_pt1
    T2=np.identity(3)
    T2[:2,:2]=R2*scale
    T2[:2,2]=t2

    # n1=np.linalg.norm(src_pt1-src_pt2)
    # n2=np.linalg.norm(ref_pt1-ref_pt2)
    r1=T1[:2,:2]@src_pt1+T1[:2,2]
    r2=T1[:2,:2]@src_pt2+T1[:2,2]
    r3=T2[:2,:2]@src_pt1+T2[:2,2]
    r4=T2[:2,:2]@src_pt2+T2[:2,2]
    return T1,T2

#lines: nx1x4
def merge_line_seg(lines_raw):
    import math
    R_90=np.array([[0,-1],[1,0]])
    lines=copy.deepcopy(lines_raw)
    #connect two lines
    connect_thresh=5
    for i in range(lines.shape[0]):
        cur_line=lines[i,0,:]
        pt1=cur_line[:2]
        pt2=cur_line[2:]
        cur_line_dir=(pt2-pt1)/np.linalg.norm(pt2-pt1)
        cur_line_dir_perpendicular=R_90@cur_line_dir
        cur_line_distance=abs(np.sum(pt2*cur_line_dir_perpendicular))
        cur_line_distance1=abs(np.sum(pt1*cur_line_dir_perpendicular))
        if pt1[0]==-1 and pt1[1]==-1 and pt2[0]==-1 and pt2[1]==-1:
            continue
        for j in range(i+1,lines.shape[0]):
            tmp_line=lines[j,0,:]
            pt1_t=tmp_line[:2]
            pt2_t=tmp_line[2:]
            tmp_line_dir=(pt2_t-pt1_t)/np.linalg.norm(pt2_t-pt1_t)
            tmp_line_dir_perpendicular=R_90@tmp_line_dir
            tmp_line_distance=abs(np.sum(pt2_t*tmp_line_dir_perpendicular))
            cos_sim=abs(np.sum(cur_line_dir*tmp_line_dir))
            if cos_sim>np.cos(30*180/math.pi) and abs(cur_line_distance-tmp_line_distance)<1: #two lines have same direction and in the same line
                if np.linalg.norm(pt1-pt1_t)<connect_thresh:
                    #pt1=pt2_t
                    lines[i,0,:2]=copy.deepcopy(pt2_t)
                    lines[j,0,:]=np.ones(4)*-1
                if np.linalg.norm(pt1-pt2_t)<connect_thresh:
                    #pt1=pt1_t
                    lines[i,0,:2]=copy.deepcopy(pt1_t)
                    lines[j,0,:]=np.ones(4)*-1
                if np.linalg.norm(pt2-pt1_t)<connect_thresh:
                    #pt2=pt2_t
                    lines[i,0,2:]=copy.deepcopy(pt2_t)
                    lines[j,0,:]=np.ones(4)*-1
                if np.linalg.norm(pt2-pt2_t)<connect_thresh:
                    #pt2=pt1_t
                    lines[i,0,2:]=copy.deepcopy(pt1_t)
                    lines[j,0,:]=np.ones(4)*-1

    #merge two parallel lines
    merge_thresh=5
    for i in range(lines.shape[0]):
        cur_line=lines[i,0,:]
        pt1=cur_line[:2]
        pt2=cur_line[2:]
        cur_line_dir=(pt2-pt1)/np.linalg.norm(pt2-pt1)
        cur_line_dir_perpendicular=R_90@cur_line_dir
        cur_line_distance_symbol=np.sum(pt2*cur_line_dir_perpendicular)
        cur_line_distance=abs(np.sum(pt2*cur_line_dir_perpendicular))
        if pt1[0]==-1 and pt1[1]==-1 and pt2[0]==-1 and pt2[1]==-1:
            continue
        for j in range(i+1,lines.shape[0]):
            tmp_line=lines[j,0,:]
            pt1_t=tmp_line[:2]
            pt2_t=tmp_line[2:]
            tmp_line_dir=(pt2_t-pt1_t)/np.linalg.norm(pt2_t-pt1_t)
            tmp_line_dir_perpendicular=R_90@tmp_line_dir
            tmp_line_distance=abs(np.sum(pt2_t*tmp_line_dir_perpendicular))
            cos_sim=abs(np.sum(cur_line_dir*tmp_line_dir))
            if cos_sim>np.cos(30*180/math.pi) and abs(cur_line_distance-tmp_line_distance)<merge_thresh: #two lines have same direction and in the same line
                pt1_proj=np.sum(pt1*cur_line_dir)
                pt2_proj=np.sum(pt2*cur_line_dir)
                cur_proj_max=max(pt1_proj,pt2_proj)
                cur_proj_min=min(pt1_proj,pt2_proj)
                pt1_t_proj=np.sum(pt1_t*cur_line_dir)
                pt2_t_proj=np.sum(pt2_t*cur_line_dir)
                tmp_proj_max=max(pt1_t_proj,pt2_t_proj)
                tmp_proj_min=min(pt1_t_proj,pt2_t_proj)
                if cur_proj_max<=tmp_proj_min or cur_proj_min>=tmp_proj_max:
                    continue
                intersect_len=abs(min(cur_proj_max,tmp_proj_max)-max(cur_proj_min,tmp_proj_min))
                union_len=abs(max(cur_proj_max,tmp_proj_max)-min(cur_proj_min,tmp_proj_min))
                if intersect_len/union_len>0.5:
                    merged_proj_max=max(cur_proj_max,tmp_proj_max)
                    merged_proj_min=min(cur_proj_min,tmp_proj_min)
                    merged_line_distance=(cur_line_distance+tmp_line_distance)/2
                    if merged_line_distance>0 and cur_line_distance_symbol>0 or merged_line_distance<0 and cur_line_distance_symbol<0:
                        perpendicular_pt=merged_line_distance*cur_line_dir_perpendicular
                    else:
                        perpendicular_pt=-merged_line_distance*cur_line_dir_perpendicular
                    merge_pt1=perpendicular_pt+merged_proj_max*cur_line_dir
                    merge_pt2=perpendicular_pt+merged_proj_min*cur_line_dir
                    lines[i,0,:2]=copy.deepcopy(merge_pt1)
                    lines[i,0,2:]=copy.deepcopy(merge_pt2)
                    lines[j,0,:]=np.ones(4)*-1

    new_lines=[]
    for i in range(lines.shape[0]):
        cur_line=lines[i,0,:]
        pt1=cur_line[:2]
        pt2=cur_line[2:]
        if pt1[0]==-1 and pt1[1]==-1 and pt2[0]==-1 and pt2[1]==-1:
            continue
        new_lines.append(cur_line)

    new_line_lens=[]
    for line in new_lines:
        new_line_lens.append(np.linalg.norm(line[:2]-line[2:]))
    new_line_lens=np.array(new_line_lens)
    mean_line=np.mean(new_line_lens)
    std_line=np.std(new_line_lens)
    line_idx=np.where(new_line_lens>(mean_line-0.5*std_line))[0]

    filter_lines=[]
    for i in range(line_idx.shape[0]):
        filter_lines.append(new_lines[line_idx[i]])

    filter_lines=np.array(filter_lines)
    filter_lines=np.expand_dims(filter_lines,axis=1)
    return filter_lines
             
def slope_angle(x,y):
    x1=x[0]
    if (x==x1).all():
        return 90
    else:
        slope,_,_,_,_=stats.linregress(x,y)
        return np.arctan(slope)*180/np.pi

def cal_distance_map(im):
    dmap=np.zeros(im.shape,np.float64)
    indx,indy=np.where(im!=0)
    indx=np.expand_dims(indx,axis=1)
    indy=np.expand_dims(indy,axis=1)
    ind=np.concatenate([indx,indy],axis=1)
    tree=KDTree(ind)
    indx,indy=np.where(dmap==0)
    indx=np.expand_dims(indx,axis=1)
    indy=np.expand_dims(indy,axis=1)
    im_coords=np.concatenate([indx,indy],axis=1)
    dd,_=tree.query(im_coords,k=1)
    dmap=dd.reshape(dmap.shape)
    return dmap

def estimate_rigid_transform_pt_normal(src_pt,src_slope,ref_pt,ref_slope,scale=1):
    #get rotation
    rotation1=R.from_euler('z',ref_slope-src_slope,degrees=True).as_matrix()[:2,:2]
    rotation2=R.from_euler('z',180-(ref_slope-src_slope),degrees=True).as_matrix()[:2,:2]

    #get translation
    t1=np.array(ref_pt)-rotation1@np.array(src_pt)*scale
    t2=np.array(ref_pt)-rotation2@np.array(src_pt)*scale
    
    #step3: combine t,r
    T1=np.identity(3)
    T1[:2,:2]=rotation1*scale
    T1[:2,2]=t1
    T2=np.identity(3)
    T2[:2,:2]=rotation2*scale
    T2[:2,2]=t2

    r1=T1[:2,:2]@src_pt+T1[:2,2]
    r3=T2[:2,:2]@src_pt+T2[:2,2]
    return T1,T2

#src_line: x1,y1,x2,y2
#return T1,T3
def estimate_rigid_transform_one_lineseg(src_line,ref_line):
    src_pt1=src_line[:2]
    src_pt2=src_line[2:]
    ref_pt1=ref_line[:2]
    ref_pt2=ref_line[2:]

    ##slope: angle between vector and the x-axis
    src_slope = np.arctan((src_pt2[1]-src_pt1[1])/(src_pt2[0]-src_pt1[0]))
    src_slope = src_slope if (src_pt2[0]-src_pt1[0])>=0 else src_slope+np.pi
    ref_slope = np.arctan((ref_pt2[1]-ref_pt1[1])/(ref_pt2[0]-ref_pt1[0]))
    ref_slope = ref_slope if (ref_pt2[0]-ref_pt1[0])>=0 else ref_slope+np.pi
    src_slope=src_slope*180/np.pi
    ref_slope=ref_slope*180/np.pi


    #determine two possible rotation
    R1=R.from_euler('z',ref_slope-src_slope,degrees=True).as_matrix()[:2,:2]

    #case1 src_pt1 to ref_pt1
    t1=ref_pt1-R1@src_pt1
    T1=np.identity(3)
    T1[:2,:2]=R1
    T1[:2,2]=t1

    #case2 src_pt2 to ref_pt2
    t3=ref_pt2-R1@src_pt2
    T3=np.identity(3)
    T3[:2,:2]=R1
    T3[:2,2]=t3

    # n1=np.linalg.norm(src_pt1-src_pt2)
    # n2=np.linalg.norm(ref_pt1-ref_pt2)
    r1=T1[:2,:2]@src_pt1+T1[:2,2]
    r2=T1[:2,:2]@src_pt2+T1[:2,2]
    r5=T3[:2,:2]@src_pt1+T3[:2,2]
    r6=T3[:2,:2]@src_pt2+T3[:2,2]
    return T1,T3

# the line used here is directional line, should be pt1[0]<pt2[0], the slope shoudl be [-90,90]
def estimate_similarity_transform_one_lineseg_origin(src_line,ref_line):
    src_pt1_orig=src_line[:2]
    src_pt2_orig=src_line[2:]
    ref_pt1_orig=ref_line[:2]
    ref_pt2_orig=ref_line[2:]
    src_len=np.linalg.norm(src_pt1_orig-src_pt2_orig)
    ref_len=np.linalg.norm(ref_pt1_orig-ref_pt2_orig)
    scale=ref_len/src_len

    ##slope: angle between vector and the x-axis (-90,90)
    if src_pt2_orig[0]==src_pt1_orig[0]:
        if src_pt2_orig[1]>src_pt1_orig[1]:
            src_pt2=src_pt2_orig
            src_pt1=src_pt1_orig
            src_slope=90
        else:
            src_pt2=src_pt2_orig
            src_pt1=src_pt1_orig
            src_slope=-90
    elif src_pt1_orig[0]<src_pt2_orig[0]:
        src_pt2=src_pt2_orig
        src_pt1=src_pt1_orig
        src_slope = np.arctan((src_pt2[1]-src_pt1[1])/(src_pt2[0]-src_pt1[0]))
        src_slope=src_slope*180/np.pi
    else:
        src_pt2=src_pt1_orig
        src_pt1=src_pt2_orig
        src_slope = np.arctan((src_pt2[1]-src_pt1[1])/(src_pt2[0]-src_pt1[0]))
        src_slope=src_slope*180/np.pi


    if ref_pt2_orig[0]==ref_pt1_orig[0]:
        if ref_pt2_orig[1]>ref_pt1_orig[1]:
            ref_pt2=ref_pt2_orig
            ref_pt1=ref_pt1_orig
            ref_slope=90
        else:
            ref_pt2=ref_pt2_orig
            ref_pt1=ref_pt1_orig
            ref_slope=-90
    elif ref_pt1_orig[0]<ref_pt2_orig[0]:
        ref_pt2=ref_pt2_orig
        ref_pt1=ref_pt1_orig
        ref_slope = np.arctan((ref_pt2[1]-ref_pt1[1])/(ref_pt2[0]-ref_pt1[0]))
        ref_slope=ref_slope*180/np.pi
    else:
        ref_pt2=ref_pt1_orig
        ref_pt1=ref_pt2_orig
        ref_slope = np.arctan((ref_pt2[1]-ref_pt1[1])/(ref_pt2[0]-ref_pt1[0]))
        ref_slope=ref_slope*180/np.pi

    #determine two possible rotation
    R1=R.from_euler('z',ref_slope-src_slope,degrees=True).as_matrix()[:2,:2]
    R1=R1*scale
    R2=R.from_euler('z',ref_slope-src_slope+180,degrees=True).as_matrix()[:2,:2]
    R2=R2*scale

    #case1 src_pt1 to ref_pt1
    t1=ref_pt1-R1@src_pt1
    T1=np.identity(3)
    T1[:2,:2]=R1
    T1[:2,2]=t1

    #case2 src_pt1 to ref_pt2
    t3=ref_pt2-R2@src_pt1
    T3=np.identity(3)
    T3[:2,:2]=R2
    T3[:2,2]=t3

    # n1=np.linalg.norm(src_pt1-src_pt2)
    # n2=np.linalg.norm(ref_pt1-ref_pt2)
    r1=T1[:2,:2]@src_pt1+T1[:2,2]
    r2=T1[:2,:2]@src_pt2+T1[:2,2]
    r5=T3[:2,:2]@src_pt1+T3[:2,2]
    r6=T3[:2,:2]@src_pt2+T3[:2,2]
    return T1,T3,[src_pt1,src_pt2],[ref_pt1,ref_pt2]

# the line used here is directional line, should be pt1[0]<pt2[0], the slope shoudl be [-90,90]
def estimate_similarity_transform_one_lineseg(src_line,src_line_normal,ref_line,ref_line_normal):
    src_pt1_orig=src_line[:2]
    src_pt2_orig=src_line[2:]
    ref_pt1_orig=ref_line[:2]
    ref_pt2_orig=ref_line[2:]
    src_mid_pt=(src_pt1_orig+src_pt2_orig)/2
    ref_mid_pt=(ref_pt1_orig+ref_pt2_orig)/2
    src_line_normal_3d=np.array([src_line_normal[0],src_line_normal[1],0])
    ref_line_normal_3d=np.array([ref_line_normal[0],ref_line_normal[1],0])

    src_len=np.linalg.norm(src_pt1_orig-src_pt2_orig)
    ref_len=np.linalg.norm(ref_pt1_orig-ref_pt2_orig)
    scale=ref_len/src_len

    if (src_line_normal_3d==ref_line_normal_3d).all():
        R1=np.array([[1,0],[0,1]])*scale
        R2=np.array([[-1,0],[0,-1]])*scale
    elif (src_line_normal_3d==-ref_line_normal_3d).all():
        R1=np.array([[-1,0],[0,-1]])*scale
        R2=np.array([[1,0],[0,1]])*scale
    else:
        rot_axis=np.cross(src_line_normal_3d,ref_line_normal_3d)
        if rot_axis[2]>0:
            rot_axis=np.array([0,0,1])
        else:
            rot_axis=np.array([0,0,-1])
        sum1=np.sum(src_line_normal_3d*ref_line_normal_3d)
        sum1=min(sum1,1)
        sum1=max(sum1,0)
        angle1=np.arccos(sum1)*180/np.pi
        angle2=angle1+180
        R1=R.from_rotvec(angle1*rot_axis,degrees=True).as_matrix()[:2,:2]
        R2=R.from_rotvec(angle2*rot_axis,degrees=True).as_matrix()[:2,:2]
        a=R1@src_line_normal
        b=R2@src_line_normal
    R1=R1*scale
    R2=R2*scale
    #case1 src_pt1 to ref_pt1
    t1=ref_mid_pt-R1@src_mid_pt
    T1=np.identity(3)
    T1[:2,:2]=R1
    T1[:2,2]=t1

    #case2 src_pt1 to ref_pt2
    t3=ref_mid_pt-R2@src_mid_pt
    T3=np.identity(3)
    T3[:2,:2]=R2
    T3[:2,2]=t3

    # n1=np.linalg.norm(src_pt1-src_pt2)
    # n2=np.linalg.norm(ref_pt1-ref_pt2)
    r1=T1[:2,:2]@src_pt1_orig+T1[:2,2]
    r2=T1[:2,:2]@src_pt2_orig+T1[:2,2]
    r5=T3[:2,:2]@src_pt1_orig+T3[:2,2]
    r6=T3[:2,:2]@src_pt2_orig+T3[:2,2]
    return T1,T3,scale,[src_mid_pt],[ref_mid_pt]

# the line used here is directional line, should be pt1[0]<pt2[0], the slope shoudl be [-90,90]
def estimate_similarity_transform_one_lineseg_new(src_line,src_line_normal,ref_line,ref_line_normal):
    src_pt1_orig=src_line[:2]
    src_pt2_orig=src_line[2:]
    ref_pt1_orig=ref_line[:2]
    ref_pt2_orig=ref_line[2:]
    src_mid_pt=(src_pt1_orig+src_pt2_orig)/2
    ref_mid_pt=(ref_pt1_orig+ref_pt2_orig)/2
    src_line_normal_3d=np.array([src_line_normal[0],src_line_normal[1],0])
    ref_line_normal_3d=np.array([ref_line_normal[0],ref_line_normal[1],0])

    src_len=np.linalg.norm(src_pt1_orig-src_pt2_orig)
    ref_len=np.linalg.norm(ref_pt1_orig-ref_pt2_orig)
    scale=ref_len/src_len

    if (src_line_normal_3d==ref_line_normal_3d).all():
        R1=np.array([[1,0],[0,1]])*scale
        R2=np.array([[-1,0],[0,-1]])*scale
    elif (src_line_normal_3d==-ref_line_normal_3d).all():
        R1=np.array([[-1,0],[0,-1]])*scale
        R2=np.array([[1,0],[0,1]])*scale
    else:
        rot_axis=np.cross(src_line_normal_3d,ref_line_normal_3d)
        if rot_axis[2]>0:
            rot_axis=np.array([0,0,1])
        else:
            rot_axis=np.array([0,0,-1])
        sum1=np.sum(src_line_normal_3d*ref_line_normal_3d)
        sum1=min(sum1,1)
        sum1=max(sum1,-1)
        angle1=np.arccos(sum1)*180/np.pi
        if rot_axis[2]>0:
            rot_axis=np.array([0,0,1])
            R1=R.from_rotvec(angle1*rot_axis,degrees=True).as_matrix()[:2,:2]
        else:
            rot_axis[2]=1
            angle1=-angle1
            R1=R.from_rotvec(angle1*rot_axis,degrees=True).as_matrix()[:2,:2]
        a=R1@src_line_normal
    R1=R1*scale
    t1=ref_mid_pt-R1@src_mid_pt
    T1=np.identity(3)
    T1[:2,:2]=R1
    T1[:2,2]=t1


    # n1=np.linalg.norm(src_pt1-src_pt2)
    # n2=np.linalg.norm(ref_pt1-ref_pt2)
    r1=T1[:2,:2]@src_pt1_orig+T1[:2,2]
    r2=T1[:2,:2]@src_pt2_orig+T1[:2,2]
    return T1,scale,[src_mid_pt],[ref_mid_pt]

def twopts_matching(src_img,ref_img,config:REG_CONFIG):
    ref_rows,ref_cols=ref_img.shape
    y,x=np.where(src_img!=0)
    src_set=np.concatenate([np.expand_dims(x,axis=1),np.expand_dims(y,axis=1)],axis=1).astype(np.float32)
    y,x=np.where(ref_img!=0)
    ref_set=np.concatenate([np.expand_dims(x,axis=1),np.expand_dims(y,axis=1)],axis=1).astype(np.float32)
    src_tree=KDTree(src_set)
    ref_tree=KDTree(ref_set)

    #find all pairs of 2pts_pair
    src_mat=np.linalg.norm(src_set[np.newaxis, :, :]-src_set[:, np.newaxis, :],axis=2)
    ref_mat=np.linalg.norm(ref_set[np.newaxis, :, :]-ref_set[:, np.newaxis, :],axis=2) ## NxN, each value is the point2point distance
    all_2pts_pairs=[]
    for i in range(config.IM_NUM_BASE_PTS):
        src_id1=np.random.randint(src_set.shape[0])
        top_n=int(0.1*src_set.shape[0])
        src_valid_pair_ids=np.argsort(src_mat[src_id1,:])[src_set.shape[0]-top_n:src_set.shape[0]]
        #src_valid_pair_ids=np.where(src_mat[src_id1,:]>config.IM_MIN_LENGTH)[0]
        src_pt1=src_set[src_id1]
        for j in range(src_valid_pair_ids.shape[0]):
            src_id2=src_valid_pair_ids[j]
            src_pt2=src_set[src_id2]
            length=src_mat[src_id1,src_id2]
            min_length=length*config.IM_MIN_SCALE
            max_length=length*config.IM_MAX_SCALE
            mask1=ref_mat>min_length
            mask2=ref_mat<max_length
            mask=np.logical_and(mask1,mask2)
            y,x=np.where(mask==True)
            for k in range(x.shape[0]):
                ref_pt1=ref_set[x[k]]
                ref_pt2=ref_set[y[k]]
                all_2pts_pairs.append([src_pt1,src_pt2,ref_pt1,ref_pt2])
    print("#pairs of 2pts: {}".format(len(all_2pts_pairs)))

    #estiamte rot,trans
    best_overlap=0
    best_pair=None
    for pair in all_2pts_pairs:
        T1,T2=estimate_smilarity_transform_2pts(pair[0],pair[1],pair[2],pair[3])
        src_trans1=(T1[:2,:2]@src_set.T+T1[:2,2].reshape(2,1)).T
        src_trans2=(T2[:2,:2]@src_set.T+T2[:2,2].reshape(2,1)).T
        d1,_=ref_tree.query(src_trans1,k=1)
        d2,_=ref_tree.query(src_trans2,k=1)
        overlap1=np.mean(d1<config.IM_OVERLAP_THESH)
        overlap2=np.mean(d2<config.IM_OVERLAP_THESH)

        if overlap1> best_overlap:
            best_overlap=overlap1
            best_pair=pair
            best_T=T1
        if overlap2 > best_overlap:
            best_overlap=overlap2
            best_pair=[pair[0],pair[1],pair[3],pair[2]]
            best_T=T2
    print("Final overlap: {}".format(best_overlap))
    warped_srcimg = cv2.warpAffine(src_img,best_T[:2,:],(ref_cols,ref_rows))
    warped_rgb=np.zeros((warped_srcimg.shape[0],warped_srcimg.shape[1],3),np.uint8)
    warped_rgb[warped_srcimg!=0]=np.array([0,0,1])*255
    cv2.imwrite(os.path.join(config.out_dir,'warp_src.png'),warped_srcimg)
    ref_rgb=cv2.cvtColor(ref_img,cv2.COLOR_GRAY2RGB)
    src_rgb=cv2.cvtColor(src_img,cv2.COLOR_GRAY2RGB)
    match_img=drawmatch(src_rgb,ref_rgb,[best_pair[0],best_pair[1]],[best_pair[2],best_pair[3]])
    cv2.imwrite(os.path.join(config.out_dir,'match.png'),match_img)

def pt_normal_ransac(src_img,ref_img,config:REG_CONFIG):
    length=3
    ref_rows,ref_cols=ref_img.shape
    y,x=np.where(src_img!=0)
    src_set=np.concatenate([np.expand_dims(x,axis=1),np.expand_dims(y,axis=1)],axis=1).astype(np.float32)
    y,x=np.where(ref_img!=0)
    ref_set=np.concatenate([np.expand_dims(x,axis=1),np.expand_dims(y,axis=1)],axis=1).astype(np.float32)
    src_tree=KDTree(src_set)
    ref_tree=KDTree(ref_set)

    ref_rgb=np.zeros((ref_img.shape[0],ref_img.shape[1],3),np.uint8)
    ref_rgb[ref_img!=0]=np.array([0,1,0])*255

    #find all pair of 2pts_pair
    all_2pts_pairs=[]
    src_iter_num=5
    for i in range(src_iter_num):
        #gen src_2pts_pair
        src_id=np.random.randint(src_set.shape[0])
        src_pt1=src_set[src_id]
        r=src_tree.query_ball_point(src_pt1,r=length)
        src_neighbor_pts=src_set[r]
        slope=slope_angle(src_neighbor_pts[:,0],src_neighbor_pts[:,1])
        src_pair=[src_pt1,np.arctan(slope)*180/np.pi]
        for j in range(ref_set.shape[0]):
            ref_pt=ref_set[j]
            r=ref_tree.query_ball_point(ref_pt,r=length)
            ref_neighbor_pts=ref_set[r]
            slope=slope_angle(ref_neighbor_pts[:,0],ref_neighbor_pts[:,1])
            ref_pair=[ref_pt,np.arctan(slope)*180/np.pi]
            all_2pts_pairs.append(src_pair+ref_pair)
    print("#pairs of 2pts: {}".format(len(all_2pts_pairs)))

    #estiamte rot,trans
    num_ref_pts=float(ref_set.shape[0])
    best_overlap=0
    best_cost=9999
    best_pair=None
    dmap=cal_distance_map(ref_img)
    for pair_id,pair in enumerate(all_2pts_pairs):
        for scale in np.arange(config.IM_MIN_SCALE,config.IM_MAX_SCALE,config.IM_SCALE_STEP):
            T1,T2=estimate_rigid_transform_pt_normal(pair[0],pair[1],pair[2],pair[3],scale)
            ## metric: overlap
            src_trans1=(T1[:2,:2]@src_set.T+T1[:2,2].reshape(2,1)).T
            src_trans2=(T2[:2,:2]@src_set.T+T2[:2,2].reshape(2,1)).T

            d1,_=ref_tree.query(src_trans1,k=1)
            d2,_=ref_tree.query(src_trans2,k=1)
            error1=np.mean(d1)
            error2=np.mean(d2)
            if error1<best_cost:
                best_cost=error1
                best_pair=pair
                best_T=T1
            if error2<best_cost:
                best_cost=error2
                best_pair=pair
                best_T=T2

    print("Best mean distance: {}".format(best_cost))
    warped_srcimg = cv2.warpAffine(src_img,best_T[:2,:],(ref_cols,ref_rows))
    warped_rgb=np.zeros((warped_srcimg.shape[0],warped_srcimg.shape[1],3),np.uint8)
    warped_rgb[warped_srcimg!=0]=np.array([0,0,1])*255
    cv2.imwrite(os.path.join(config.out_dir,"warpped_src.png"),warped_srcimg)
    match_img=drawmatch(src_img,ref_img,[best_pair[0]],[best_pair[2]])
    cv2.imwrite(os.path.join(config.out_dir,"match.png"),match_img)
    return best_T
    
def line_based_matching_origin(src_facade_pc,ref_facade_pc,config:REG_CONFIG):
    print("#### Start Line-based Matching ####")
    z_axis=np.array([0,0,1])
    src_facade_pts=np.array(src_facade_pc.points)
    src_facade_normals=np.array(src_facade_pc.normals)
    src_z_axis=plane_normal(src_facade_pts)
    ref_facade_pts=np.array(ref_facade_pc.points)
    ref_facade_normals=np.array(ref_facade_pc.normals)
    ref_z_axis=-plane_normal(ref_facade_pts)

    # rotate src to z_axis & generate img
    rot_axis=np.cross(src_z_axis,z_axis)
    rot_axis=rot_axis/np.linalg.norm(rot_axis)
    angle=np.arccos(src_z_axis@z_axis)
    src_rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()
    src_facade_pts_z_corrected=(src_rot_plane@src_facade_pts.T).T
    src_facade_normals_z_corrected=(src_rot_plane@src_facade_normals.T).T
    src_z_height=np.median(src_facade_pts_z_corrected[:,2])
    src_bound_min=np.min(src_facade_pts_z_corrected[:,:3],axis=0)
    src_bound_max=np.max(src_facade_pts_z_corrected[:,:3],axis=0)
    src_x_len=src_bound_max[0]-src_bound_min[0]
    src_y_len=src_bound_max[1]-src_bound_min[1]

    # rotate ref to z_axis & generate img
    rot_axis=np.cross(ref_z_axis,z_axis)
    rot_axis=rot_axis/np.linalg.norm(rot_axis)
    angle=np.arccos(ref_z_axis@z_axis)
    ref_rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()
    ref_facade_pts_z_corrected=(ref_rot_plane@ref_facade_pts.T).T
    ref_facade_normals_z_corrected=(ref_rot_plane@ref_facade_normals.T).T
    ref_z_height=np.median(ref_facade_pts_z_corrected[:,2])

    ##determine the resolution
    ref_bound_min=np.min(ref_facade_pts_z_corrected[:,:3],axis=0)
    ref_bound_max=np.max(ref_facade_pts_z_corrected[:,:3],axis=0)
    ref_x_len=ref_bound_max[0]-ref_bound_min[0]
    ref_y_len=ref_bound_max[1]-ref_bound_min[1]
    max_len=max(ref_x_len,src_x_len,ref_y_len,src_y_len)
    reso=max_len/config.IM_MAX_LENGTH_PIX

    src_img,src_pixel,src_pixel_normals=plot_boundary(src_facade_pts_z_corrected,src_facade_normals_z_corrected,reso)
    ref_img,ref_pixel,ref_pixel_normals=plot_boundary(ref_facade_pts_z_corrected,ref_facade_normals_z_corrected,reso)
    cv2.imwrite(os.path.join(config.out_dir,"src_img.png"),src_img)
    cv2.imwrite(os.path.join(config.out_dir,"ref_img.png"),ref_img)

    # line-based solver
    ref_lines,ref_lines_normals,ref_line_pixels,ref_line_pixel_normals,ref_line_pixel_weights=line_detect(ref_img,ref_pixel,ref_pixel_normals,config,'ref_line')
    src_lines,src_lines_normals,src_line_pixels,src_line_pixel_normals,src_line_pixel_weights=line_detect(src_img,src_pixel,src_pixel_normals,config,'src_line')
    # np.savetxt(r'J:\xuningli\wriva\data\cross_view\pair2\out2\ref_lines.txt',ref_lines[:,0,:])
    # np.savetxt(r'J:\xuningli\wriva\data\cross_view\pair2\out2\src_lines.txt',src_lines[:,0,:])

    Trans_line_2D=line_seg_matching_scale(src_lines,src_lines_normals,src_line_pixels,src_line_pixel_normals,
                                        ref_lines,ref_lines_normals,ref_line_pixels,ref_line_pixel_normals,
                                        src_img,ref_img,config)

    # change to 3D transformation, src_pts -> rotate plane to Z axis -> center to lefttop bbox -> object to image space -> Line Transformation 3D -> image to object space-> back to ref lefttop bbox -> rotate back to ref
    scale=np.sqrt(Trans_line_2D[0,0]**2+Trans_line_2D[0,1]**2)
    Rot_3D=np.identity(3)*scale
    Rot_3D[:2,:2]=Trans_line_2D[:2,:2]
    Trans_3D=np.zeros(3)
    Trans_3D[:2]=Trans_line_2D[:2,2]*reso
    Trans_3D[2]=ref_z_height-src_z_height
    Transformation_Line_3D=np.identity(4)
    Transformation_Line_3D[:3,:3]=Rot_3D
    Transformation_Line_3D[:3,3]=Trans_3D
    Trans_object2ortho=np.identity(4)
    Trans_object2ortho[1,1]=-1
    Trans_SRC_LEFTTOP=np.identity(4)
    Trans_SRC_LEFTTOP[0,3]=-src_bound_min[0]
    Trans_SRC_LEFTTOP[1,3]=-src_bound_max[1]
    Trans_REF_LEFTTOP=np.identity(4)
    Trans_REF_LEFTTOP[0,3]=-ref_bound_min[0]
    Trans_REF_LEFTTOP[1,3]=-ref_bound_max[1]
    Trans_SRC_ROTPLANE=np.identity(4)
    Trans_SRC_ROTPLANE[:3,:3]=src_rot_plane
    Trans_REF_ROTPLANE=np.identity(4)
    Trans_REF_ROTPLANE[:3,:3]=ref_rot_plane

    out_transformation=np.linalg.inv(Trans_REF_ROTPLANE)@np.linalg.inv(Trans_REF_LEFTTOP)@np.linalg.inv(Trans_object2ortho)@Transformation_Line_3D@Trans_object2ortho@Trans_SRC_LEFTTOP@Trans_SRC_ROTPLANE
    return out_transformation

def line_based_matching(config:REG_CONFIG):
    import math
    ## STEP0: preprocess, 1. downsample if too big, 2. remove outlier
    REF_PC=o3d.io.read_point_cloud(config.ref_path)
    num_ref_pts=np.array(REF_PC.points).shape[0]
    if num_ref_pts>config.max_pts:
        sample_every=math.ceil(num_ref_pts/config.max_pts)
        REF_PC=REF_PC.uniform_down_sample(every_k_points=sample_every)
    REF_PC, _ = REF_PC.remove_statistical_outlier(nb_neighbors=5,std_ratio=1)

    SRC_PC=o3d.io.read_point_cloud(config.src_path)
    SRC_PC=SRC_PC.transform(config.init_trans)
    num_src_pts=np.array(SRC_PC.points).shape[0]
    if num_src_pts>config.max_pts:
        sample_every=math.ceil(num_src_pts/config.max_pts)
        SRC_PC=SRC_PC.uniform_down_sample(every_k_points=sample_every)
    SRC_PC, _ = SRC_PC.remove_statistical_outlier(nb_neighbors=5,std_ratio=0.01)
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_preprocessed_line.ply'),SRC_PC)

    ## STEP1: extract building facade, for ref data, first interpolate, and then combine
    ref_x_len,ref_y_len=get_bound_along_plane(REF_PC)
    REF_FACADE_PC,_=extract_facade_part(REF_PC,config)
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'ref_facade_line.ply'),REF_FACADE_PC)
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'ref_preprocessed_line.ply'),REF_PC)
    SRC_FACADE_PC,_=extract_facade_part(SRC_PC,config)
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_facade_line.ply'),SRC_FACADE_PC)

    print("#### Start Line-based Matching ####")
    z_axis=np.array([0,0,1])
    src_facade_pts=np.array(SRC_FACADE_PC.points)
    src_facade_normals=np.array(SRC_FACADE_PC.normals)
    src_z_axis=plane_normal(np.array(SRC_PC.points))
    ref_facade_pts=np.array(REF_FACADE_PC.points)
    ref_facade_normals=np.array(REF_FACADE_PC.normals)
    ref_z_axis=plane_normal(ref_facade_pts)

    # rotate src to z_axis & generate img
    rot_axis=np.cross(src_z_axis,z_axis)
    rot_axis=rot_axis/np.linalg.norm(rot_axis)
    angle=np.arccos(src_z_axis@z_axis)
    src_rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()
    src_facade_pts_z_corrected=(src_rot_plane@src_facade_pts.T).T
    src_facade_normals_z_corrected=(src_rot_plane@src_facade_normals.T).T
    src_z_height=np.median(src_facade_pts_z_corrected[:,2])
    src_bound_min=np.min(src_facade_pts_z_corrected[:,:3],axis=0)
    src_bound_max=np.max(src_facade_pts_z_corrected[:,:3],axis=0)
    src_x_len=src_bound_max[0]-src_bound_min[0]
    src_y_len=src_bound_max[1]-src_bound_min[1]

    # rotate ref to z_axis & generate img
    rot_axis=np.cross(ref_z_axis,z_axis)
    rot_axis=rot_axis/np.linalg.norm(rot_axis)
    angle=np.arccos(ref_z_axis@z_axis)
    ref_rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()
    ref_facade_pts_z_corrected=(ref_rot_plane@ref_facade_pts.T).T
    ref_facade_normals_z_corrected=(ref_rot_plane@ref_facade_normals.T).T
    ref_z_height=np.median(ref_facade_pts_z_corrected[:,2])

    ##determine the resolution
    ref_bound_min=np.min(ref_facade_pts_z_corrected[:,:3],axis=0)
    ref_bound_max=np.max(ref_facade_pts_z_corrected[:,:3],axis=0)
    ref_x_len=ref_bound_max[0]-ref_bound_min[0]
    ref_y_len=ref_bound_max[1]-ref_bound_min[1]
    max_len=max(ref_x_len,src_x_len,ref_y_len,src_y_len)
    reso=max_len/config.IM_MAX_LENGTH_PIX

    src_img,src_pixel,src_pixel_normals=plot_boundary(src_facade_pts_z_corrected,src_facade_normals_z_corrected,reso)
    ref_img,ref_pixel,ref_pixel_normals=plot_boundary(ref_facade_pts_z_corrected,ref_facade_normals_z_corrected,reso)
    cv2.imwrite(os.path.join(config.out_dir,"src_img.png"),src_img)
    cv2.imwrite(os.path.join(config.out_dir,"ref_img.png"),ref_img)

    # line-based solver
    ref_lines,ref_lines_normals,ref_line_pixels,ref_line_pixel_normals,src_line_pixel_weights=line_detect(ref_img,ref_pixel,ref_pixel_normals,config,'ref_line')
    src_lines,src_lines_normals,src_line_pixels,src_line_pixel_normals,src_line_pixel_weights=line_detect(src_img,src_pixel,src_pixel_normals,config,'src_line')

    Trans_line_2D=line_seg_matching_scale(src_lines,src_lines_normals,src_line_pixels,src_line_pixel_normals,
                                        ref_lines,ref_lines_normals,ref_line_pixels,ref_line_pixel_normals,
                                        src_img,ref_img,config)

    # change to 3D transformation, src_pts -> rotate plane to Z axis -> center to lefttop bbox -> object to image space -> Line Transformation 3D -> image to object space-> back to ref lefttop bbox -> rotate back to ref
    scale=np.sqrt(Trans_line_2D[0,0]**2+Trans_line_2D[0,1]**2)
    Rot_3D=np.identity(3)*scale
    Rot_3D[:2,:2]=Trans_line_2D[:2,:2]
    Trans_3D=np.zeros(3)
    Trans_3D[:2]=Trans_line_2D[:2,2]*reso
    Trans_3D[2]=ref_z_height-src_z_height*scale
    Transformation_Line_3D=np.identity(4)
    Transformation_Line_3D[:3,:3]=Rot_3D
    Transformation_Line_3D[:3,3]=Trans_3D
    Trans_object2ortho=np.identity(4)
    Trans_object2ortho[1,1]=-1
    Trans_SRC_LEFTTOP=np.identity(4)
    Trans_SRC_LEFTTOP[0,3]=-src_bound_min[0]
    Trans_SRC_LEFTTOP[1,3]=-src_bound_max[1]
    Trans_REF_LEFTTOP=np.identity(4)
    Trans_REF_LEFTTOP[0,3]=-ref_bound_min[0]
    Trans_REF_LEFTTOP[1,3]=-ref_bound_max[1]
    Trans_SRC_ROTPLANE=np.identity(4)
    Trans_SRC_ROTPLANE[:3,:3]=src_rot_plane
    Trans_REF_ROTPLANE=np.identity(4)
    Trans_REF_ROTPLANE[:3,:3]=ref_rot_plane

    out_transformation=np.linalg.inv(Trans_REF_ROTPLANE)@np.linalg.inv(Trans_REF_LEFTTOP)@np.linalg.inv(Trans_object2ortho)@Transformation_Line_3D@Trans_object2ortho@Trans_SRC_LEFTTOP@Trans_SRC_ROTPLANE@config.init_trans
    return out_transformation

def line_based_matching_sem_origin(config:REG_CONFIG):
    import math
    if not os.path.exists(config.src_sem_path) or not os.path.exists(config.ref_sem_path):
        print(" Line-based matching with semantics failed: No semantic information available")
        return np.identity(4)
    ## STEP0: preprocess, 1. downsample if too big, 2. remove outlier
    SRC_PC=o3d.io.read_point_cloud(config.src_path)
    SRC_PC=SRC_PC.transform(config.init_trans)
    SRC_SEMANTIC=np.loadtxt(config.src_sem_path)
    num_src_pts=np.array(SRC_PC.points).shape[0]
    src_inliers=np.zeros(num_src_pts)
    if num_src_pts>config.max_pts:
        sample_every=math.ceil(num_src_pts/config.max_pts)
        select_arr=np.arange(0,num_src_pts,sample_every)
        src_inliers[select_arr]=True
    else:
        select_arr=np.arange(0,num_src_pts)
        src_inliers[select_arr]=True
    _, src_inliers_ind1 = SRC_PC.remove_statistical_outlier(nb_neighbors=5,std_ratio=2)
    src_inliers1=np.zeros(num_src_pts)
    src_inliers1[src_inliers_ind1]=True
    src_inliers=np.logical_and(src_inliers,src_inliers1)
    SRC_PC=SRC_PC.select_by_index(np.where(src_inliers==True)[0])
    SRC_SEMANTIC=SRC_SEMANTIC[src_inliers]
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_preprocessed_line.ply'),SRC_PC)

    REF_PC=o3d.io.read_point_cloud(config.ref_path)
    REF_SEMANTIC=np.loadtxt(config.ref_sem_path)
    num_ref_pts=np.array(REF_PC.points).shape[0]
    ref_inliers=np.zeros(num_ref_pts)
    if num_ref_pts>config.max_pts:
        sample_every=math.ceil(num_ref_pts/config.max_pts)
        select_arr=np.arange(0,num_ref_pts,sample_every)
        ref_inliers[select_arr]=True
    else:
        select_arr=np.arange(0,num_ref_pts)
        ref_inliers[select_arr]=True
    _, ref_inliers_ind1 = REF_PC.remove_statistical_outlier(nb_neighbors=5,std_ratio=1)
    ref_inliers1=np.zeros(num_ref_pts)
    ref_inliers1[ref_inliers_ind1]=True
    ref_inliers=np.logical_and(ref_inliers,ref_inliers1)
    REF_PC=REF_PC.select_by_index(np.where(ref_inliers==True)[0])
    REF_SEMANTIC=REF_SEMANTIC[ref_inliers]

    ## STEP1: extract building or facade data
    REF_BUILDING_PC,REF_BUILDING_HEIGHT,REF_GROUND_NORMAL=extract_building_part_zaxis(REF_PC,REF_SEMANTIC)
    ref_x_len,ref_y_len=get_bound_along_plane(REF_PC)
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'ref_building.ply'),REF_BUILDING_PC)
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'ref_preprocessed_line.ply'),REF_PC)
    
    SRC_BUILDING_PC,SRC_BUILDING_HEIGHT,SRC_GROUND_NORMAL=extract_building_part_zaxis(SRC_PC,SRC_SEMANTIC)
    SRC_FACADE_PC,_=extract_facade_part(SRC_BUILDING_PC,config,SRC_GROUND_NORMAL)
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_building.ply'),SRC_BUILDING_PC)
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_facade.ply'),SRC_FACADE_PC)

    # STEP2: start line-based matching
    print("#### Start Line-based Matching ####")
    z_axis=np.array([0,0,1])
    src_facade_pts=np.array(SRC_FACADE_PC.points)
    src_facade_normals=np.array(SRC_FACADE_PC.normals)
    src_z_axis=SRC_GROUND_NORMAL
    ref_building_pts=np.array(REF_BUILDING_PC.points)
    ref_building_normals=np.array(REF_BUILDING_PC.normals)
    ref_z_axis=REF_GROUND_NORMAL

    # rotate src to z_axis, scale to ref
    rot_axis=np.cross(src_z_axis,z_axis)
    rot_axis_norm=np.linalg.norm(rot_axis)
    if rot_axis_norm==0:
        rot_axis_norm=1
    rot_axis=rot_axis/rot_axis_norm
    angle=np.arccos(src_z_axis@z_axis)
    src_rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()
    src_facade_pts_z_corrected=(src_rot_plane@src_facade_pts.T).T
    src_facade_normals_z_corrected=(src_rot_plane@src_facade_normals.T).T

    scale_init= REF_BUILDING_HEIGHT/SRC_BUILDING_HEIGHT
    src_facade_pts_z_corrected=src_facade_pts_z_corrected*scale_init

    src_ground_pts=np.array(SRC_PC.points)[np.where(SRC_SEMANTIC==1)[0]]
    src_ground_pts_z_corrected=(src_rot_plane@src_ground_pts.T).T*scale_init
    src_pts_z_corrected=(src_rot_plane@np.array(SRC_PC.points).T).T*scale_init
    src_z_height=np.quantile(src_ground_pts_z_corrected[:,2],0.2)

    src_bound_min=np.min(src_facade_pts_z_corrected[:,:3],axis=0)
    src_bound_max=np.max(src_facade_pts_z_corrected[:,:3],axis=0)
    src_x_len=src_bound_max[0]-src_bound_min[0]
    src_y_len=src_bound_max[1]-src_bound_min[1]

    # rotate ref to z_axis & generate img
    rot_axis=np.cross(ref_z_axis,z_axis)
    rot_axis=rot_axis/np.linalg.norm(rot_axis)
    angle=np.arccos(ref_z_axis@z_axis)
    ref_rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()
    ref_building_pts_z_corrected=(ref_rot_plane@ref_building_pts.T).T
    ref_building_normals_z_corrected=(ref_rot_plane@ref_building_normals.T).T
    ref_ground_pts=np.array(REF_PC.points)[np.where(REF_SEMANTIC==1)[0]]
    ref_ground_pts_z_corrected=(ref_rot_plane@ref_ground_pts.T).T
    ref_z_height=np.quantile(ref_ground_pts_z_corrected[:,2],0.2)
    ref_pts_z_corrected=(ref_rot_plane@np.array(REF_PC.points).T).T

    ##determine the resolution
    ref_bound_min=np.min(ref_building_pts_z_corrected[:,:3],axis=0)
    ref_bound_max=np.max(ref_building_pts_z_corrected[:,:3],axis=0)
    ref_x_len=ref_bound_max[0]-ref_bound_min[0]
    ref_y_len=ref_bound_max[1]-ref_bound_min[1]
    max_len=max(ref_x_len,src_x_len,ref_y_len,src_y_len)
    reso=max_len/config.IM_MAX_LENGTH_PIX

    src_img,src_pixel,src_pixel_normals,src_bound_min,src_bound_max=plot_boundary(src_facade_pts_z_corrected,src_facade_normals_z_corrected,reso)
    cv2.imwrite(os.path.join(config.out_dir,"src_img.png"),src_img)
    ref_img,ref_pixel,ref_pixel_normals,ref_bound_min,ref_bound_max=plot_boundary(ref_building_pts_z_corrected,ref_building_normals_z_corrected,reso)
    cv2.imwrite(os.path.join(config.out_dir,"ref_img.png"),ref_img)
    src_sem_img,src_sem_color_img=plot_boundary_sem(src_pts_z_corrected,SRC_SEMANTIC,reso,src_bound_min,src_bound_max)
    cv2.imwrite(os.path.join(config.out_dir,"src_sem_img.png"),src_sem_img)
    cv2.imwrite(os.path.join(config.out_dir,"src_sem_color_img.png"),src_sem_color_img)
    ref_sem_img,ref_sem_color_img=plot_boundary_sem(ref_pts_z_corrected,REF_SEMANTIC,reso,ref_bound_min,ref_bound_max)
    cv2.imwrite(os.path.join(config.out_dir,"ref_sem_img.png"),ref_sem_img)
    cv2.imwrite(os.path.join(config.out_dir,"ref_sem_color_img.png"),ref_sem_color_img)

    # line-based solver
    ref_lines,ref_lines_normals,ref_line_pixels,ref_line_pixel_normals,ref_line_pixel_weights=line_detect_new(ref_img,ref_pixel,ref_pixel_normals,config,'ref_line')
    src_lines,src_lines_normals,src_line_pixels,src_line_pixel_normals,src_line_pixel_weights=line_detect_new(src_img,src_pixel,src_pixel_normals,config,'src_line')

    Trans_line_2D=line_seg_matching_scale_new(src_lines,src_lines_normals,src_line_pixels,src_line_pixel_normals,src_line_pixel_weights,
                                        ref_lines,ref_lines_normals,ref_line_pixels,ref_line_pixel_normals,ref_line_pixel_weights,
                                        src_img,ref_img,config,True)
    
    # change to 3D transformation, src_pts -> rotate plane to Z axis -> scale init -> center to lefttop bbox -> object to image space -> Line Transformation 3D -> image to object space-> back to ref lefttop bbox -> rotate back to ref
    scale=np.sqrt(Trans_line_2D[0,0]**2+Trans_line_2D[0,1]**2)
    Rot_3D=np.identity(3)*scale
    Rot_3D[:2,:2]=Trans_line_2D[:2,:2]
    Trans_3D=np.zeros(3)
    Trans_3D[:2]=Trans_line_2D[:2,2]*reso
    Transformation_Line_3D=np.identity(4)
    Transformation_Line_3D[:3,:3]=Rot_3D
    Transformation_Line_3D[:3,3]=Trans_3D
    Trans_object2ortho=np.identity(4)
    Trans_object2ortho[1,1]=-1
    Trans_SRC_LEFTTOP=np.identity(4)
    Trans_SRC_LEFTTOP[0,3]=-src_bound_min[0]
    Trans_SRC_LEFTTOP[1,3]=-src_bound_max[1]
    Trans_REF_LEFTTOP=np.identity(4)
    Trans_REF_LEFTTOP[0,3]=-ref_bound_min[0]
    Trans_REF_LEFTTOP[1,3]=-ref_bound_max[1]
    Trans_SRC_ROTPLANE=np.identity(4)
    Trans_SRC_ROTPLANE[:3,:3]=src_rot_plane
    Trans_REF_ROTPLANE=np.identity(4)
    Trans_REF_ROTPLANE[:3,:3]=ref_rot_plane
    HEIGHT_SHIFT=np.identity(4)
    HEIGHT_SHIFT[2,3]=ref_z_height-src_z_height*scale
    SCALE_INIT=np.identity(4)
    SCALE_INIT[:3,:3]*=scale_init
    out_transformation=np.linalg.inv(Trans_REF_ROTPLANE)@HEIGHT_SHIFT@np.linalg.inv(Trans_REF_LEFTTOP)@np.linalg.inv(Trans_object2ortho)@Transformation_Line_3D@Trans_object2ortho@Trans_SRC_LEFTTOP@SCALE_INIT@Trans_SRC_ROTPLANE@config.init_trans
    return out_transformation

def line_seg_matching(src_lines,ref_lines,src_img,ref_img,config:REG_CONFIG):
    ref_rows,ref_cols=ref_img.shape
    y,x=np.where(src_img!=0)
    src_set=np.concatenate([np.expand_dims(x,axis=1),np.expand_dims(y,axis=1)],axis=1).astype(np.float32)
    y,x=np.where(ref_img!=0)
    ref_set=np.concatenate([np.expand_dims(x,axis=1),np.expand_dims(y,axis=1)],axis=1).astype(np.float32)
    src_tree=KDTree(src_set)
    ref_tree=KDTree(ref_set)

    num_src_lines=src_lines.shape[0]
    num_ref_lines=ref_lines.shape[0]
    best_overlap=0
    best_pair=None
    best_T=np.identity(4)
    print("Total # line seg matches: {}".format(num_src_lines*num_ref_lines))
    for i in range(num_src_lines):
        for j in range(num_ref_lines):
            T1,T2=estimate_similarity_transform_one_lineseg(src_lines[i,0,:],ref_lines[j,0,:])
            src_trans1=(T1[:2,:2]@src_set.T+T1[:2,2].reshape(2,1)).T
            src_trans2=(T2[:2,:2]@src_set.T+T2[:2,2].reshape(2,1)).T
            d1,_=ref_tree.query(src_trans1,k=1)
            d2,_=ref_tree.query(src_trans2,k=1)
            overlap1=np.mean(d1<config.IM_OVERLAP_THESH)
            overlap2=np.mean(d2<config.IM_OVERLAP_THESH)

            if overlap1> best_overlap:
                best_overlap=overlap1
                best_T=T1
            if overlap2 > best_overlap:
                best_overlap=overlap2
                best_T=T2
                
    print("Final overlap: {}".format(best_overlap))
    warped_srcimg = cv2.warpAffine(src_img,best_T[:2,:],(ref_cols,ref_rows))
    warped_rgb=np.zeros((warped_srcimg.shape[0],warped_srcimg.shape[1],3),np.uint8)
    warped_rgb[warped_srcimg!=0]=np.array([0,0,1])*255
    cv2.imwrite(os.path.join(config.out_dir,'warp_src.png'),warped_srcimg)
    ref_rgb=cv2.cvtColor(ref_img,cv2.COLOR_GRAY2RGB)
    src_rgb=cv2.cvtColor(src_img,cv2.COLOR_GRAY2RGB)
    return best_T

def get_line_pixel_info(lines,pixel_set,pixel_normals,tag='ref'):
    from skimage.draw import line
    tree=KDTree(pixel_set)
    line_normals_list=[]
    src_pts_list=[]
    src_normal_list=[]
    src_weight_list=[]
    line_lens=[]
    for i in range(lines.shape[0]):
        cur_line=lines[i,0,:]
        pt1=cur_line[:2].astype(np.int32)
        pt2=cur_line[2:].astype(np.int32)
        line_lens.append(np.linalg.norm(pt2-pt1))
    line_lens=np.array(line_lens)
    line_weights=line_lens/np.sum(line_lens)

    for i in range(lines.shape[0]):
        cur_line=lines[i,0,:]
        pt1=cur_line[:2].astype(np.int32)
        pt2=cur_line[2:].astype(np.int32)
        y,x=line(pt1[1],pt1[0],pt2[1],pt2[0])
        if (pt2[0]==pt1[0]):
            if pt2[1]>pt1[1]:
                normal1=np.array([-1,0])
                normal2=np.array([1,0])
            else:
                normal1=np.array([1,0])
                normal2=np.array([-1,0])
        else:
            direction=np.array([pt2[0]-pt1[0],pt2[1]-pt1[1]])
            direction=direction/np.linalg.norm(direction)
            R_90=np.array([[0,-1],[1,0]])
            R_90_neg=np.array([[0,1],[-1,0]])
            normal1=R_90@direction
            normal2=R_90_neg@direction

        src_set=np.concatenate([np.expand_dims(x,axis=1),np.expand_dims(y,axis=1)],axis=1).astype(np.float32)
        dd,ii=tree.query(src_set,k=1)
        line_normals=pixel_normals[ii]
        sim1=np.sum(line_normals@normal1)
        sim2=np.sum(line_normals@normal2)

        if sim1<sim2:
            normal=normal2
        else:
            normal=normal1

        src_normal=np.ones((src_set.shape[0],2))*normal
        src_weights=np.ones(src_set.shape[0])*line_weights[i]
        src_pts_list.append(src_set)
        src_normal_list.append(src_normal)
        src_weight_list.append(src_weights)
        line_normals_list.append(normal)
    src_set=np.concatenate(src_pts_list)
    src_slope=np.concatenate(src_normal_list)
    src_weight=np.concatenate(src_weight_list)
    src_weight/=np.sum(src_weight)
    line_normals=np.array(line_normals_list)
    return line_normals,src_set,src_slope,src_weight

def line_seg_matching_scale(src_lines,src_lines_normals,src_line_pixels,src_line_pixel_normals,
                            ref_lines,ref_lines_normals,ref_line_pixels,ref_line_pixel_normals,
                            src_img,ref_img,config:REG_CONFIG):
    from skimage.draw import line
    ref_cols=ref_img.shape[1]
    ref_rows=ref_img.shape[0]

    src_tree=KDTree(src_line_pixels)
    ref_tree=KDTree(ref_line_pixels)

    num_src_lines=src_lines.shape[0]
    num_ref_lines=ref_lines.shape[0]
    best_sim=-999
    best_src_line=None
    best_ref_line=None
    best_T=np.identity(4)
    best_T_list=[]
    best_sim_list=[]
    best_src_line_list=[]
    best_ref_line_list=[]
    print("Total # line seg matches: {}".format(num_src_lines*num_ref_lines))
    for i in range(num_src_lines):
        for j in range(num_ref_lines):
            T1,T2,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg(src_lines[i,0,:],src_lines_normals[i],ref_lines[j,0,:],ref_lines_normals[j])
            src_set_trans1=(T1[:2,:2]@src_line_pixels.T+T1[:2,2].reshape(2,1)).T
            src_set_trans2=(T2[:2,:2]@src_line_pixels.T+T2[:2,2].reshape(2,1)).T
            R1=T1[:2,:2]/scale
            R2=T2[:2,:2]/scale
            src_normals_trans1=(R1@src_line_pixel_normals.T).T
            src_normals_trans2=(R2@src_line_pixel_normals.T).T
            d1,i1=ref_tree.query(src_set_trans1,k=1)
            d2,i2=ref_tree.query(src_set_trans2,k=1)
            #normal difference
            ref_pixels1=ref_line_pixels[i1]
            match_direction=(src_line_pixels-ref_pixels1)/np.expand_dims(np.linalg.norm(src_line_pixels-ref_pixels1,axis=1),axis=1)
            sim1_match_src=np.sum(abs(np.sum(match_direction*src_normals_trans1,axis=1)))
            sim1_match_ref=np.sum(abs(np.sum(match_direction*ref_line_pixel_normals[i1],axis=1)))
            sim1_src_ref=np.sum(ref_line_pixel_normals[i1]*src_normals_trans1)
            sim1=sim1_src_ref+sim1_match_ref+sim1_match_src

            ref_pixels2=ref_line_pixels[i2]
            match_direction=(src_line_pixels-ref_pixels2)/np.expand_dims(np.linalg.norm(src_line_pixels-ref_pixels2,axis=1),axis=1)
            sim2_match_src=np.sum(abs(np.sum(match_direction*src_normals_trans2,axis=1)))
            sim2_match_ref=np.sum(abs(np.sum(match_direction*ref_line_pixel_normals[i2],axis=1)))
            sim2_src_ref=np.sum(ref_line_pixel_normals[i2]*src_normals_trans2)
            sim2=sim2_src_ref+sim2_match_ref+sim2_match_src

            best_sim_list.append(sim1)
            best_T_list.append(T1)
            best_src_line_list.append(src_pair)
            best_ref_line_list.append(ref_pair)

            best_sim_list.append(sim2)
            best_T_list.append(T2)
            best_src_line_list.append(src_pair)
            best_ref_line_list.append(ref_pair)

    best_sim_list_sorted=sorted(best_sim_list,reverse=True)
    sim=best_sim_list_sorted[0]
    index=best_sim_list.index(sim)
    best_T_final=best_T_list[index]
    for i in range(25):
        sim=best_sim_list_sorted[i]
        index=best_sim_list.index(sim)
        best_T=best_T_list[index]
        best_src_line=best_src_line_list[index]
        best_ref_line=best_ref_line_list[index]

        warped_lines=src_lines.copy()
        warped_lines[:,0,:2]=(best_T[:2,:2]@src_lines[:,0,:2].T+best_T[:2,2].reshape(2,1)).T
        warped_lines[:,0,2:]=(best_T[:2,:2]@src_lines[:,0,2:].T+best_T[:2,2].reshape(2,1)).T
        warped_line_img=np.zeros((ref_img.shape[0],ref_img.shape[1],3),dtype=ref_img.dtype)
        for ii in range(src_lines.shape[0]):
            cv2.line(warped_line_img,warped_lines[ii,0,:2].astype(np.int32),warped_lines[ii,0,2:].astype(np.int32),(0,0,255),3,-1)
        cv2.imwrite(os.path.join(config.out_dir,'warp_src_line_'+str(i)+'.png'),warped_line_img)

        warped_srcimg = cv2.warpAffine(src_img,best_T[:2,:],(ref_cols,ref_rows))
        cv2.imwrite(os.path.join(config.out_dir,'warp_src_'+str(i)+'.png'),warped_srcimg)
        match_img=drawmatch(src_img,ref_img,best_src_line,best_ref_line)
        cv2.imwrite(os.path.join(config.out_dir,"match_"+str(i)+".png"),match_img)

    return best_T_final

def line_seg_matching_scale_new(src_lines,src_lines_normals,src_line_pixels,src_line_pixel_normals,src_line_pixel_weights,
                            ref_lines,ref_lines_normals,ref_line_pixels,ref_line_pixel_normals,ref_line_pixel_weights,
                            src_img,ref_img,config:REG_CONFIG,use_sem=False):
    src_line_pixels=src_line_pixels.astype(np.int32)
    ref_line_pixels=ref_line_pixels.astype(np.int32)
    a=np.sum(src_line_pixel_weights)
    b=np.sum(ref_line_pixel_weights)

    src_line_img=np.zeros((src_img.shape[0],src_img.shape[1]))
    ref_line_img=np.zeros((ref_img.shape[0],ref_img.shape[1]))
    for i in range(src_line_pixels.shape[0]):
        src_line_img[src_line_pixels[i][1],src_line_pixels[i][0]]=1
    for i in range(ref_line_pixels.shape[0]):
        ref_line_img[ref_line_pixels[i][1],ref_line_pixels[i][0]]=1
    src_line_img*=255
    ref_line_img*=255
    for i in range(src_lines.shape[0]):
        pt1=src_lines[i][0][:2]
        pt2=src_lines[i][0][2:]
        mid_pt=(pt1+pt2)/2
        normal_end_pt=mid_pt+src_lines_normals[i]*5
        cv2.line(src_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
    for i in range(ref_lines.shape[0]):
        pt1=ref_lines[i][0][:2]
        pt2=ref_lines[i][0][2:]
        mid_pt=(pt1+pt2)/2
        normal_end_pt=mid_pt+ref_lines_normals[i]*5
        cv2.line(ref_line_img,mid_pt.astype(np.int32),normal_end_pt.astype(np.int32),255,1)
    cv2.imwrite(os.path.join(config.out_dir,"src_line_pixel.png"),src_line_img)
    cv2.imwrite(os.path.join(config.out_dir,"ref_line_pixel.png"),ref_line_img)

    ref_tree=KDTree(ref_line_pixels)

    src_end_pts=[]
    for i in range(src_lines.shape[0]):
        src_end_pts.append(src_lines[i,0,:2])
        src_end_pts.append(src_lines[i,0,2:])
    src_end_pts=np.array(src_end_pts)
    num_src_pts=src_end_pts.shape[0]
    num_ref_lines=ref_lines.shape[0]
    best_sim=-999
    best_src_line=None
    best_ref_line=None
    best_T=np.identity(4)
    best_T_list=[]
    final_score_list=[]
    corr_normal_consis_list=[]
    normal_consis_list=[]
    dis_list=[]
    
    best_src_line_list=[]
    best_ref_line_list=[]

    #print("Total # line seg matches: {}".format(num_src_lines*num_ref_lines))
    cnc_weight=0.1
    dis_weight=0.5
    nc_weight=0.4
    for j in range(num_ref_lines):
        for ii1 in range(num_src_pts):
            for ii2 in range(ii1,num_src_pts):
                pt1=src_end_pts[ii1]
                pt2=src_end_pts[ii2]
                src_line_len=np.linalg.norm(pt1-pt2)
                ref_line_len=np.linalg.norm(ref_lines[j,0,:2]-ref_lines[j,0,2:])
                if src_line_len/ref_line_len<0.3 or src_line_len/ref_line_len>3:
                    continue
                src_line=np.array([pt1[0],pt1[1],pt2[0],pt2[1]])
                src_line_dir=(pt2-pt1)/np.linalg.norm(pt2-pt1)
                R_90=np.array([[0,-1],[1,0]])
                R_90_neg=np.array([[0,1],[-1,0]])
                src_normal1=R_90@src_line_dir
                src_normal2=R_90_neg@src_line_dir
                T1,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_normal1,ref_lines[j,0,:],ref_lines_normals[j])
                T2,scale,src_pair,ref_pair=estimate_similarity_transform_one_lineseg_new(src_line,src_normal2,ref_lines[j,0,:],ref_lines_normals[j])
                if np.sum(np.isnan(T1))>0 or np.sum(np.isnan(T2))>0:
                    continue
                src_set_trans1=(T1[:2,:2]@src_line_pixels.T+T1[:2,2].reshape(2,1)).T
                src_set_trans2=(T2[:2,:2]@src_line_pixels.T+T2[:2,2].reshape(2,1)).T
                R1=T1[:2,:2]/scale
                R2=T2[:2,:2]/scale
                src_normals_trans1=(R1@src_line_pixel_normals.T).T
                src_normals_trans2=(R2@src_line_pixel_normals.T).T

                d1,i1=ref_tree.query(src_set_trans1,k=1)
                d2,i2=ref_tree.query(src_set_trans2,k=1)

                #normal difference
                ref_pixels1=ref_line_pixels[i1]
                mask=np.linalg.norm(src_set_trans1-ref_pixels1,axis=1)
                mask=mask<3
                match_direction=(src_set_trans1-ref_pixels1)/np.expand_dims(np.linalg.norm(src_set_trans1-ref_pixels1,axis=1),axis=1)
                sim1_match_src=abs(np.sum(match_direction*src_normals_trans1,axis=1))
                sim1_match_ref=abs(np.sum(match_direction*ref_line_pixel_normals[i1],axis=1))
                sim1_match_src[mask]=1
                sim1_match_ref[mask]=1
                sim1_match_src=np.sum(sim1_match_src*src_line_pixel_weights)
                sim1_match_ref=np.sum(sim1_match_ref*src_line_pixel_weights)
                sim1_src_ref=np.sum(np.sum((ref_line_pixel_normals[i1]*src_normals_trans1),axis=1)*src_line_pixel_weights)
                dis1=np.sum(d1<12)/d1.shape[0]
                sim1=nc_weight*sim1_src_ref+cnc_weight*(sim1_match_ref+sim1_match_src)/2+dis_weight*dis1
                

                ref_pixels2=ref_line_pixels[i2]
                mask=np.linalg.norm(src_set_trans2-ref_pixels2,axis=1)
                mask=mask<3
                match_direction=(src_set_trans2-ref_pixels2)/np.expand_dims(np.linalg.norm(src_set_trans2-ref_pixels2,axis=1),axis=1)
                sim2_match_src=abs(np.sum(match_direction*src_normals_trans2,axis=1))
                sim2_match_ref=abs(np.sum(match_direction*ref_line_pixel_normals[i2],axis=1))
                sim2_match_src[mask]=1
                sim2_match_ref[mask]=1
                sim2_match_src=np.sum(sim2_match_src*src_line_pixel_weights)
                sim2_match_ref=np.sum(sim2_match_ref*src_line_pixel_weights)
                sim2_src_ref=np.sum(np.sum((ref_line_pixel_normals[i2]*src_normals_trans2),axis=1)*src_line_pixel_weights)
                dis2=np.sum(d2<12)/d2.shape[0]
                sim2=nc_weight*sim2_src_ref+cnc_weight*(sim2_match_ref+sim2_match_src)/2+dis_weight*dis2
                

                src_pair=src_line
                ref_pair=ref_lines[j,0,:]
                #best_sim_list.append(sim1)
                final_score_list.append(sim1)
                normal_consis_list.append(sim1_src_ref)
                corr_normal_consis_list.append(0.5*(sim1_match_ref+sim1_match_src))
                dis_list.append(dis1)
                best_T_list.append(T1)
                best_src_line_list.append(src_pair)
                best_ref_line_list.append(ref_pair)

                #best_sim_list.append(sim2)
                final_score_list.append(sim2)
                normal_consis_list.append(sim2_src_ref)
                corr_normal_consis_list.append(0.5*(sim2_match_ref+sim2_match_src))
                dis_list.append(dis2)
                best_T_list.append(T2)
                best_src_line_list.append(src_pair)
                best_ref_line_list.append(ref_pair)
                
    best_sim_list=sorted(final_score_list,reverse=True)
    sim=best_sim_list[0]
    index=best_sim_list.index(sim)
    best_T_final=best_T_list[index]
    if use_sem:
        ref_sem_arr=cv2.imread(os.path.join(config.out_dir,"ref_sem_img.png"))
        src_sem_arr=cv2.imread(os.path.join(config.out_dir,"src_sem_img.png"))
        ref_sem_color_arr=cv2.imread(os.path.join(config.out_dir,"ref_sem_color_img.png"))
        src_sem_color_arr=cv2.imread(os.path.join(config.out_dir,"src_sem_color_img.png"))
    for i in range(20):
        sim=best_sim_list[i]
        index=final_score_list.index(sim)
        normal_consis=normal_consis_list[index]
        corr_normal_consis=corr_normal_consis_list[index]
        dis=dis_list[index]
        best_T=best_T_list[index]
        best_src_line=best_src_line_list[index]
        best_ref_line=best_ref_line_list[index]

        warped_lines=src_lines.copy()
        warped_lines[:,0,:2]=(best_T[:2,:2]@src_lines[:,0,:2].T+best_T[:2,2].reshape(2,1)).T
        warped_lines[:,0,2:]=(best_T[:2,:2]@src_lines[:,0,2:].T+best_T[:2,2].reshape(2,1)).T
        warped_line_img=np.zeros((ref_img.shape[0],ref_img.shape[1],3),dtype=ref_img.dtype)
        for ii in range(src_lines.shape[0]):
            cv2.line(warped_line_img,warped_lines[ii,0,:2].astype(np.int32),warped_lines[ii,0,2:].astype(np.int32),(0,0,255),1)
        cv2.imwrite(os.path.join(config.out_dir,'warp_src_line_'+str(i)+'.png'),warped_line_img)

        match_img=drawmatch_line(src_img,ref_img,best_src_line,best_ref_line)
        cv2.imwrite(os.path.join(config.out_dir,"match_{}_score_{:5.4f}_nc_{:5.4f}_cnc_{:5.4f}_dis_{:5.4f}.png".format(i,sim,normal_consis,corr_normal_consis,dis)),match_img)

        if use_sem:
            warped_srcimg = cv2.warpAffine(src_sem_arr,best_T[:2,:],(ref_sem_arr.shape[1],ref_sem_arr.shape[0]))   
            warped_srccolorimg = cv2.warpAffine(src_sem_color_arr,best_T[:2,:],(ref_sem_arr.shape[1],ref_sem_arr.shape[0]))    
            valid_ind=warped_srcimg!=0
            same_ind=warped_srcimg==ref_sem_arr
            ind=same_ind*valid_ind
            sem_consis=np.sum(ind)/np.sum(valid_ind)
            cv2.imwrite(os.path.join(config.out_dir,'warp_src_sem_{}_score_{:5.4f}.png'.format(i,sem_consis)),warped_srccolorimg)

    return best_T_final

## return the detected line and line orientation (building inside and outside)
def line_detect(img,pixel_set,pixel_normals,config,tag='1'):
    lsd = cv2.createLineSegmentDetector(1)
    lines = lsd.detect(img)[0]
    ref_lines=lines.copy()
    src_lines=lines.copy()
    drawn_img = lsd.drawSegments(img,lines)

    for i in range(lines.shape[0]):
        line=lines[i,0,:]
        pt1=line[:2].astype(np.int32)
        pt2=line[2:].astype(np.int32)
        cv2.circle(drawn_img,pt1,1,(255,0,0),-1)
        cv2.circle(drawn_img,pt2,1,(255,0,0),-1)
    cv2.imwrite(os.path.join(config.out_dir,tag+'.png'),drawn_img)

    #debug
    #ref_lines=np.array([[142,103,184,97],[197,118,243,114],[244,114,243,95],[243,95,286,92],[286,92,291,158],[291,158,148,166],[148,166,143,102],[159,344,150,203],[150,203,218,197],[217,198,219,239],
    #                    [203,244,212,347],[235,195,293,191],[293,191,305,330],[305,330,262,332],[252,239,262,332],[237,239,234,198],[357,77,368,221],[368,221,402,219]])
    
    # ref_lines=np.array([[189,124,188,189],[189,124,231,124],[231,124,232,144],[287,125,333,125],[333,125,333,193],[333,193,189,191],[186,227,184,366],[188,227,246,229],[246,228,245,270],[330,385,330,227],[263,227,262,271],[332,229,263,227]])
    # src_lines=np.array([[52,220,229,523],[229,523,620,299],[711,244,574,7],[711,244,788,201]])
    # ref_offset=np.expand_dims(np.array([36,25]),axis=0)
    # src_offset=np.expand_dims(np.array([0,-149]),axis=0)

    #ref_lines[:,:2]+=ref_offset
    #ref_lines[:,2:]+=ref_offset
    #src_lines[:,:2]+=src_offset
    #src_lines[:,2:]+=src_offset

    # ref_lines=np.expand_dims(ref_lines,axis=1)
    # src_lines=np.expand_dims(src_lines,axis=1)
    if tag=='ref_line':
        lsd = cv2.createLineSegmentDetector(1)
        mat=np.zeros(img.shape,dtype=img.dtype)
        ref_drawn_img = lsd.drawSegments(mat,ref_lines)
        for i in range(ref_lines.shape[0]):
            line=ref_lines[i,0,:]
            pt1=line[:2].astype(np.int32)
            pt2=line[2:].astype(np.int32)
            cv2.circle(ref_drawn_img,pt1,1,(255,0,0),-1)
            cv2.circle(ref_drawn_img,pt2,1,(255,0,0),-1)
            cv2.line(ref_drawn_img,pt1,pt2,(0,255,0),3,-1)
        cv2.imwrite(os.path.join(config.out_dir,'ref_lines.png'),ref_drawn_img)
        lines=ref_lines
    else:
        lsd = cv2.createLineSegmentDetector(1)
        mat=np.zeros(img.shape,dtype=img.dtype)
        src_drawn_img = lsd.drawSegments(mat,src_lines)
        for i in range(src_lines.shape[0]):
            line=src_lines[i,0,:]
            pt1=line[:2].astype(np.int32)
            pt2=line[2:].astype(np.int32)
            cv2.circle(src_drawn_img,pt1,1,(255,0,0),-1)
            cv2.circle(src_drawn_img,pt2,1,(255,0,0),-1)
            cv2.line(src_drawn_img,pt1,pt2,(0,255,0),3,-1)
        cv2.imwrite(os.path.join(config.out_dir,'src_lines.png'),src_drawn_img)
        lines=src_lines

    line_normals,line_pixel_set,line_pixel_normals,line_pixel_weights=get_line_pixel_info(lines,pixel_set,pixel_normals)
    return lines,line_normals,line_pixel_set,line_pixel_normals,line_pixel_weights

def line_detect_new(img,pixel_set,pixel_normals,config,tag='1'):
    if tag=='ref_line':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(os.path.join(config.out_dir,tag+'_denoise.png'),img)
        lsd = cv2.createLineSegmentDetector(1)
        lines=run_fit_main(img,config.out_dir,90)
        ref_lines=lines.copy()
        drawn_img = lsd.drawSegments(img,lines)
        for i in range(lines.shape[0]):
            line=lines[i,0,:]
            pt1=line[:2].astype(np.int32)
            pt2=line[2:].astype(np.int32)
            cv2.circle(drawn_img,pt1,1,(255,0,0),-1)
            cv2.circle(drawn_img,pt2,1,(255,0,0),-1)
        cv2.imwrite(os.path.join(config.out_dir,tag+'.png'),drawn_img)
        mat=np.zeros(img.shape,dtype=img.dtype)
        ref_drawn_img = lsd.drawSegments(mat,ref_lines)
        for i in range(ref_lines.shape[0]):
            line=ref_lines[i,0,:]
            pt1=line[:2].astype(np.int32)
            pt2=line[2:].astype(np.int32)
            cv2.circle(ref_drawn_img,pt1,1,(255,0,0),-1)
            cv2.circle(ref_drawn_img,pt2,1,(255,0,0),-1)
            cv2.line(ref_drawn_img,pt1,pt2,(0,255,0),3,-1)
        cv2.imwrite(os.path.join(config.out_dir,'ref_lines.png'),ref_drawn_img)
        lines=ref_lines
        line_normals,line_pixel_set,line_pixel_normals,line_pixel_weights=get_line_pixel_info(lines,pixel_set,pixel_normals,tag='ref')
    else:
        lsd = cv2.createLineSegmentDetector(1)
        lines_raw = lsd.detect(img)[0]
        lines=merge_line_seg(lines_raw)
        src_lines=lines.copy()
        drawn_img_raw = lsd.drawSegments(img,lines_raw)
        drawn_img = lsd.drawSegments(img,lines)
        for i in range(lines_raw.shape[0]):
            line=lines_raw[i,0,:]
            pt1=line[:2].astype(np.int32)
            pt2=line[2:].astype(np.int32)
            cv2.circle(drawn_img_raw,pt1,1,(255,0,0),-1)
            cv2.circle(drawn_img_raw,pt2,1,(255,0,0),-1)
        for i in range(lines.shape[0]):
            line=lines[i,0,:]
            pt1=line[:2].astype(np.int32)
            pt2=line[2:].astype(np.int32)
            cv2.circle(drawn_img,pt1,1,(255,0,0),-1)
            cv2.circle(drawn_img,pt2,1,(255,0,0),-1)
        cv2.imwrite(os.path.join(config.out_dir,tag+'.png'),drawn_img)
        cv2.imwrite(os.path.join(config.out_dir,tag+'_raw.png'),drawn_img_raw)

        mat=np.zeros(img.shape,dtype=img.dtype)
        src_drawn_img = lsd.drawSegments(mat,src_lines)
        for i in range(src_lines.shape[0]):
            line=src_lines[i,0,:]
            pt1=line[:2].astype(np.int32)
            pt2=line[2:].astype(np.int32)
            cv2.circle(src_drawn_img,pt1,1,(255,0,0),-1)
            cv2.circle(src_drawn_img,pt2,1,(255,0,0),-1)
            cv2.line(src_drawn_img,pt1,pt2,(0,255,0),3,-1)
        cv2.imwrite(os.path.join(config.out_dir,'src_lines.png'),src_drawn_img)
        lines=src_lines
        line_normals,line_pixel_set,line_pixel_normals,line_pixel_weights=get_line_pixel_info(lines,pixel_set,pixel_normals,tag='src')

    return lines,line_normals,line_pixel_set,line_pixel_normals,line_pixel_weights

def get_utm_fromLatLon(lat, lon):
    import math
    #Special Cases for Norway and Svalbard
    if (lat > 55 and lat < 64 and lon > 2 and lon < 6):
        return 32
    elif (lat > 71 and lon >= 6 and lon < 9):
        return 31
    elif (lat > 71 and ((lon >= 9 and lon < 12) or (lon >= 18 and lon < 21))):
        return 33
    elif (lat > 71 and ((lon >= 21 and lon < 24) or (lon >= 30 and lon < 33))):
        return 35
    # Rest of the world
    elif (lon >= -180 and lon <= 180):
        return (math.floor((lon + 180) / 6) % 60) + 1
    else:
        raise ValueError('Cannot figure out UTM zone from given Lat: {0}, Lon: {1}.'.format(lat, lon))

def parse_kml(in_kml):
    import xml.etree.ElementTree as ET
    from pyproj import Proj
    tree = ET.parse(in_kml)
    root = tree.getroot()
    tag=root.tag.rstrip('kml')
    documents=root.findall("{}Document".format(tag))
    build_polys=[]
    for doc in documents:
        folders=doc.findall('{}Folder'.format(tag))
        if len(folders)==0:
            folders=[doc]
        for folder in folders:
            placemarks=folder.findall('{}Placemark'.format(tag))
            for placemark in placemarks:
                geometrys=placemark.findall('{}MultiGeometry'.format(tag))
                for geometry in geometrys:
                    build=[]
                    coords=geometry[0][0][0][0].text.split(' ')
                    for coord in coords:
                        x=float(coord.split(',')[0])
                        y=float(coord.split(',')[1])
                        build.append([x,y])
                    build_polys.append(build)
                polygons=placemark.findall('{}Polygon'.format(tag))
                for polygon in polygons:
                    build=[]
                    coords=polygon[0][0][0].text.split(' ')
                    for coord in coords:
                        x=float(coord.split(',')[0])
                        y=float(coord.split(',')[1])
                        build.append([x,y])
                    build_polys.append(build)
    zone_number=get_utm_fromLatLon(build_polys[0][0][1],build_polys[0][0][0])
    p = Proj(proj='utm',zone=zone_number,ellps='WGS84', preserve_units=False)
    build_polys_utm=[]
    for build in build_polys:
        build_utm=[]
        for coord in build:
            x,y=p(coord[0],coord[1])
            build_utm.append([x,y])
        build_polys_utm.append(build_utm)
        #print(build_utm)
    return build_polys_utm

def parse_kml_noutm(in_kml):
    import xml.etree.ElementTree as ET
    from pyproj import Proj
    tree = ET.parse(in_kml)
    root = tree.getroot()
    tag=root.tag.rstrip('kml')
    documents=root.findall("{}Document".format(tag))
    build_polys=[]
    for doc in documents:
        folders=doc.findall('{}Folder'.format(tag))
        if len(folders)==0:
            folders=[doc]
        for folder in folders:
            placemarks=folder.findall('{}Placemark'.format(tag))
            for placemark in placemarks:
                geometrys=placemark.findall('{}MultiGeometry'.format(tag))
                for geometry in geometrys:
                    build=[]
                    coords=geometry[0][0][0][0].text.split(' ')
                    for coord in coords:
                        x=float(coord.split(',')[0])
                        y=float(coord.split(',')[1])
                        build.append([x,y])
                    build_polys.append(build)
                polygons=placemark.findall('{}Polygon'.format(tag))
                for polygon in polygons:
                    build=[]
                    coords=polygon[0][0][0].text.split(' ')
                    for coord in coords:
                        x=float(coord.split(',')[0])
                        y=float(coord.split(',')[1])
                        build.append([x,y])
                    build.append(build[0])
                    build_polys.append(build)
    # zone_number=get_utm_fromLatLon(build_polys[0][0][1],build_polys[0][0][0])
    # p = Proj(proj='utm',zone=zone_number,ellps='WGS84', preserve_units=False)
    build_polys_utm=[]
    for build in build_polys:
        build_utm=[]
        for coord in build:
            x,y=coord[0],coord[1]
            build_utm.append([x,y])
        build_polys_utm.append(build_utm)
        #print(build_utm)
    return build_polys_utm
   

def line_based_matching_footprint(config:REG_CONFIG,name):
    import math
    if not os.path.exists(config.out_dir):
        os.mkdir(config.out_dir)
    if not os.path.exists(config.src_sem_path):
        print(" Line-based matching with semantics failed: No semantic information available")
        return np.identity(4)
    ## STEP0: preprocess, 1. downsample if too big, 2. remove outlier
    SRC_PC=o3d.io.read_point_cloud(config.src_path)
    SRC_PC=SRC_PC.transform(config.init_trans)
    SRC_SEMANTIC=np.loadtxt(config.src_sem_path)
    num_src_pts=np.array(SRC_PC.points).shape[0]
    src_inliers=np.zeros(num_src_pts)
    if num_src_pts>config.max_pts:
        sample_every=math.ceil(num_src_pts/config.max_pts)
        select_arr=np.arange(0,num_src_pts,sample_every)
        src_inliers[select_arr]=True
    else:
        select_arr=np.arange(0,num_src_pts)
        src_inliers[select_arr]=True
    _, src_inliers_ind1 = SRC_PC.remove_statistical_outlier(nb_neighbors=5,std_ratio=2)
    src_inliers1=np.zeros(num_src_pts)
    src_inliers1[src_inliers_ind1]=True
    src_inliers=np.logical_and(src_inliers,src_inliers1)
    SRC_PC=SRC_PC.select_by_index(np.where(src_inliers==True)[0])
    SRC_SEMANTIC=SRC_SEMANTIC[src_inliers]
    o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_preprocessed_line.ply'),SRC_PC)

    ## STEP1: extract building or facade data
    if name=='ground':
        SRC_BUILDING_PC,SRC_BUILDING_HEIGHT,SRC_GROUND_NORMAL=extract_building_part_zaxis(SRC_PC,SRC_SEMANTIC)
        SRC_FACADE_PC,_=extract_facade_part(SRC_BUILDING_PC,config,SRC_GROUND_NORMAL)
        o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_building.ply'),SRC_BUILDING_PC)
        o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_facade.ply'),SRC_FACADE_PC)
        z_axis=np.array([0,0,1])
        src_facade_pts=np.array(SRC_FACADE_PC.points)
        src_facade_normals=np.array(SRC_FACADE_PC.normals)
        src_z_axis=SRC_GROUND_NORMAL
        # STEP2: start line-based matching
        print("#### Start Line-based Matching ####")
        # rotate src to z_axis, scale to ref
        rot_axis=np.cross(src_z_axis,z_axis)
        rot_axis_norm=np.linalg.norm(rot_axis)
        if rot_axis_norm==0:
            rot_axis_norm=1
        rot_axis=rot_axis/rot_axis_norm
        angle=np.arccos(src_z_axis@z_axis)
        src_rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()
        src_facade_pts_z_corrected=(src_rot_plane@src_facade_pts.T).T
        src_facade_normals_z_corrected=(src_rot_plane@src_facade_normals.T).T
        src_img,src_pixel,src_pixel_normals,_,_=plot_boundary(src_facade_pts_z_corrected,src_facade_normals_z_corrected,config.footprint_gsd)
        cv2.imwrite(os.path.join(config.out_dir,"src_img.png"),src_img)
        src_lines,src_lines_normals,src_line_pixels,src_line_pixel_normals,src_line_pixel_weights=line_detect_new(src_img,src_pixel,src_pixel_normals,config,'src_line')
    else:
        SRC_BUILDING_PC,SRC_BUILDING_HEIGHT,SRC_GROUND_NORMAL=extract_building_part_zaxis(SRC_PC,SRC_SEMANTIC)
        # o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_building.ply'),SRC_BUILDING_PC)
        # o3d.io.write_point_cloud(os.path.join(config.out_dir,'src_preprocessed_line.ply'),SRC_PC)
        src_building_pts=np.array(SRC_BUILDING_PC.points)
        src_building_normals=np.array(SRC_BUILDING_PC.normals)
        src_z_axis=SRC_GROUND_NORMAL
        z_axis=np.array([0,0,1])
        rot_axis=np.cross(src_z_axis,z_axis)
        rot_axis=rot_axis/np.linalg.norm(rot_axis)
        angle=np.arccos(src_z_axis@z_axis)
        src_rot_plane=R.from_rotvec(angle*rot_axis).as_matrix()
        src_building_pts_z_corrected=(src_rot_plane@src_building_pts.T).T
        src_building_normals_z_corrected=(src_rot_plane@src_building_normals.T).T
        src_ground_pts=np.array(SRC_PC.points)[np.where(SRC_SEMANTIC==1)[0]]
        src_img,src_pixel,src_pixel_normals,src_bound_min,src_bound_max=plot_boundary(src_building_pts_z_corrected,src_building_normals_z_corrected,config.footprint_gsd)
        cv2.imwrite(os.path.join(config.out_dir,"src_img.png"),src_img)
        src_lines,src_lines_normals,src_line_pixels,src_line_pixel_normals,src_line_pixel_weights=line_detect_new(src_img,src_pixel,src_pixel_normals,config,'ref_line')
        os.rename(os.path.join(config.out_dir,"ref_line.png"),os.path.join(config.out_dir,"src_line.png"))
        os.rename(os.path.join(config.out_dir,"ref_line_denoise.png"),os.path.join(config.out_dir,"src_line_denoise.png"))
        os.rename(os.path.join(config.out_dir,"ref_lines.png"),os.path.join(config.out_dir,"src_lines.png"))


    # load footprint kml
    polygons_utm=parse_kml(config.footprint_path)
    ref_img,ref_line_pixels,ref_line_pixel_normals,ref_lines,ref_lines_normals,ref_line_pixel_weights=plot_boundary_footprint(polygons_utm,config.footprint_gsd*5)
    cv2.imwrite(os.path.join(config.out_dir,"footprint_img.png"),ref_img)

    # line-based solver
    #ref_lines,ref_lines_normals,ref_line_pixels,ref_line_pixel_normals,ref_line_pixel_weights=line_detect_new(ref_img,ref_pixel,ref_pixel_normals,config,'ref_line')
    Trans_line_2D=line_seg_matching_scale_new(src_lines,src_lines_normals,src_line_pixels,src_line_pixel_normals,src_line_pixel_weights,
                                        ref_lines,ref_lines_normals,ref_line_pixels,ref_line_pixel_normals,ref_line_pixel_weights,
                                        src_img,ref_img,config)
    
    # change to 3D transformation, src_pts -> rotate plane to Z axis -> scale init -> center to lefttop bbox -> object to image space -> Line Transformation 3D -> image to object space-> back to ref lefttop bbox -> rotate back to ref
    # scale=np.sqrt(Trans_line_2D[0,0]**2+Trans_line_2D[0,1]**2)
    # Rot_3D=np.identity(3)*scale
    # Rot_3D[:2,:2]=Trans_line_2D[:2,:2]
    # Trans_3D=np.zeros(3)
    # Trans_3D[:2]=Trans_line_2D[:2,2]*reso
    # Transformation_Line_3D=np.identity(4)
    # Transformation_Line_3D[:3,:3]=Rot_3D
    # Transformation_Line_3D[:3,3]=Trans_3D
    # Trans_object2ortho=np.identity(4)
    # Trans_object2ortho[1,1]=-1
    # Trans_SRC_LEFTTOP=np.identity(4)
    # Trans_SRC_LEFTTOP[0,3]=-src_bound_min[0]
    # Trans_SRC_LEFTTOP[1,3]=-src_bound_max[1]
    # Trans_REF_LEFTTOP=np.identity(4)
    # Trans_REF_LEFTTOP[0,3]=-ref_bound_min[0]
    # Trans_REF_LEFTTOP[1,3]=-ref_bound_max[1]
    # Trans_SRC_ROTPLANE=np.identity(4)
    # Trans_SRC_ROTPLANE[:3,:3]=src_rot_plane
    # Trans_REF_ROTPLANE=np.identity(4)
    # Trans_REF_ROTPLANE[:3,:3]=ref_rot_plane
    # HEIGHT_SHIFT=np.identity(4)
    # HEIGHT_SHIFT[2,3]=ref_z_height-src_z_height*scale
    # SCALE_INIT=np.identity(4)
    # SCALE_INIT[:3,:3]*=scale_init
    # out_transformation=np.linalg.inv(Trans_REF_ROTPLANE)@HEIGHT_SHIFT@np.linalg.inv(Trans_REF_LEFTTOP)@np.linalg.inv(Trans_object2ortho)@Transformation_Line_3D@Trans_object2ortho@Trans_SRC_LEFTTOP@SCALE_INIT@Trans_SRC_ROTPLANE@config.init_trans
    # return out_transformation
    return 1
 
def draw_shengxi_plygons(poly_json,w,h,out_img_path):
    import json
    in_data=open(poly_json,'r')
    poly_json=json.load(in_data)
    img=np.zeros((h,w),dtype=np.uint8)
    lines=[]
    for line_group in poly_json:
        for line_dr in line_group[:-1]:
            x0 = int(round(line_dr[1]))
            y0 = int(round(line_dr[3]))
            x1 = int(round(line_dr[2]))
            y1 = int(round(line_dr[4]))
            cv2.line(img, (x0, y0), (x1, y1), 255, 5, cv2.LINE_AA)
            lines.append([x0,y0,x1,y1])
    cv2.imwrite(out_img_path, img)

def line_based_matching_sem(config:REG_CONFIG):
    os.makedirs(config.out_dir,exist_ok=True)
    ground=GeoData(config,'ground')
    air=GeoData(config,'air')
    air.plane_rot()
    air.line_extraction()
    ground.plane_rot(air)
    ground.line_extraction()
    line_reg=Line_Matching(ground,air,None,config)
    line_reg.align_g2a_simple()
    np.savetxt(os.path.join
    (config.out_dir,'transformation.txt'),line_reg.T_g2a)
    return line_reg.T_g2a

def line_based_matching_sem_vis(config:REG_CONFIG):
    os.makedirs(config.out_dir,exist_ok=True)
    ground=GeoData(config,'ground')
    air=GeoData(config,'air')
    air.plane_rot()
    air.line_extraction()
    ground.plane_rot(air)
    ground.line_extraction()
    #line_reg=Line_Matching(ground,air,None,config)
    #line_reg.align_g2a()
    #line_reg.align_g2a_simple()
    #line_reg.align_g2a_full()


def line_based_matching_footprint(config:REG_CONFIG):
    air=GeoData(config,'air')
    air.plane_rot()
    air.line_extraction()
    ground=GeoData(config,'ground')
    ground.plane_rot()
    ground.line_extraction()
    footprint=GeoData(config,'footprint')
    footprint.line_extraction()
    line_reg=Line_Matching(ground,air,footprint,config)
    line_reg.align_g2a_w_f()

def line_based_matching_g2f(config:REG_CONFIG):
    # air=GeoData(config,'air')
    # air.plane_rot()
    # air.line_extraction()
    footprint=GeoData(config,'footprint')
    footprint.line_extraction()
    ground=GeoData(config,'ground')
    ground.plane_rot()
    ground.line_extraction()

    line_reg=Line_Matching(ground,None,footprint,config)
    line_reg.align_g2f()
    np.savetxt(os.path.join(config.out_dir,'transformation.txt'),line_reg.T)

def single_test(src,ref,outdir,init_trans,gt_trans):
    ground_path=src
    drone_path=ref
    config=REG_CONFIG()
    config.use_sem=False
    config.sem_label_type='coco'
    config.ref_path=drone_path
    config.src_path=ground_path
    config.out_dir=outdir
    config.init_trans=init_trans
    T=line_based_matching_sem(config)
    print(T)
    np.savetxt(os.path.join(outdir,'transformation.txt'),T)
    src_pc=o3d.io.read_point_cloud(src)
    src_pc1=copy.deepcopy(src_pc)
    src_reg=src_pc.transform(T)
    src_gt=src_pc1.transform(gt_trans)
    src_reg=np.array(src_reg.points)
    src_gt=np.array(src_gt.points)
    err=np.mean(np.linalg.norm(src_reg-src_gt,axis=1))
    return err

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
        

if __name__=="__main__": 
    #DEMO1 (recommend): Ground to Air
    # ground_path=r'J:\xuningli\wriva\wriva-baseline-toolkit\cross_view\pipeline\test_data\g2a\ground.ply'
    # drone_path=r'J:\xuningli\wriva\wriva-baseline-toolkit\cross_view\pipeline\test_data\g2a\air.ply'
    # config=REG_CONFIG()
    # config.sem_label_type='coco'
    # config.ref_path=drone_path
    # config.src_path=ground_path
    # config.out_dir=r'E:\data\tmp\5'
    # T=line_based_matching_sem(config)
    # print(T)

    #DEMO 2: Ground to Footprint
    ground_path=r'E:\data\wriva\ce7\new_pts\4_0\output\ascii_with_labels_binary.ply'
    footprint=r'J:\xuningli\wriva\data\ce7\ortho1\kml\ortho1.kml'
    out_dir=r'E:\data\wriva\ce7\new_pts\4_0\reg2'
    config=REG_CONFIG()
    config.sem_label_type='coco'
    config.src_path=ground_path
    config.out_dir=out_dir
    config.footprint_path=footprint
    line_based_matching_g2f(config)
