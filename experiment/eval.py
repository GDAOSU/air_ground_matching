import glob
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from batch_test import dataset_enumerate,dataset_scale_enumerate
import open3d as o3d
import copy
import matplotlib.pyplot as plt
import math
from matplotlib.scale import ScaleBase
from matplotlib.transforms import Transform
from matplotlib.ticker import FixedLocator, NullFormatter
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.scale as mscale
import matplotlib.pylab as pylab


CENTRE_SCALE=13.98
ZECHE_SCALE=11.01
APL_SCALE=1

LABEL_SIZE=18
TITLE_SIZE=20
TICK_SIZE=14
MS=8
LW=3
ELW=3
T_COLOR=[255/255,102/255,102/255]
T_MK_COLOR=[255/255,51/255,51/255]
O_COLOR=[178/255,102/255,1]
O_MK_COLOR=[153/255,51/255,1]
OURS_COLOR=[102/255,178/255,1]
OURS_MK_COLOR=[51/255,153/255,255/255]
params = {
         'axes.labelsize': '{}'.format(LABEL_SIZE),
         'axes.titlesize':'{}'.format(TITLE_SIZE),
         'xtick.labelsize':'{}'.format(TICK_SIZE),
         'ytick.labelsize':'{}'.format(TICK_SIZE)}
pylab.rcParams.update(params)
class CustomScale(ScaleBase):
    name = 'scale'
    
    def __init__(self, axis, **kwargs):
        super().__init__(axis)
    
    def get_transform(self):
        return self.CustomTransform()
    
    def set_default_locators_and_formatters(self, axis):
        # Major ticks
        major_ticks = [0, 5, 20, 50, 3200]
        # Minor ticks
        minor_ticks = np.concatenate([
            np.linspace(0, 5, 3),
            np.linspace(5, 20, 5)[1:],
            np.linspace(20, 50, 10)[1:],
            np.linspace(50, 3200, 20)[1:]
        ])
        
        axis.set_major_locator(FixedLocator(major_ticks))
        axis.set_major_formatter(FuncFormatter(self.format_major_ticks))
        
        axis.set_minor_locator(FixedLocator(minor_ticks))
        axis.set_minor_formatter(FuncFormatter(self.format_minor_ticks))
    
    def format_major_ticks(self, val, pos):
        return f'{val:g}'
    
    def format_minor_ticks(self, val, pos):
        return f'{val:g}' if val in [0, 5, 20, 50, 3200] else ''
    
    class CustomTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        
        def transform_non_affine(self, a):
            a = np.array(a)
            mask_1_0=(a >= -1) & (a <= 0)
            mask_0_5 = (a > 0) & (a <= 5)
            mask_5_20 = (a > 5) & (a <= 20)
            mask_20_50 = (a > 20) & (a <= 50)
            mask_50_3000 = (a > 50) & (a <= 3200)
            
            b = np.empty_like(a)
            b[mask_1_0] = a[mask_1_0]+1
            b[mask_0_5] = 2*a[mask_0_5]+1
            b[mask_5_20] = 10*(a[mask_5_20] - 5) / 15 + 11
            b[mask_20_50] = 10*(a[mask_20_50] - 20) / 30 + 21
            b[mask_50_3000] = 10*(a[mask_50_3000] - 50) / 3150 + 31
            
            return b
        
        def inverted(self):
            return CustomScale.InvertedCustomTransform()
    
    class InvertedCustomTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        
        def transform_non_affine(self, a):
            a = np.array(a)
            mask_0_10 = (a >= 0) & (a <= 10)
            mask_10_20 = (a > 10) & (a <= 20)
            mask_20_50 = (a > 20) & (a <= 50)
            mask_50_3000 = (a > 50) & (a <= 3200)
            
            b = np.empty_like(a)
            b[mask_0_10] = a[mask_0_10]
            b[mask_10_20] = 10*(a[mask_10_20] - 10) / 10 + 10
            b[mask_20_50] = 10*(a[mask_20_50] - 20) / 30 + 20
            b[mask_50_3000] = 10*(a[mask_50_3000] - 50) / 3150 + 30
            
            return b


class OverScale(ScaleBase):
    name = 'over'
    
    def __init__(self, axis, **kwargs):
        super().__init__(axis)
    
    def get_transform(self):
        return self.CustomTransform()
    
    def set_default_locators_and_formatters(self, axis):
        # Major ticks
        major_ticks = [0, 5, 10, 200]
        # Minor ticks
        minor_ticks = np.concatenate([
            np.linspace(0, 5, 5),
            np.linspace(5, 10, 5)[1:],
            np.linspace(10, 200, 10)[1:]
        ])
        
        axis.set_major_locator(FixedLocator(major_ticks))
        axis.set_major_formatter(FuncFormatter(self.format_major_ticks))
        
        axis.set_minor_locator(FixedLocator(minor_ticks))
        axis.set_minor_formatter(FuncFormatter(self.format_minor_ticks))
    
    def format_major_ticks(self, val, pos):
        return f'{val:g}'
    
    def format_minor_ticks(self, val, pos):
        return f'{val:g}' if val in [0, 5, 10, 200] else ''
    
    class CustomTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        
        def transform_non_affine(self, a):
            a = np.array(a)
            mask_0_5 = (a > 0) & (a <= 5)
            mask_5_10 = (a > 5) & (a <= 10)
            mask_10_200 = (a > 10) & (a <= 200)
            
            b = np.empty_like(a)
            b[mask_0_5] = 2*a[mask_0_5]
            b[mask_5_10] = 2*a[mask_5_10]
            b[mask_10_200] = 10*(a[mask_10_200] - 10) / 190 + 10
            return b
        
        def inverted(self):
            return CustomScale.InvertedCustomTransform()
    
    class InvertedCustomTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        
        def transform_non_affine(self, a):
            a = np.array(a)
            mask_0_5 = (a >= 0) & (a <= 5)
            mask_5_10 = (a >= 5) & (a <= 10)
            mask_10_200 = (a > 10) & (a <= 200)
            
            b = np.empty_like(a)
            b[mask_0_5] = 2*a[mask_0_5]
            b[mask_5_10] = 2*a[mask_5_10]
            b[mask_10_200] = 10*(a[mask_10_200] - 10) / 190 + 10
            return b
#mscale.register_scale(OverScale)

def cal_over():
    from scipy.spatial import KDTree
    #eval teaser++
    pair_list=dataset_enumerate()
    for pair in pair_list:
        uav=pair[0]
        ground=pair[1]
        if 'apl' in uav:
            thresh=1
        else:
            thresh=0.3
        src_pc=o3d.io.read_point_cloud(ground).voxel_down_sample(thresh/2)
        ref_pc=o3d.io.read_point_cloud(uav).voxel_down_sample(thresh/2)
        tree=KDTree(np.array(src_pc.points))

        dd,ii=tree.query(np.array(ref_pc.points))
        print("Prcess {} {},overlap:{}".format(os.path.basename(uav),os.path.basename(ground),np.mean(dd<thresh)))

def cal_acc(trans_est,trans_gt,src_pc,scale):
    src_gt=np.array(src_pc.points)
    src_pc1=copy.deepcopy(src_pc)
    src_reg1=src_pc1.transform(trans_est)
    src_reg1=np.array(src_reg1.points)
    rmse=np.mean(np.linalg.norm(src_reg1-src_gt,axis=1)) 
    R_est=trans_est[:3,:3]
    s_est=math.sqrt((R_est@R_est.T)[0,0])
    R_est=R_est/s_est
    t_est=trans_est[:3,3]
    re=np.degrees(np.arccos(0.5*(np.trace(R_est.T@trans_gt[:3,:3])-1)))
    te=np.linalg.norm(t_est)
    se=s_est
    return rmse*scale,re,te*scale,s_est

def eval_acc():
    init_scale=1.5
    res_dir=r'J:\xuningli\papers\g2a\data\results\acc_result'
    init_trans=np.identity(4)
    rot=R.from_euler('xyz',[90,0,90],degrees=True).as_matrix()
    init_trans[:3,:3]=rot*init_scale
    init_trans[2,3]=5
    print(init_trans)
    np.savetxt(r'J:\xuningli\papers\g2a\data\results\data\init_trans.txt',init_trans)
    #eval teaser++
    pair_list=dataset_enumerate()
    teaser_res=[]
    teaser_star_res=[]
    opransac_res=[]
    opransac_star_res=[]
    our_res=[]
    for pair in pair_list:

        uav=pair[0]
        ground=pair[1]
        uav_name=os.path.basename(uav)
        ground_name=os.path.basename(ground)

        scale=1
        # scale=CENTRE_SCALE
        # if 'zeche' in ground:
        #     scale=ZECHE_SCALE
        # elif 'apl' in ground:
        #     scale=APL_SCALE

        teaser_result=glob.glob(os.path.join(res_dir,"yoho_1.5_teaser","{}_{}*".format(ground_name,uav_name)))[0]
        teaser_trans=np.loadtxt(teaser_result)
        opransace_result=glob.glob(os.path.join(res_dir,"yoho_1.5_opransac","{}_{}*".format(ground_name,uav_name)))[0]
        opransace_trans=np.loadtxt(opransace_result)

        our_result=glob.glob(os.path.join(res_dir,"our_1.5","{}_{}*".format(ground_name,uav_name)))[0]
        our_trans=np.loadtxt(our_result)
        src_pc=o3d.io.read_point_cloud(ground)
        src_gt=np.array(src_pc.points)

        rmse1,re1,te1,s1=cal_acc(teaser_trans@init_trans,np.identity(4),src_pc,scale)
        rmse2,re2,te2,s2=cal_acc(opransace_trans@init_trans,np.identity(4),src_pc,scale)
        rmse3,re3,te3,s3=cal_acc(our_trans,np.identity(4),src_pc,scale)

        print("Prcess {} (RMSE,RE,TE,S_est)\n, teaser: {:.3f},{:.3f},{:.3f},{:.3f}\n opransac:{:.3f},{:.3f},{:.3f},{:.3f} \n our: {:.3f},{:.3f},{:.3f},{:.3f}".format(ground_name,rmse1,re1,te1,s1,rmse2,re2,te2,s2,rmse3,re3,te3,s3))

def eval_acc_new():
    init_scale=1.5
    res_dir=r'J:\xuningli\papers\g2a\data\results\scale_result'
    init_trans=np.identity(4)
    rot=R.from_euler('xyz',[90,0,90],degrees=True).as_matrix()
    init_trans[:3,:3]=rot*init_scale
    init_trans[2,3]=5
    print(init_trans)
    np.savetxt(r'J:\xuningli\papers\g2a\data\results\data\init_trans.txt',init_trans)
    #eval teaser++
    pair_list=dataset_scale_enumerate()
    teaser_res=[]
    opransac_res=[]
    our_res=[]

    for pair in pair_list:
        uav=pair[0]
        ground=pair[1]
        uav_name=os.path.basename(uav)
        ground_name=os.path.basename(ground)

        scale=CENTRE_SCALE
        if 'zeche' in ground:
            scale=ZECHE_SCALE
        elif 'apl' in ground:
            scale=APL_SCALE

        teaser_result=glob.glob(os.path.join(res_dir,"yoho_teaser","{}_{}*{:.2f}_teaser.txt".format(ground_name,uav_name,init_scale)))[0]
        teaser_trans=np.loadtxt(teaser_result)
        opransace_result=glob.glob(os.path.join(res_dir,"yoho_opransac","{}_{}*{:.2f}_opransac.txt".format(ground_name,uav_name,init_scale)))[0]
        opransace_trans=np.loadtxt(opransace_result)
        our_results=glob.glob(os.path.join(res_dir,"our_test","{}_{}*".format(ground_name,uav_name)))
        our_result=None
        for res in our_results:
            if float(os.path.basename(res)[:-8].split("_")[-1])==init_scale:
                our_result=res
        our_trans=np.loadtxt(our_result)

        src_pc=o3d.io.read_point_cloud(ground)
        src_gt=np.array(src_pc.points)

        rmse1,re1,te1,s1=cal_acc(teaser_trans@init_trans,np.identity(4),src_pc,scale)
        rmse2,re2,te2,s2=cal_acc(opransace_trans@init_trans,np.identity(4),src_pc,scale)
        rmse3,re3,te3,s3=cal_acc(our_trans,np.identity(4),src_pc,scale)

        print("Prcess {} (RMSE,RE,TE,S_est)\n, teaser: {:.3f},{:.3f},{:.3f},{:.3f}\n opransac:{:.3f},{:.3f},{:.3f},{:.3f} \n our: {:.3f},{:.3f},{:.3f},{:.3f}".format(ground_name,rmse1,re1,te1,s1,rmse2,re2,te2,s2,rmse3,re3,te3,s3))

def eval_scale():

    res_dir=r'J:\xuningli\papers\g2a\data\results\scale_result'
    scale_list=[1,1.25,1.5,1.75,2,3,5,7.5,10]
    #scale_list=[1,5,10,15]
    #scale_list2=[5,7.5,10,15]
    #eval teaser++
    pair_list=dataset_scale_enumerate()


    fig, axs = plt.subplots(3, 3,sharex=True,sharey=True)
    axs_list=axs.flat
    subtitles=['ISPRS_Multi Center Obelisk', 'ISPRS_Multi Center Rathaus', 'ISPRS_Multi Center Stadthaus','ISPRS_Multi Zeche Lohnhalle','ISPRS_Multi Zeche Pferdestall','ISPRS_Multi Zeche Verwaltung','APL MP2','APL MP4','APL MP5']
    for id,pair in enumerate(pair_list):
        uav=pair[0]
        ground=pair[1]
        uav_name=os.path.basename(uav)
        ground_name=os.path.basename(ground)

        teaser_res=[]
        opransac_res=[]
        our_res=[]
        print("cal:{}".format(ground_name))
        src_pc=o3d.io.read_point_cloud(ground)
        src_gt=np.array(src_pc.points)
        if "mp5" in ground_name:
            a=1
        for scale in scale_list:
            init_trans=np.identity(4)
            rot=R.from_euler('xyz',[90,0,90],degrees=True).as_matrix()
            init_trans[:3,:3]=rot*scale
            init_trans[2,3]=5

            teaser_result=glob.glob(os.path.join(res_dir,"yoho_teaser","{}_{}*{:.2f}_teaser.txt".format(ground_name,uav_name,scale)))[0]
            teaser_trans=np.loadtxt(teaser_result)
            # teaser_star_result=glob.glob(os.path.join(res_dir,"yoho_star_teaser","{}_{}*{:.2f}_teaser.txt".format(ground_name,uav_name,scale)))[0]
            # teaser_star_trans=np.loadtxt(teaser_star_result)
            opransace_result=glob.glob(os.path.join(res_dir,"yoho_opransac","{}_{}*{:.2f}_opransac.txt".format(ground_name,uav_name,scale)))[0]
            opransace_trans=np.loadtxt(opransace_result)
            # opransace_star_result=glob.glob(os.path.join(res_dir,"yoho_star_opransac","{}_{}*{:.2f}_opransac.txt".format(ground_name,uav_name,scale)))[0]
            # opransace_star_trans=np.loadtxt(opransace_star_result)

            our_results=glob.glob(os.path.join(res_dir,"our_test","{}_{}*".format(ground_name,uav_name)))
            our_result=None
            for res in our_results:
                if float(os.path.basename(res)[:-8].split("_")[-1])==scale:
                    our_result=res
            our_trans=np.loadtxt(our_result)

            # our_results_new=glob.glob(os.path.join(res_dir,"our_test","{}_{}*".format(ground_name,uav_name)))
            # our_result_new=None
            # for res in our_results_new:
            #     if float(os.path.basename(res)[:-8].split("_")[-1])==scale:
            #         our_result_new=res
            # our_trans_new=np.loadtxt(our_result_new)

            src_pc1=copy.deepcopy(src_pc)
            src_reg1=src_pc1.transform(teaser_trans@init_trans)
            src_reg1=np.array(src_reg1.points)

            src_pc2=copy.deepcopy(src_pc)
            src_reg2=src_pc2.transform(opransace_trans@init_trans)
            src_reg2=np.array(src_reg2.points)


            src_pc3=copy.deepcopy(src_pc)
            src_reg3=src_pc3.transform(our_trans)
            src_reg3=np.array(src_reg3.points)

            if 'zeche' in ground_name:
                scale1=ZECHE_SCALE
            elif 'center' in ground_name:
                scale1=CENTRE_SCALE
            else:
                scale1=APL_SCALE


            err1=np.mean(np.linalg.norm(src_reg1-src_gt,axis=1))*scale1
            #err11=np.mean(np.linalg.norm(src_reg11-src_gt,axis=1))
            err2=np.mean(np.linalg.norm(src_reg2-src_gt,axis=1))*scale1
            #err22=np.mean(np.linalg.norm(src_reg22-src_gt,axis=1))
            err3=np.mean(np.linalg.norm(src_reg3-src_gt,axis=1))*scale1
            #err4=np.mean(np.linalg.norm(src_reg4-src_gt,axis=1))
            teaser_res.append(err1)
            #teaser_star_res.append(err11)
            opransac_res.append(err2)
            #opransac_star_res.append(err22)
            our_res.append(err3)
            #our_new_res.append(err4)

            #print("Prcess {} {} {}, teaser:{}, teaser*: {}, opransac:{}, opransac*: {}, our: {}".format(uav_name,ground_name,scale,err1,err11,err2,err22,err3))

        ax=axs_list[id]

        teaser_area=0
        op_area=0
        our_area=0
        for i in range(0,len(scale_list)-1):
            scale_h=scale_list[i+1]-scale_list[i]
            teaser_area+=(teaser_res[i+1]+teaser_res[i])*scale_h/2
            op_area+=(opransac_res[i+1]+opransac_res[i])*scale_h/2
            our_area+=(our_res[i+1]+our_res[i])*scale_h/2

        line1=ax.plot(scale_list,teaser_res,color=T_COLOR,linewidth=LW,marker='o',markerfacecolor=T_COLOR,markeredgecolor=T_MK_COLOR, markeredgewidth=ELW, markersize=MS)
        line2=ax.plot(scale_list,opransac_res,color=O_COLOR,linewidth=LW,marker='o',markerfacecolor=O_COLOR,markeredgecolor=O_MK_COLOR, markeredgewidth=ELW, markersize=MS)
        line3=ax.plot(scale_list,our_res,color=OURS_COLOR,linewidth=LW,marker='o',markerfacecolor=OURS_COLOR,markeredgecolor=OURS_MK_COLOR, markeredgewidth=ELW, markersize=MS)
        # line1[0].set_label("TEASER++: AUC={}".format(teaser_area))
        # line2[0].set_label("OP-SAC: AUC={}".format(op_area))
        # line3[0].set_label("Ours: AUC={}".format(our_area))
        # ax.legend()
        ax.fill_between([0.5,10.5],[-1,-1],[5,5],color=[255/255,204/255,153/255],alpha=.5)

        ax.set_title(subtitles[id],fontsize=TITLE_SIZE)
        
        ax.set_yscale('scale')
        #ax.set(yscale='log')
        ax.set_xlim([0.5,10.5])
        ax.set_ylim([-1,3200])
        ax.xaxis.grid(True,linestyle='--')
        ax.yaxis.grid(True,linestyle='--')
        #plt.plot(scale_list,our_new_res)
        #plt.legend(['Teaser','Teaser*','Opransac','Opransac*','Our','Our_new'])
        #ax.legend(['Teaser','Opransac','Our'])
        # plt.yscale('log')
        # plt.xscale('log')
        #plt.show()
    axs_list[0].set(ylabel='RMSE[m]')
    axs_list[3].set(ylabel='RMSE[m]')
    axs_list[6].set(xlabel='scale', ylabel='RMSE[m]')
    axs_list[7].set(xlabel='scale')
    axs_list[8].set(xlabel='scale')
    fig.tight_layout()
    plt.legend(['TEASER++','OP-SAC','Ours'])
    plt.show()
    # teaser_res=np.array(teaser_res)
    # teaser_star_res=np.array(teaser_star_res)
    # opransac_res=np.array(opransac_res)
    # opransac_star_res=np.array(opransac_star_res)
    # our_res=np.array(our_res)

    # thresh=3
    # print("ISPRS (thresh={}): teaser: {}, teaser*: {}, opransac:{}, opransac*:{}, our:{}".format(thresh,np.mean(teaser_res[:11]<thresh),np.mean(teaser_star_res[:11]<thresh),np.mean(opransac_res[:11]<thresh),np.mean(opransac_star_res[:11]<thresh),np.mean(our_res[:11]<thresh)))
    # print("APL (thresh={}): teaser: {}, teaser*: {}, opransac:{}, opransac*:{}, our:{}".format(thresh,np.mean(teaser_res[11:]<thresh),np.mean(teaser_star_res[11:]<thresh),np.mean(opransac_res[11:]<thresh),np.mean(opransac_star_res[11:]<thresh),np.mean(our_res[11:]<thresh)))
    # print("Overall (thresh={}): teaser: {}, teaser*: {}, opransac:{}, opransac*:{}, our:{}".format(thresh,np.mean(teaser_res<thresh),np.mean(teaser_star_res<thresh),np.mean(opransac_res<thresh),np.mean(opransac_star_res<thresh),np.mean(our_res<thresh)))

def eval_scale_test():

    res_dir=r'J:\xuningli\papers\g2a\data\results\scale_result'
    scale_list=[1,1.25,1.5,1.75,2,3,5,7.5,10,15]
    
    #eval teaser++
    pair_list=dataset_scale_enumerate()


    fig, axs = plt.subplots(3, 3)
    axs_list=axs.flat
    subtitles=['ISPRS_Multi Center Obelisk', 'ISPRS_Multi Center Rathaus', 'ISPRS_Multi Center Stadthaus','ISPRS_Multi Zeche Lohnhalle','ISPRS_Multi Zeche Pferdestall','ISPRS_Multi Zeche Verwaltung','APL MP2','APL MP4','APL MP5']
    for id,pair in enumerate(pair_list):
        uav=pair[0]
        ground=pair[1]
        uav_name=os.path.basename(uav)
        ground_name=os.path.basename(ground)

        teaser_res=[]
        teaser_star_res=[]
        opransac_res=[]
        opransac_star_res=[]
        our_res=[]
        our_new_res=[]

        for scale in scale_list:
            init_trans=np.identity(4)
            rot=R.from_euler('xyz',[90,0,90],degrees=True).as_matrix()
            init_trans[:3,:3]=rot*scale
            init_trans[2,3]=5

            our_results_new=glob.glob(os.path.join(res_dir,"our_test","{}_{}*".format(ground_name,uav_name)))
            our_result_new=None
            for res in our_results_new:
                if float(os.path.basename(res)[:-8].split("_")[-1])==scale:
                    our_result_new=res
            our_trans_new=np.loadtxt(our_result_new)

            src_pc=o3d.io.read_point_cloud(ground)
            src_gt=np.array(src_pc.points)

            src_pc4=copy.deepcopy(src_pc)
            src_reg4=src_pc4.transform(our_trans_new)
            src_reg4=np.array(src_reg4.points)

            err4=np.mean(np.linalg.norm(src_reg4-src_gt,axis=1))
            our_res.append(err4)

            print("Prcess {} {} {}, ours_test:{}".format(uav_name,ground_name,scale,err4))
        
        ax=axs_list[id]

        #ax.plot(scale_list,teaser_res)
        #plt.plot(scale_list,teaser_star_res)
        #ax.plot(scale_list,opransac_res)
        #plt.plot(scale_list,opransac_star_res)
        ax.plot(scale_list,our_res)
        ax.set_title(subtitles[id])
        ax.set(xlabel='scale', ylabel='RMSE')

        #plt.plot(scale_list,our_new_res)
        #plt.legend(['Teaser','Teaser*','Opransac','Opransac*','Our','Our_new'])
        #ax.legend(['Teaser','Opransac','Our'])
        #plt.yscale('log')
        #plt.show()
    fig.tight_layout()
    plt.legend(['Our'])
    plt.show()
    # teaser_res=np.array(teaser_res)
    # teaser_star_res=np.array(teaser_star_res)
    # opransac_res=np.array(opransac_res)
    # opransac_star_res=np.array(opransac_star_res)
    our_res=np.array(our_res)

    # thresh=3
    # print("ISPRS (thresh={}): teaser: {}, teaser*: {}, opransac:{}, opransac*:{}, our:{}".format(thresh,np.mean(teaser_res[:11]<thresh),np.mean(teaser_star_res[:11]<thresh),np.mean(opransac_res[:11]<thresh),np.mean(opransac_star_res[:11]<thresh),np.mean(our_res[:11]<thresh)))
    # print("APL (thresh={}): teaser: {}, teaser*: {}, opransac:{}, opransac*:{}, our:{}".format(thresh,np.mean(teaser_res[11:]<thresh),np.mean(teaser_star_res[11:]<thresh),np.mean(opransac_res[11:]<thresh),np.mean(opransac_star_res[11:]<thresh),np.mean(our_res[11:]<thresh)))
    # print("Overall (thresh={}): teaser: {}, teaser*: {}, opransac:{}, opransac*:{}, our:{}".format(thresh,np.mean(teaser_res<thresh),np.mean(teaser_star_res<thresh),np.mean(opransac_res<thresh),np.mean(opransac_star_res<thresh),np.mean(our_res<thresh)))

def read_line(in_path):
    in_obj=open(in_path,'r')
    lines=in_obj.readlines()
    lines3d=[]
    pt_cnt=0
    tmp_line=[]
    for line in lines:
        if "v " not in line:
            continue
        line=line.rstrip('\n')
        x=float(line.split(" ")[1])
        y=float(line.split(" ")[2])
        z=float(line.split(" ")[3])
        if len(tmp_line)==9:
            lines3d.append([tmp_line[0],tmp_line[1],tmp_line[2],tmp_line[6],tmp_line[7],tmp_line[8]])
            tmp_line.clear()
        tmp_line.append(x)
        tmp_line.append(y)
        tmp_line.append(z)
    if len(tmp_line)==9:
        lines3d.append([tmp_line[0],tmp_line[1],tmp_line[2],tmp_line[6],tmp_line[7],tmp_line[8]])
    return lines3d

def dist_point_to_segment(p, a, b):
    """Calculate the distance from point p to the line segment ab."""
    ab = b - a
    ap = p - a
    bp = p - b
    if np.dot(ap, ab) <= 0.0:
        return np.linalg.norm(ap)
    if np.dot(bp, ab) >= 0.0:
        return np.linalg.norm(bp)
    return np.linalg.norm(np.cross(ab, ap)) / np.linalg.norm(ab)

def dist_between_segments(a0, a1, b0, b1):
    """Calculate the minimum distance between two line segments a0a1 and b0b1."""
    if (a0 == a1).all() and (b0 == b1).all():  # both segments are points
        return np.linalg.norm(a0 - b0)
    if (a0 == a1).all():  # first segment is a point
        return dist_point_to_segment(a0, b0, b1)
    if (b0 == b1).all():  # second segment is a point
        return dist_point_to_segment(b0, a0, a1)
    
    # Get the closest points between the lines extended infinitely
    da = a1 - a0
    db = b1 - b0
    dp = a0 - b0
    
    dap = np.cross(da, db)
    denom = np.dot(dap, dap)
    
    if denom == 0:  # lines are parallel
        return min(dist_point_to_segment(a0, b0, b1), dist_point_to_segment(a1, b0, b1),
                   dist_point_to_segment(b0, a0, a1), dist_point_to_segment(b1, a0, a1))
    
    numer = np.dot(np.cross(dp, db), dap)
    t1 = numer / denom
    
    closest_point_a = a0 + t1 * da
    closest_point_b = b0 + t1 * db
    
    if (0 <= t1 <= 1) and (0 <= t1 <= 1):
        return np.linalg.norm(closest_point_a - closest_point_b)
    else:
        return min(dist_point_to_segment(a0, b0, b1), dist_point_to_segment(a1, b0, b1),
                   dist_point_to_segment(b0, a0, a1), dist_point_to_segment(b1, a0, a1))

def repeat_inlier_gravity_line(in_dir):
    subdirs=glob.glob(os.path.join(in_dir,"*"))
    for subdir in subdirs:
        thresh=0.7
        if "apl" in subdir:
            thresh=2
        name=os.path.basename(subdir)
        ground_name=name.split("_")[-1]
        air_lines=read_line(os.path.join(subdir,"air_line3d.obj"))
        ground_lines=read_line(os.path.join(subdir,"ground_line3d.obj"))
        repeat_cnt=0
        for ground_line in ground_lines:
            g1=np.array(ground_line[:3])
            g2=np.array(ground_line[3:])
            status=False
            for air_line in air_lines:
                a1=np.array(air_line[:3])
                a2=np.array(air_line[3:])
                dis=dist_between_segments(g1,g2,a1,a2)
                if dis<thresh:
                    status=True
            if status:
                repeat_cnt+=1
        print("{}, #ground lines:{}, #air lines:{}, repeatablity:{}".format(ground_name,len(ground_lines),len(air_lines),repeat_cnt/len(ground_lines)))

def repeat_inlier_dual_gravity_line(in_dir):
    subdirs=glob.glob(os.path.join(in_dir,"*"))
    for subdir in subdirs:
        thresh=0.7
        if "apl" in subdir:
            thresh=2
        name=os.path.basename(subdir)
        ground_name=name.split("_")[-1]
        air_lines=read_line(os.path.join(subdir,"air_line3d.obj"))
        ground_lines=read_line(os.path.join(subdir,"ground_line3d.obj"))
        repeat_cnt=0
        match_cnt=0
        for gid in range(0,len(ground_lines),2):
            gline1=ground_lines[gid]
            gline2=ground_lines[gid+1]
            g1=np.array(gline1[:3])
            g2=np.array(gline1[3:])
            g3=np.array(gline2[:3])
            g4=np.array(gline2[3:])
            status=False
            for aid in range(0,len(air_lines),2):
                aline1=air_lines[aid]
                aline2=air_lines[aid+1]
                a1=np.array(aline1[:3])
                a2=np.array(aline1[3:])
                a3=np.array(aline2[:3])
                a4=np.array(aline2[3:])

                dis1=dist_between_segments(g1,g2,a1,a2)
                dis2=dist_between_segments(g3,g4,a3,a4)

                dis3=dist_between_segments(g1,g2,a3,a4)
                dis4=dist_between_segments(g3,g4,a1,a2)
                match_cnt+=1
                if dis1<thresh and dis2<thresh or dis3<thresh and dis4<thresh:
                    status=True
            if status:
                repeat_cnt+=1
        print("{}, #ground lines:{}, #air lines:{}, repeatablity:{}, #inlier match:{}".format(ground_name,len(ground_lines)/2,len(air_lines)/2,2*repeat_cnt/len(ground_lines), repeat_cnt))

def plot_overlap():
    o1=[26.1,12.4,9.6,5.4]
    o2=[16.4,8.3,5.7,5.6]
    o3=[5.5,3.2,2.4,2.2]
    o4=[5.9,2.7,1.4,0.9]
    o5=[7.5,10.5,3.1,13.4,4.3]
    o_highlight=[9.6,8.3,5.6,5.4,2.2,0.9]
    l1=['a1','a2','a3','a4']
    l2=['b1','b2','b3','b4']
    l3=['c1','c2','c3','c4']
    l4=['d1','d2','d3','d4']

    r1=[0.24,0.19,0.38,0.97]
    r2=[0.37,0.44,0.09,4.91]
    r3=[2.21,1.63,3.84,1.35]
    r4=[2.08,166.22,1.78,2.00]
    r5=[1.69,1.51,1.13,0.72,1.37]
    r_highlight=[0.38,0.44,4.91,0.97,1.35,2.00]

    o=o1+o2+o3+o4+o5
    r=r1+r2+r3+r4+r5
    o_list=[]
    for id,item in enumerate(o):
        o_list.append([item,r[id]])
    o_list=sorted(o_list,key=lambda x:x[0])
    o=[]
    r=[]
    for item in o_list:
        o.append(item[0])
        r.append(item[1])


    # plt.bar(l1,r1)
    # plt.bar(l2,r2)
    # plt.bar(l3,r3)
    # plt.bar(l4,r4)
    plt.gca().invert_xaxis()
    plt.plot(o,r,color=OURS_COLOR,linewidth=LW,marker='o',markerfacecolor=OURS_COLOR,markeredgecolor=OURS_MK_COLOR, markeredgewidth=ELW, markersize=MS)
    #plt.axvline(0.9,ymax=2.00,ymin=0,linestyle='--',label='0.9')
    #plt.annotate('(0.9,2.00)',(0.9,2.00))
    plt.ylim([-1,3200])
    plt.xlim([27,0])
    plt.yscale('scale')
    plt.grid(True,linestyle='--')
    plt.fill_between([0,27],[-1,-1],[5,5],color=[255/255,204/255,153/255],alpha=.5)
    plt.xlabel('Overlap [%]')
    plt.ylabel('RMSE [m]')
    plt.legend(['Ours'])
    #plt.yaxis.grid(True,linestyle='--')
    
    plt.show()

def minimumEffortPath(heights):
    h=len(heights)
    w=len(heights[0])
    def dfs(row,col,h,w,cur_path,total_path):
        if row==h-1 and col==w-1:
            total_path.append(cur_path)
            return
        else:              
            if row-1>=0 and [row-1,col] not in cur_path:
                dfs(row-1,col,h,w,cur_path+[[row-1,col]],total_path)
            if row+1<h and [row+1,col] not in cur_path:
                dfs(row+1,col,h,w,cur_path+[[row+1,col]],total_path)
            if col-1>=0 and [row,col-1] not in cur_path:
                dfs(row,col-1,h,w,cur_path+[[row,col-1]],total_path)
            if col+1<w and [row,col+1] not in cur_path:
                dfs(row,col+1,h,w,cur_path+[[row,col+1]],total_path)
    cur_path=[[0,0]]
    total_path=[]
    dfs(0,0,h,w,cur_path,total_path)
    for path in total_path:
        print(path)

def dfs(matrix, x, y, path,total_path):
    if x == len(matrix) - 1 and y == len(matrix[0]) - 1:
        total_path.append(path)
        return
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and (nx, ny) not in path:
            dfs(matrix, nx, ny, path + [(nx, ny)],total_path)

def bfs(matrix):
    queue=[[0,0]]
    total_path=[]
    


def enumerate_paths(matrix):
    total_path=[]
    dfs(matrix, 0, 0, [(0, 0)],total_path)
    for path in total_path:
        print(path)

if __name__=='__main__':
    mscale.register_scale(CustomScale)
    mscale.register_scale(OverScale)
    #eval_acc()
    #eval_scale()
    #eval_acc_new()
    #eval_scale_test()
    #cal_over()
    #repeat_inlier_gravity_line(r'E:\data\isprs_vis')
    #repeat_inlier_dual_gravity_line(r'E:\data\isprs_vis')
    plot_overlap()
    #heights = [[1,2,2],[3,8,2],[5,3,5]]
    #minimumEffortPath(heights)
    #matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    #enumerate_paths(matrix)

    