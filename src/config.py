import numpy as np

class REG_CONFIG:
    sem_label_type: str='COCO'
    src_path: str=None
    ref_path: str=None
    src_sem_path: str=None
    ref_sem_path: str=None
    out_dir: str=None
    density_ratio: float=2 ## density ratio between src to ref pc
    init_trans: np.ndarray= np.identity(4)
    footprint_path: str=None

    use_sem:bool=False
    #dataset
    max_pts: int=10000000
    max_facade_pts:int=100000
    ref_max_len:float=400
    ref_min_len:float=200
    footprint_path:str=None
    gsd_ratio:float=2
    footprint_gsd: float=0.5
    air_gsd: float=0.02
    air_max_pixel_length: int=1100
    ground_gsd: float=0.01
    ground_max_pixel_length: int=200

    #ICP
    ICP_MAX_ITER:int=150
    ICP_MAX_SRC_PTS:int=50000
    ICP_MAX_REF_PTS:int=200000
    ICP_SOLVE_SCALE:bool=False
    ICP_RMSE_THRESHOLD:float=1e-5
    ICP_ANGLE_THRESHOLD:float=1e-5
    ICP_DISTANCE_THRESHOLD:float=1e-5
    ICP_SAVE_RESIDUALS:bool=False
    ICP_ROBUST:bool=True
    ##Image-Matching
    IM_NUM_BASE_PTS:int=20
    IM_MIN_SCALE:float=0.1
    IM_MAX_SCALE:float=1
    IM_SCALE_STEP:float=0.1
    IM_MIN_LENGTH:int=10
    IM_OVERLAP_THESH:float=2
    IM_MAX_LENGTH_PIX:int=1200