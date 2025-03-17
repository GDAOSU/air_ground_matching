import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def merge_line_segments(segments, threshold):
    merged_segments = []

    while segments:
        merged=False
        segment = segments.pop(0)
        p1=segment[:2]
        p2=segment[2:]
        for i, existing_segment in enumerate(merged_segments):
            q1=existing_segment[:2]
            q2=existing_segment[2:]
            dis1=distance(p1,q1)
            dis2=distance(p2,q2)
            dis3=distance(p1,q2)
            dis4=distance(p2,q1)
            if (dis1<threshold and dis2<threshold) :
                merged_segments[i] = [(p1[0]+q1[0])/2,(p1[1]+q1[1])/2,(p2[0]+q2[0])/2,(p2[1]+q2[1])/2]
                merged = True
                break
            elif (dis3<threshold and dis4<threshold):
                merged_segments[i] = [(p1[0]+q2[0])/2,(p1[1]+q2[1])/2,(p2[0]+q1[0])/2,(p2[1]+q1[1])/2]
                merged = True
                break
        if not merged:
            merged_segments.append(segment)
    
    return merged_segments

def predict_lsd(input):
    import os
    import numpy as np
    import cv2
    import copy
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap as lsc
    import torch
    import h5py

    from deeplsd.utils.tensor import batch_to_device
    from deeplsd.models.deeplsd_inference import DeepLSD
    from deeplsd.geometry.viz_2d import plot_images, plot_lines
    MAX_SIZE=3000
    # Model config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf = {
        'detect_lines': True,  # Whether to detect lines or only DF/AF
        'line_detection_params': {
            'merge': True,  # Whether to merge close-by lines
            'filtering': False,  # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
            'grad_thresh': 3,
            'grad_nfa': True,  # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
        }
    }

    # Load the model
    ckpt = 'deeplsd/weights/deeplsd_wireframe.tar'
    ckpt = torch.load(str(ckpt), map_location='cpu', weights_only=False)
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()

    gray_img=copy.deepcopy(input)
    h,w=gray_img.shape
    max_hw=max(h,w)
    scale=1
    if max_hw>MAX_SIZE:
        scale=MAX_SIZE/max_hw
        h_scale=int(h*scale)
        w_scale=int(w*scale)
        gray_img=cv2.resize(gray_img,(w_scale,h_scale))
    # img = cv2.imread(input)[:, :, ::-1]
    # gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Detect (and optionally refine) the lines
    inputs = {'image': torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.}
    with torch.no_grad():
        out = net(inputs)
        pred_lines = out['lines'][0]/scale
    print("DeepLSD #lines: {}".format(pred_lines.shape[0]))

    img_rgb=np.zeros((h,w,3),dtype=np.uint8)
    lines=[]
    for i in range(pred_lines.shape[0]):
        line=pred_lines[i]
        lines.append([line[0][0],line[0][1],line[1][0],line[1][1]])
        pt1=line[0].astype(np.int32)
        pt2=line[1].astype(np.int32)
        cv2.circle(img_rgb,pt1,2,(255,0,0),-1)
        cv2.circle(img_rgb,pt2,2,(255,0,0),-1)
        cv2.line(img_rgb,pt1,pt2,(0,0,255),1,16)
    
    lines=merge_line_segments(lines,6)
    #cv2.imwrite(input[:-4]+"_deeplsd.png",img_rgb)
    lines=np.array(lines)
    lines=lines.astype(np.int32)
    lines=np.expand_dims(lines,axis=1)
    return lines

