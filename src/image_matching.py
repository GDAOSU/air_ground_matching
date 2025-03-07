MIN_MATCH_COUNT=2
def img_matching(src_img,ref_img):
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt
    
    img1 = cv.imread(src_img,cv.IMREAD_GRAYSCALE) # queryImage
    img2 = cv.imread(ref_img,cv.IMREAD_GRAYSCALE) # trainImage
    
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.95*n.distance:
            good.append([m])
    
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
        
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = 255, # draw matches in green color
    singlePointColor = None,
    matchesMask = matchesMask, # draw only inliers
    flags = 2)
    good1=[m[0] for m in good]
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good1,None,matchColor=255,matchesMask=matchesMask,matchesThickness=2)
 
    plt.imshow(img3, 'gray'),plt.show()



if __name__=='__main__':
    img_matching(r'E:\data\test1\4422_osu\pcl\out_a2f\air_boundary.png',r'E:\data\test1\4422_osu\pcl\out_a2f\footprint_img.png')