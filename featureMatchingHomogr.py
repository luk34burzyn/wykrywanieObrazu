import numpy as np
import cv2
from matplotlib import pyplot as plt


def featureMatchingHomogr(img1, img11, img2, img22, MIN_MATCH_COUNT, img3, dyst):

    img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    #wartość k=2 dobrana domyślnie, doświadczalnie
    #znajduje 2 najlepsze dopasowania dla każdego deskryptora
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < dyst*n.distance:
            good.append(m)

    if len(good)>=MIN_MATCH_COUNT:
        print("Enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        Mm, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        matchesMask = mask.ravel().tolist()
        h, w= img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.perspectiveTransform(pts, Mm)

        # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        MIN_MATCH_COUNT = len(good)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img11, kp1, img22, kp2, good, None, **draw_params)

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        img3 = img3

    return img3, MIN_MATCH_COUNT



    # w, h, ssl = img3.shape
    # plt.imshow(img3, 'gray'), plt.show()

    # for pt in pts:
    #     cv2.rectangle(img3, pt, (pt[0] + round(1.5 * w), pt[1] + h), (0, 255, 255), 2)
