import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

def ejercicio1():
    # Read image and convert to grayscale
    imageName = 'Imagenes/a0004.jpg'
    ima = cv2.imread(imageName)
    gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)

    # Create detector and descriptor structures to compute SIFT
    detector=cv2.FeatureDetector_create('SIFT')
    descriptor = cv2.DescriptorExtractor_create('SIFT')

    # Detect keypoints with SIFT and sort them according to their response
    print 'Extracting Keypoints'
    init=time.time()
    kpts=detector.detect(gray)
    kpts = sorted(kpts, key = lambda x:x.response)

    end=time.time()
    print 'Extracted '+str(len(kpts))+' keypoints.'
    print 'Done in '+str(end-init)+' secs.'
    print ''

    # Compute SIFT descriptor for all keypoints
    print 'Computing SIFT descriptors'
    init=time.time()
    kpts,des=descriptor.compute(gray,kpts)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'

    # Show result of detecting keypoints
    Npoints = 1000
    if len(kpts) < Npoints: Npoints = len(kpts)

    im_with_keypoints = cv2.drawKeypoints(ima, kpts[1:Npoints], np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey()




def get_N_sift_points(image, n_points=0):



    # Create detector and descriptor structures to compute SIFT
    detector = cv2.FeatureDetector_create('SIFT')
    descriptor = cv2.DescriptorExtractor_create('SIFT')

    kpts=detector.detect(image)
    kpts = sorted(kpts, key = lambda x:x.response)
    kpts,des = descriptor.compute(image,kpts)

    if (len(kpts) < n_points or n_points == 0): n_points = len(kpts)

    return kpts[:n_points], des[:n_points]


def matching(image1, image2):

    kp1, des1 = get_N_sift_points(image1, n_points=0)
    kp2, des2 = get_N_sift_points(image2, n_points=0)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches,key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = drawMatches(image1,kp1,image2,kp2,matches[:10])


    plt.imshow(img3), plt.show()



def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    # cv2.imshow('Matched Features', out)
    # cv2.waitKey(0)
    # cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

def calculate_distance(image1, image2):

    return 0

if __name__ == '__main__':

    imagename1 = 'Imagenes/a0004.jpg'
    image1 = cv2.imread(imagename1)
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    imagename2 = 'Imagenes/a0005.jpg'
    image2 = cv2.imread(imagename2)
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

    matching(gray1, gray2)

    distance = calculate_distance(gray1, gray2)

    print 'Distance between images is ' + str(distance) + '.'
