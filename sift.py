import cv2 

#sift
sift = cv2.SIFT_create()

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

#read images 
img1 = cv2.imread("C:/Users/Ahmed Gamal/Desktop/FiQE8r1WIAAYrgK.jpg")
img2 = cv2.imread("C:/Users/Ahmed Gamal/Desktop/FiQE8r1WIAAYrgK - Copy.jpg")

#convert BGR to gray
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#detect interest points and find the descriptors
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)

#match the descriptors and sort them 
matches = bf.match(descriptors_1,descriptors_2)
matches = sorted(matches, key = lambda x:x.distance)

#draw the matches 
img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[300:1000], img2, flags=2)

cv2.imshow('SIFT', img3)

cv2.waitKey(0)