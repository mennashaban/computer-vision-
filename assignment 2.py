import cv2
import os
import numpy as np

def sift(image1_path, image2_path):
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # David Lowe's ratio test
    goodMatches = []
    for firstMatch, secondMatch in matches:
        if firstMatch.distance < 0.7 * secondMatch.distance:
            goodMatches.append(firstMatch)

    # Cross-check technique
    crossOUTlist = []
    for m in goodMatches:
        revMatch = next((x[0] for x in matches if x[0].queryIdx == m.queryIdx and x[0].trainIdx == m.trainIdx), None)
        if revMatch:
            crossOUTlist.append(m)

    # point 4
    src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 20)
    matchesMask = mask.ravel().tolist()

    # draw matches
    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, goodMatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Create a resizable window
    cv2.namedWindow(f"Matches between {os.path.basename(image1_path)} and {os.path.basename(image2_path)}",
                    cv2.WINDOW_NORMAL)


    # Resize the image for better visualization
    height, width = img_matches.shape[:2]
    # img_matches_resized = cv2.resize(img_matches, (width // 2, height // 2))

    cv2.imshow(f"Matches between {os.path.basename(image1_path)} and {os.path.basename(image2_path)}", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # rule point
    similarity_score = len(matchesMask) / min(len(kp1), len(kp2))
    return similarity_score


def main():
    folder_path = "D:/Desktop/study/material/computer vision/labs pdf/lab 6/assignment data"  # Change this to the path of your image folder

    # Define the image pairs to compare
    image_pairs = [
        ("image1a.jpeg", "image1b.jpeg"),
        ("image2a.jpeg", "image2b.jpeg"),
        ("image3a.jpeg", "image3b.jpeg"),
        ("image4a.jpeg", "image4b.jpeg"),
        ("image4a.jpeg", "image4c.png"),
        ("image5a.jpeg", "image5b.jpeg"),
        ("image6a.jpeg", "image6b.jpeg"),
        ("image7a.jpeg", "image7b.jpeg")
    ]

    for image_pair in image_pairs:
        image1_path = os.path.join(folder_path, image_pair[0])
        image2_path = os.path.join(folder_path, image_pair[1])
        similarity_score = sift(image1_path, image2_path)
        print(f" Similarity between {image_pair[0]} and {image_pair[1]}: {similarity_score:.4f}")

if __name__ == "__main__":
    main()
