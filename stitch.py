from panorama import Panaroma
import imutils
import cv2
import os
import sys
from skimage.metrics import structural_similarity as compare_ssim


def compute_ssim(imageA, imageB):
    height = min(imageA.shape[0], imageB.shape[0])
    width = min(imageA.shape[1], imageB.shape[1])
    imageA = cv2.resize(imageA, (width, height))
    imageB = cv2.resize(imageB, (width, height))

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, _) = compare_ssim(grayA, grayB, full=True)
    return score




BASE_DIR = os.path.dirname(os.path.abspath(__file__))

no_of_images = int(input("Enter the number of images you want to concatenate: "))
print("Enter the image names with extension in order of left to right in the way you want to concatenate: ")

filename = []
for i in range(no_of_images):
    filename.append(input("Enter the %d image name along with path and extension: " % (i + 1)))



images = []

for fname in filename:
    img_path = os.path.join(BASE_DIR, "inputs", fname)
    print(f"[LOADING] {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not load image: {img_path}")
        sys.exit(1)

    img = imutils.resize(img, width=400)
    img = imutils.resize(img, height=400)
    images.append(img)


panorama = Panaroma()
if no_of_images == 2:
    (result, matched_points) = panorama.image_stitch([images[0], images[1]], match_status=True)
else:
    (result, matched_points) = panorama.image_stitch([images[no_of_images - 2], images[no_of_images - 1]], match_status=True)
    for i in range(no_of_images - 2):
        (result, matched_points) = panorama.image_stitch([images[no_of_images - i - 3], result], match_status=True)


print("[INFO] Stitching with OpenCV's default Stitcher...")
opencv_stitcher = cv2.Stitcher_create()
(status, stitched_opencv) = opencv_stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    ssim_score = compute_ssim(result, stitched_opencv)
    print(f"[INFO] SSIM between custom and OpenCV panorama: {ssim_score:.4f}")

    cv2.imshow("OpenCV Panorama", stitched_opencv)
    cv2.imwrite("output/opencv_panorama.jpg", stitched_opencv)
else:
    print("[ERROR] OpenCV stitcher failed.")



cv2.imshow("Keypoint Matches", matched_points)
cv2.imshow("Panorama", result)

cv2.imwrite("output/matched_points.jpg", matched_points)
cv2.imwrite("output/panorama_image.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
