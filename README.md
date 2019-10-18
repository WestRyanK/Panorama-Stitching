A panorama stitcher program. Give it several images and it will automatically stitch them together for you.

![Provo Temple Stitched](stitched.png "Provo Temple Stitched")

![BYU Campus Stitched](stitched02.png "BYU Campus Stitched")


In order to load new images with my image stitcher, go to the bottom of the code and change the paths on the lines:

im1 = cv2.imread("temple_horiz1_1200.png")
im2 = cv2.imread("temple_horiz2_1200.png")
im3 = cv2.imread("temple_horiz3_1200.png")

Run the code.

The stitched image will be saved as stitched.png
