import argparse
import imutils
import cv2
import numpy as np
import os


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def draw_one_contour(image, c, area, color):
    m = cv2.moments(c)
    if m["m00"] != 0:
        cx = int((m["m10"] / m["m00"]))
        cy = int((m["m01"] / m["m00"]))
    else:
        cx = 0
        cy = 0
    cv2.drawContours(image, [c], -1, color, 1)
    cv2.putText(image, f"{area}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    high_contrast = cv2.convertScaleAbs(blurred, alpha=2.5, beta=0)
    high_contrast = adjust_gamma(high_contrast, 1.5)
    binary = cv2.adaptiveThreshold(high_contrast, 500, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 2)
    clr_high_contrast = cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)
    return binary, high_contrast, clr_high_contrast


def get_contours(binary_image):
    contours = cv2.findContours(binary_image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(contours)


def draw_contours(image, contours):
    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if 200 < area < 100000:
            _, radius = cv2.minEnclosingCircle(hull)
            encl_area = 3.1415 * radius * radius
            area_diff = abs(1 - area / encl_area)
            if area_diff < 0.21:
                draw_one_contour(image, hull, area, (0, 255, 0))  # GREEN
            else:
                draw_one_contour(image, hull, area, (0, 0, 255))  # RED


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False,
                    help="path to the input image")
    ap.add_argument("-d", "--directory", required=False,
                    help="path to the directory with input images")
    args = vars(ap.parse_args())
    if 'image' not in args and 'directory' not in args:
        raise Exception('Either -i or -d flags must be provided!')
    return args


def get_image_paths(image, directory):
    images = []
    if image:
        images.append(image)
    else:
        for r, d, f in os.walk(directory):
            for file in f:
                if os.path.splitext(file)[1] in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
                    images.append(os.path.join(r, file))
    return images


def write_images(output_image, binary_image, high_contrast_image, image_path):
    cv2.imwrite(f'./out/out-{os.path.split(image_path)[1]}', output_image)
    cv2.imwrite(f'./out/contrast-{os.path.split(image_path)[1]}', high_contrast_image)
    cv2.imwrite(f'./out/binary-{os.path.split(image_path)[1]}', binary_image)


def write_data(image_path, contours):

    a = np.asarray([['CONTOUR_NUMBER', 2, 3]])
    image_file_name = os.path.split(image_path)[1]
    image_name = os.path.splitext(image_file_name)[0]
    np.savetxt(f'data-{image_name}.csv', a, delimiter=",")
#todo

def main():
    args = parse_args()

    image_paths = get_image_paths(args['image'], args['directory'])
    for image_path in image_paths:
        image = cv2.imread(image_path)

        binary_image, high_contrast, clr_high_contrast = process_image(image)
        contours = get_contours(binary_image)
        output = clr_high_contrast.copy()
        draw_contours(output, contours)

        write_images(output, binary_image, high_contrast, image_path)
        write_data(image_path, contours)
        # cv2.imshow("Binary image", binary_image)
        # cv2.imshow("Image", np.hstack((output, clr_high_contrast)))
        # cv2.waitKey(0)


if __name__ == '__main__':
    main()
