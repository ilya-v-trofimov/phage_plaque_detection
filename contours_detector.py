import argparse
import imutils
import cv2
import numpy as np
import os
import pandas as pd

out_dir_path = './out'


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


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


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


def draw_contours(image, green_df, red_df, other_df):
    image_copy = image.copy()
    for index, green in green_df.iterrows():
        draw_one_contour(image_copy, green, (0, 255, 0))
    for index, red in red_df.iterrows():
        draw_one_contour(image_copy, red, (0, 0, 255))
    for index, other in other_df.iterrows():
        draw_one_contour(image_copy, other, (150, 150, 150))
    return image_copy


def draw_one_contour(image, c_df, color):
    m = cv2.moments(c_df['CONTOURS'])
    if m["m00"] != 0:
        cx = int((m["m10"] / m["m00"]))
        cy = int((m["m01"] / m["m00"]))
    else:
        cx = 0
        cy = 0
    cv2.drawContours(image, [c_df['HULL']], -1, color, 1)
    cv2.putText(image, f"#{c_df['INDEX_COL']}:{c_df['AREA']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


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


def write_images(out_dir, output_image, binary_image, high_contrast_image, image_path):
    cv2.imwrite(f'{out_dir}/out-{os.path.split(image_path)[1]}', output_image)
    cv2.imwrite(f'{out_dir}/contrast-{os.path.split(image_path)[1]}', high_contrast_image)
    cv2.imwrite(f'{out_dir}/binary-{os.path.split(image_path)[1]}', binary_image)


def write_data(out_dir, image_path, green_df, red_df, other_df):
    write_one_data(out_dir, image_path, 'green', green_df)
    write_one_data(out_dir, image_path, 'red', red_df)
    write_one_data(out_dir, image_path, 'other', other_df)


def write_one_data(out_dir, image_path, prefix, df):
    image_file_name = os.path.split(image_path)[1]
    image_name = os.path.splitext(image_file_name)[0]
    df.to_csv(path=f'{out_dir}/data-{prefix}-{image_name}.csv')


def calc_area_diff(contour_df):
    encl_area = 3.1415 * (contour_df['ENCL_DIAMETER'] ** 2) / 4
    return abs(1 - contour_df['AREA'] / encl_area)


def prepare_df(contours):
    df = pd.DataFrame(contours, columns=['CONTOURS'])
    df['HULL'] = df.apply(lambda x: cv2.convexHull(x['CONTOURS']), axis=1)
    df['AREA'] = df.apply(lambda x: cv2.contourArea(x['HULL']), axis=1)
    encl_circle = df.apply(lambda x: cv2.minEnclosingCircle(x['HULL']), axis=1)
    df['ENCL_CENTER'] = encl_circle.str[0]
    df['ENCL_DIAMETER'] = encl_circle.str[1] * 2
    df['PERIMETER'] = df.apply(lambda x: cv2.arcLength(x['HULL'], True), axis=1)
    return df


def filter_contours(contours):
    df = prepare_df(contours)
    filter_other = df.apply(lambda x: x['AREA'] < 100 or x['AREA'] > 100000, axis=1)
    other_df = df[filter_other]
    wo_other_df = df[~filter_other]
    filter_green = wo_other_df.apply(lambda x: calc_area_diff(x) < 0.21, axis=1)
    green_df = wo_other_df[filter_green]
    red_df = wo_other_df[~filter_green]
    green_df['INDEX_COL'] = green_df.index
    red_df['INDEX_COL'] = red_df.index
    other_df['INDEX_COL'] = other_df.index
    return green_df, red_df, other_df


def main():
    args = parse_args()

    image_paths = get_image_paths(args['image'], args['directory'])
    for image_path in image_paths:
        image = cv2.imread(image_path)

        binary_image, high_contrast, clr_high_contrast = process_image(image)
        contours = get_contours(binary_image)
        green_df, red_df, other_df = filter_contours(contours)
        output = draw_contours(clr_high_contrast, green_df, red_df, other_df)

        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)
        write_images(out_dir_path, output, binary_image, high_contrast, image_path)
        write_data(out_dir_path, image_path, green_df, red_df, other_df)
        # cv2.imshow("Binary image", binary_image)
        # cv2.imshow("Image", np.hstack((output, clr_high_contrast)))
        # cv2.waitKey(0)


if __name__ == '__main__':
    main()
