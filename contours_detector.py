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


def process_image(image, contrast):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("./test_pic_grey.jpg", gray)

    gray = cv2.fastNlMeansDenoising(gray, h=5)
    #cv2.imwrite("./test_pic_grey_thresh_denoise.jpg", gray)

    # ET to get full circle plate
    # ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    # ret, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite("./test_pic_green_grey_thresh.jpg", thresh)

    # gray = unsharp_mask(gray)
    # cv2.imwrite("./test_pic_grey_unsharp.jpg", gray)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # blurred = cv2.medianBlur(gray, 9)
    #cv2.imwrite("./test_pic_blur.jpg", blurred)

    high_contrast = cv2.convertScaleAbs(blurred, alpha=contrast, beta=0)
    #cv2.imwrite("./test_pic_high.jpg", high_contrast)

    high_contrast = adjust_gamma(high_contrast, 1.0)
    #cv2.imwrite("./test_pic_green_gamma.jpg", high_contrast)

    # binary = cv2.adaptiveThreshold(high_contrast, 500, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 2)
    # binary = cv2.adaptiveThreshold(high_contrast, 500, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 55, 2)

    # ret, thresh = cv2.threshold(high_contrast, 162, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(high_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 265, 2)
    #cv2.imwrite("./test_pic_grey_thresh.jpg", thresh)

    # laplacian = cv2.Laplacian(blur, -1, ksize=17, delta=-50)
    # laplacian = cv2.Laplacian(thresh, cv2.CV_64F)
    laplacian = cv2.Laplacian(thresh, cv2.CV_8UC1)
    #cv2.imwrite("./test_pic_laplacian.jpg", laplacian)
    # gray_lapl = cv2.cvtColor(laplacian, cv2.COLOR_BGR2GRAY)

    # binary = cv2.threshold(laplacian, 165, 255, cv2.THRESH_BINARY)
    # binary = cv2.adaptiveThreshold(laplacian, 500, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 2)
    # cv2.imwrite("./test_pic_green_laplacian_binary.jpg", binary)

    clr_high_contrast = cv2.cvtColor(high_contrast, cv2.COLOR_GRAY2BGR)
    return laplacian, high_contrast, clr_high_contrast
    # return binary, high_contrast, clr_high_contrast


def get_contours(binary_image):
    contours = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(binary_image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # contours_hierarchy = cv2.findContours(binary_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return imutils.grab_contours(contours)
    # return contours


def draw_contours(image, green_df, red_df, other_df, second_run=False):
    image_copy = image.copy()
    for index, green in green_df.iterrows():
        draw_one_contour(image_copy, green, (0, 255, 0))
    #for index, red in red_df.iterrows():
    #    draw_one_contour(image_copy, red, (0, 0, 255))
    #for index, other in other_df.iterrows():
    #   draw_one_contour(image_copy, other, (150, 150, 150))
    return image_copy


def draw_one_contour(image, c_df, color):
    m = cv2.moments(c_df['CONTOURS'])
    if m["m00"] != 0:
        cx = int((m["m10"] / m["m00"]))
        cy = int((m["m01"] / m["m00"]))
    else:
        cx = 0
        cy = 0

    image_w_contours = cv2.drawContours(image, [c_df['HULL']], -1, color, 1)
    cv2.putText(image, f"#{c_df['INDEX_COL']}:{c_df['AREA']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image_w_contours


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
    cv2.imwrite(f'{out_dir}/out_{os.path.split(image_path)[1]}', output_image)
    # cv2.imwrite(f'{out_dir}/out_red-{os.path.split(image_path)[1]}', output_image_red)
    cv2.imwrite(f'{out_dir}/contrast-{os.path.split(image_path)[1]}', high_contrast_image)
    cv2.imwrite(f'{out_dir}/binary-{os.path.split(image_path)[1]}', binary_image)


def write_data(out_dir, image_path, green_df, red_df, other_df):
    write_one_data(out_dir, image_path, 'green', green_df)
    # write_one_data(out_dir, image_path, 'red', red_df)
    # write_one_data(out_dir, image_path, 'other', other_df)


def write_one_data(out_dir, image_path, prefix, df):
    image_file_name = os.path.split(image_path)[1]
    image_name = os.path.splitext(image_file_name)[0]
    df.to_csv(path_or_buf=f'{out_dir}/data-{prefix}-{image_name}.csv',
              columns=['INDEX_COL', 'AREA', 'PERIMETER', 'ENCL_CENTER', 'ENCL_DIAMETER'])


def calc_area_diff(contour_df):
    encl_area = 3.1415 * (contour_df['ENCL_DIAMETER'] ** 2) / 4
    return abs(1 - contour_df['AREA'] / encl_area)


def prepare_df(contours):
    df = pd.DataFrame(contours, columns=['CONTOURS'])
    # df = pd.DataFrame(imutils.grab_contours(contours), columns=['CONTOURS'])
    df['HULL'] = df.apply(lambda x: cv2.convexHull(x['CONTOURS']), axis=1)
    df['AREA'] = df.apply(lambda x: cv2.contourArea(x['HULL']), axis=1)
    encl_circle = df.apply(lambda x: cv2.minEnclosingCircle(x['HULL']), axis=1)
    df['ENCL_CENTER'] = encl_circle.str[0]
    df['ENCL_DIAMETER'] = encl_circle.str[1] * 2
    df['PERIMETER'] = df.apply(lambda x: cv2.arcLength(x['HULL'], True), axis=1)
    return df


def filter_contours(contours):
    df = prepare_df(contours)
    # df = prepare_df(imutils.grab_contours(contours))
    filter_other = df.apply(lambda x: x['AREA'] < 100 or x['AREA'] > 100000, axis=1)
    other_df = df[filter_other]
    wo_other_df = df[~filter_other]
    filter_green = wo_other_df.apply(lambda x: calc_area_diff(x) < 0.21, axis=1)
    #filter_green = wo_other_df.apply(lambda x: calc_area_diff(x) < 0.21, axis=1)
    green_df = wo_other_df[filter_green]
    red_df = wo_other_df[~filter_green]
    green_df.reset_index()
    green_df['INDEX_COL'] = green_df.index
    red_df.reset_index()
    red_df['INDEX_COL'] = red_df.index
    other_df.reset_index()
    other_df['INDEX_COL'] = other_df.index
    return green_df, red_df, other_df


def unsharp_mask(image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def check_duplicates(green, green_df):
    for x in green_df.iterrows():
        if green['INDEX_COL'] != x[0]:
            close_centers = abs(green['ENCL_CENTER'][0] - x[1]['ENCL_CENTER'][0]) <= 25
            close_areas = abs(green['AREA'] - x[1]['AREA']) <= 25
            if close_centers is True and close_areas is True:
                return True


def main():
    args = parse_args()

    image_paths = get_image_paths(args['image'], args['directory'])
    for image_path in image_paths:
        image = cv2.imread(image_path)

        binary_image, high_contrast, clr_high_contrast = process_image(image, 2.5)
        #       cv2.imshow("Binary image", binary_image)
        contours = get_contours(binary_image)
        green_df, red_df, other_df = filter_contours(contours)

        # Filter plaques duplicates (circle in circle)
        # TODO use apply
        green_df['DUPLICATE'] = False
        green_df_copy = green_df
        for green in green_df.iterrows():
            # green_df_copy['DUPLICATE'][green[0]] = check_duplicates(green[1], green_df)
            if (check_duplicates(green[1], green_df_copy)):
                green_df_copy = green_df_copy.drop(green_df_copy[green_df_copy.index == green[0]].index)
            #test = green_df['INDEX_COL'].apply(lambda x: check_duplicates(green, x))

        output = draw_contours(clr_high_contrast, green_df_copy, red_df, other_df)

        write_images(out_dir_path, output, binary_image, high_contrast, image_path)
        write_data(out_dir_path, image_path, green_df_copy, red_df, other_df)


if __name__ == '__main__':
    main()
