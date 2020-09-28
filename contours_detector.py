import argparse
import imutils
import cv2
import numpy as np
import os
import pandas as pd
import math

out_dir_path = './out'


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False,
                    help="path to the input image")
    ap.add_argument("-d", "--directory", required=False,
                    help="path to the directory with input images")
    ap.add_argument("-p", "--plate_size", required=False,
                    help="plate size (mm)")
    ap.add_argument("-small", "--small_plaque", required=False,
                    help="small plaques (Y/N)")
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


def process_image(image, contrast, small_plaque=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("./test_pic_grey.jpg", gray)

    # ET to get full circle plate
    # plate_only = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    # plate_only = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite("./test_pic_plate_only.jpg", plate_only)

    # ET added
    h = 5
    if small_plaque:
        h = 3
    gray = cv2.fastNlMeansDenoising(gray, h=h)
    cv2.imwrite("./test_pic_grey_thresh_denoise.jpg", gray)

    # gray = unsharp_mask(gray)
    # cv2.imwrite("./test_pic_grey_unsharp.jpg", gray)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # blurred = cv2.medianBlur(gray, 9)
    cv2.imwrite("./test_pic_blur.jpg", blurred)

    # ET point where it depends on a pic
    high_contrast = cv2.convertScaleAbs(blurred, alpha=contrast, beta=0)
    cv2.imwrite("./test_pic_high.jpg", high_contrast)

    gamma_test = adjust_gamma(high_contrast, 7.1)
    cv2.imwrite("./test_pic_green_gamma_0.jpg", gamma_test)

    high_contrast = adjust_gamma(high_contrast, 1.0)
    cv2.imwrite("./test_pic_green_gamma.jpg", high_contrast)

    # binary = cv2.adaptiveThreshold(high_contrast, 500, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 2)
    # binary = cv2.adaptiveThreshold(high_contrast, 500, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 55, 2)

    plate_only = cv2.adaptiveThreshold(high_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 1)
    # plate_only = cv2.threshold(high_contrast, 65, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("./test_pic_plate_only.jpg", plate_only)

    # ret, thresh = cv2.threshold(high_contrast, 162, 255, cv2.THRESH_BINARY_INV)

    # ET added
    # blockSize affects large/small plaques (circles in circles)
    # change blockSize based on the AREA
    block_size = 265
    if small_plaque:
        block_size = 49

    thresh = cv2.adaptiveThreshold(high_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                   block_size, 2)
    # thresh = cv2.adaptiveThreshold(high_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 2)
    cv2.imwrite("./test_pic_grey_adapt_thresh.jpg", thresh)

    # laplacian = cv2.Laplacian(blur, -1, ksize=17, delta=-50)
    # laplacian = cv2.Laplacian(thresh, cv2.CV_64F)
    laplacian = cv2.Laplacian(thresh, cv2.CV_8UC1)
    # cv2.imwrite("./test_pic_laplacian.jpg", laplacian)
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


def draw_contours(image, green_df, red_df, other_df, plate_df):
    image_copy = image.copy()
    for index, green in green_df.iterrows():
        draw_one_contour(image_copy, green, (0, 255, 0))
    # for index, red in red_df.iterrows():
    #    draw_one_contour(image_copy, red, (0, 0, 255))
    # for index, other in other_df.iterrows():
    #   draw_one_contour(image_copy, other, (150, 150, 150))
    # for index, plate in plate_df.iterrows():
    #    draw_one_contour(image_copy, plate, (0, 128, 255))
    return image_copy


def draw_one_contour(image, c_df, color):
    m = cv2.moments(c_df['CONTOURS'])
    if m["m00"] != 0:
        cx = int((m["m10"] / m["m00"]))
        cy = int((m["m01"] / m["m00"]))
    else:
        cx = 0
        cy = 0

    # ET

    pd.set_option('display.precision', 2)
    image_w_contours = cv2.drawContours(image, [c_df['HULL']], -1, color, 1)
    cv2.putText(image, f"#{c_df['INDEX_COL']}:{c_df['ENCL_DIAMETER_MM']}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1)

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
              columns=['INDEX_COL', 'AREA_PXL', 'PERIMETER_PXL', 'ENCL_CENTER', 'ENCL_DIAMETER_PXL', 'AREA_MM2', 'ENCL_DIAMETER_MM','MEAN_COLOUR'])


def calc_AREA_PXL_diff(contour_df):
    encl_AREA_PXL = 3.1415 * (contour_df['ENCL_DIAMETER_PXL'] ** 2) / 4
    return abs(1 - contour_df['AREA_PXL'] / encl_AREA_PXL)


def prepare_df(contours):
    pd.options.display.float_format = '{:,.2f}'.format
    df = pd.DataFrame(contours, columns=['CONTOURS'])
    # df = pd.DataFrame(imutils.grab_contours(contours), columns=['CONTOURS'])
    df['HULL'] = df.apply(lambda x: cv2.convexHull(x['CONTOURS']), axis=1)
    df['AREA_PXL'] = df.apply(lambda x: cv2.contourArea(x['HULL']), axis=1)
    encl_circle = df.apply(lambda x: cv2.minEnclosingCircle(x['HULL']), axis=1)
    df['ENCL_CENTER'] = encl_circle.str[0]
    df['ENCL_DIAMETER_PXL'] = encl_circle.str[1] * 2
    df['PERIMETER_PXL'] = df.apply(lambda x: f"{cv2.arcLength(x['HULL'], True):.2f}", axis=1)

    return df


def filter_contours(contours):
    df = prepare_df(contours)
    # df = prepare_df(imutils.grab_contours(contours))

    # filter_other = df.apply(lambda x: x['AREA_PXL'] < 100 or x['AREA_PXL'] > 100000, axis=1)
    filter_other = df.apply(lambda x: x['AREA_PXL'] < 100, axis=1)
    other_df = df[filter_other]
    wo_other_df = df[~filter_other]
    filter_green = wo_other_df.apply(lambda x: x['AREA_PXL'] < 100000 and calc_AREA_PXL_diff(x) < 0.21, axis=1)
    filter_plate = wo_other_df.apply(lambda x: x['AREA_PXL'] > 100000 and calc_AREA_PXL_diff(x) < 0.21, axis=1)
    # filter_plate = filter_plate.apply(lambda x: calc_AREA_PXL_diff(x) < 0.21, axis=1)

    green_df = wo_other_df[filter_green]
    red_df = wo_other_df[~filter_green]
    plate_df = wo_other_df[filter_plate]

    green_df.reset_index()
    green_df['INDEX_COL'] = green_df.index

    red_df.reset_index()
    red_df['INDEX_COL'] = red_df.index

    other_df.reset_index()
    other_df['INDEX_COL'] = other_df.index

    plate_df.reset_index()
    plate_df['INDEX_COL'] = plate_df.index
    return green_df, red_df, other_df, plate_df


def unsharp_mask(image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    # ET
    # blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    # sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = float(amount + 1) * image - float(amount) * image
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def check_duplicates(obj, obj_df):
    for x in obj_df.iterrows():
        if obj['INDEX_COL'] != x[0]:
            close_centers = abs(obj['ENCL_CENTER'][0] - x[1]['ENCL_CENTER'][0]) <= 25
            close_AREA_PXLs = abs(obj['AREA_PXL'] - x[1]['AREA_PXL']) <= 200
            #if close_centers is True:
            #    return True
            if close_centers is True and close_AREA_PXLs is True:
                return True

def check_duplicates_diameter(obj, obj_df):
    for x in obj_df.iterrows():
        if obj['INDEX_COL'] != x[0]:
            encl_diameter_pxl_1 = obj['ENCL_DIAMETER_PXL']
            encl_diameter_pxl_2 = x[1]['ENCL_DIAMETER_PXL']
            diff = encl_diameter_pxl_1 - encl_diameter_pxl_2
            close_diameter = abs(obj['ENCL_DIAMETER_PXL'] - x[1]['ENCL_DIAMETER_PXL']) <= 25
            if close_diameter:
                return True


def getPlateSize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    high_contrast = cv2.convertScaleAbs(gray, alpha=None, beta=0)
    # high_contrast = adjust_gamma(high_contrast, 1.0)

    gamma_test = adjust_gamma(gray, 7.1)
    # cv2.imwrite("./test_pic_green_gamma_0.jpg", gamma_test)

    binary_image = cv2.adaptiveThreshold(gamma_test, 500, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 2)

    contours = get_contours(binary_image)
    plate_df = filter_contours(contours)
    output = draw_contours(gamma_test, plate_df, None, None)
    # write_images(out_dir_path, output, binary_image, high_contrast, "./out")


def main():
    args = parse_args()

    image_paths = get_image_paths(args['image'], args['directory'])
    plate_size = args['plate_size']
    small_plaques = args['small_plaque']
    if small_plaques == 'Y':
        small_plaques = True
    for image_path in image_paths:
        print("Processing " + image_path)
        image = cv2.imread(image_path)
        binary_image, high_contrast, clr_high_contrast = process_image(image, 2.5, small_plaques)
        #       cv2.imshow("Binary image", binary_image)
        contours = get_contours(binary_image)
        green_df, red_df, other_df, plate_df = filter_contours(contours)

        # Filter plaques duplicates (circle in circle)
        # TODO use apply
        green_df['DUPLICATE'] = False
        green_df_copy = green_df
        for green in green_df.iterrows():
            # green_df_copy['DUPLICATE'][green[0]] = check_duplicates(green[1], green_df)
            if check_duplicates(green[1], green_df_copy):
                green_df_copy = green_df_copy.drop(green_df_copy[green_df_copy.index == green[0]].index)
            # test = green_df['INDEX_COL'].apply(lambda x: check_duplicates(green, x))

        # remove duplicates from possible plate contours
        # plate_df['DUPLICATE'] = False
        plate_df_copy = plate_df
        for plate in plate_df.iterrows():
            if (check_duplicates_diameter(plate[1], plate_df_copy)):
                plate_df_copy = plate_df_copy.drop(plate_df_copy[plate_df_copy.index == plate[0]].index)

        max_plate_diameter = plate_df_copy['ENCL_DIAMETER_PXL'].max()
        pxl_per_mm = float(plate_size) / float(max_plate_diameter)
        # green_df_copy['DIAMETER_MM'] = green_df_copy['ENCL_DIAMETER_PXL']
        green_df_copy['ENCL_DIAMETER_MM'] = green_df_copy.apply(lambda x: f"{x['ENCL_DIAMETER_PXL'] * pxl_per_mm:.2f}",
                                                                axis=1)

        green_df_copy['AREA_MM2'] = green_df_copy.apply(lambda x: f"{(math.pi * (((float(x['ENCL_DIAMETER_MM']))/2)**2.0)):.2f}",
                                                                axis=1)


        red_df_copy = red_df
        red_df_copy['ENCL_DIAMETER_MM'] = red_df.apply(lambda x: f"{x['ENCL_DIAMETER_PXL'] * pxl_per_mm:.2f}",
                                                       axis=1)

        other_df_copy = other_df
        other_df_copy['ENCL_DIAMETER_MM'] = other_df.apply(lambda x: f"{x['ENCL_DIAMETER_PXL'] * pxl_per_mm:.2f}",
                                                           axis=1)

        # get Petri dish size and adjust plaques sizes
        # test mask
        contours = green_df_copy['CONTOURS']
        greens_mean_colour = get_mean_grey_colour(high_contrast, contours)

        # for index, c in contours.items():
        #     contour_mean_colour = get_mean_grey_colour(high_contrast, [c])
        #     print(f'{index} mean_colour: {contour_mean_colour}')

        green_df_copy['MEAN_COLOUR'] = green_df_copy.apply(lambda x: get_mean_grey_colour(high_contrast, x['CONTOURS']),
                                                           axis=1)
        filter_dev_colour = green_df_copy.apply(lambda x: abs(x['MEAN_COLOUR'] - greens_mean_colour) > 100, axis=1)
        green_df_copy = green_df_copy[~filter_dev_colour]

        # filter_other = df.apply(lambda x: x['MEAN_COLOUR'] < 100, axis=1)
        # other_df = df[filter_other]
        # wo_other_df = df[~filter_other]
        # getPlateSize(clr_high_contrast)

        # green_df_copy = green_df_copy1.applymap("${0:.2f}".format)
        output = draw_contours(clr_high_contrast, green_df_copy, red_df_copy, other_df_copy, plate_df)

        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)
        write_images(out_dir_path, output, binary_image, high_contrast, image_path)

        # format float values
        # green_df_copy['ENCL_DIAMETER_MM'] = green_df_copy['ENCL_DIAMETER_MM'].apply(lambda x: f"{x:.2f}")
        write_data(out_dir_path, image_path, green_df_copy, red_df_copy, other_df)


def get_mean_grey_colour(img, contours):
    contour_mask = np.zeros(img.shape[:2], dtype="uint8")
    for c in contours:
        cv2.drawContours(contour_mask, [c], -1, 255, -1)

    mean = cv2.mean(img, contour_mask)

    return mean[0]


if __name__ == '__main__':
    main()
