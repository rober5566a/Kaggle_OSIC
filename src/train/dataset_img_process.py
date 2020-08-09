import numpy as np
import cv2
import pydicom
from Model.find_file_name import get_filenames
from Model.BoundaryDescriptor import *


def get_dataset_paths(usr_imgs_path, NUM_DIVIDED=20):
    usr_img_paths = get_filenames(usr_imgs_path, 'dcm')
    num_img_list = []
    for usr_img_path in usr_img_paths:
        num_img = int(usr_img_path.split('/')[-1][:-4])
        num_img_list.append(num_img)
    num_img_list.sort()
    # print(num_img_list)

    dist = len(num_img_list) / NUM_DIVIDED
    dataset_img_paths = []
    for i in range(NUM_DIVIDED):
        num_img_path = '{}/{}.dcm'.format(usr_imgs_path,
                                          num_img_list[int(i * dist)])
        dataset_img_paths.append(num_img_path)

    return dataset_img_paths


def remove_img_nosie(img, contours, isShow=False):
    '''
        Only save contours part, else place become back.
        ===
        create a np.zeros array(black),
        use cv2.drawContours() make contours part become 255 (white),
        final, use cv2.gitwise_and() to remove noise for img
    '''
    crop_img = np.zeros(img.shape, dtype="uint8")
    crop_img = cv2.drawContours(
        crop_img.copy(), contours, -1, 255, thickness=-1)
    crop_img = cv2.bitwise_and(img, crop_img)

    if isShow is True:
        cv2.imshow('remove_img_nosie', crop_img)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    return crop_img


def remove_black_frame(img, contour, isShow=False):
    feature = calc_contour_feature(img, contour)
    x, y, w, h = feature[0][3]
    img_center = [int(img.shape[0]/2)+1, int(img.shape[1]/2)+1]

    if img_center[0] > y:
        w = (img_center[1] - x) * 2 - 2
        h = (img_center[0] - y) * 2 - 2
        feature[0][3] = (x, y, w, h)
    else:
        x += w
        y += h
        w = (x - (img_center[1])) * 2 - 2
        h = (y - (img_center[0])) * 2 - 2
        feature[0][3] = (2*img_center[1] - x, 2*img_center[0] - y, w, h)

    img = get_crop_img_list(
        img, feature, extra_W=-1, extra_H=-1, isShow=False)[0]
    new_img = np.ones(
        (img.shape[0]+2, img.shape[1]+2), dtype='uint8') * 255
    new_img[1:-1, 1:-1] = img

    if isShow:
        cv2.imshow('remove_black_frame', new_img)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    return new_img


def get_biggest_countour(img, isShow=False):
    contours = get_contours_binary(img, THRESH_VALUE=100, whiteGround=False)
    new_contours = []
    contour_area_list = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > (img.size * 0.05) and contour_area < (img.size * 0.9) and contour.size > 8:
            contour_area_list.append(contour_area)
            new_contours.append(contour)

    if len(contour_area_list) != 0:
        biggest_contour = [
            new_contours[contour_area_list.index(max(contour_area_list))]]
    else:
        # need to fix : no contour fit the constrain
        biggest_contour = []
        # print(filename)

    if isShow is True:
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        bgr_img = cv2.drawContours(
            bgr_img, biggest_contour, -1, (0, 255, 0), 3)
        cv2.imshow('lung_contour', bgr_img)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

    return biggest_contour


def get_lung_img(img, isShow=False):
    lung_contour = get_biggest_countour(img, isShow=False)

    # detect image has black frame or not
    if len(lung_contour) != 0 and np.average(img[0:10, 0:10]) < 50:
        img = remove_black_frame(img, lung_contour, isShow=False)
        lung_contour = get_biggest_countour(img)

    lung_img = remove_img_nosie(img, lung_contour, isShow=False)
    features = calc_contour_feature(lung_img, lung_contour)
    lung_img = get_crop_img_list(lung_img, features)[0]

    if isShow:
        cv2.imshow('lung_img', lung_img)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

    return lung_img


def get_square_img(img):
    square_size = max(img.shape[:])
    square_center = int(square_size / 2)
    output_img = np.zeros(
        (square_size, square_size), dtype='uint8')
    start_point_x = square_center - int(img.shape[0]/2)
    start_point_y = square_center - int(img.shape[1]/2)
    output_img[start_point_x: start_point_x + img.shape[0],
               start_point_y: start_point_y + img.shape[1]] = img

    return output_img
