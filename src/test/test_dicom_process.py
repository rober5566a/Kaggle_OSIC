from Model.find_file_name import get_filenames
from Model.BoundaryDescriptor import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pydicom
import pandas as pd


def normalize_imgs(imgs_arr, usr_imgs_path):
    usr_img_paths = get_filenames(usr_imgs_path, 'dcm')
    num_img_ls = []
    for usr_img_path in usr_img_paths:
        num_img = int(usr_img_path.split('/')[-1][:-4])
        num_img_ls.append(num_img)
    num_img_ls.sort()
    # print(num_img_ls)

    dist = len(num_img_ls) / imgs_arr.shape[0]
    dataset_img_paths = []
    for i in range(imgs_arr.shape[0]):
        num_img_path = '{}/{}.dcm'.format(usr_imgs_path,
                                          num_img_ls[int(i * dist)])
        dataset_img_paths.append(num_img_path)

    print(dataset_img_paths)

    for i, dataset_img_path in enumerate(dataset_img_paths):
        ct_dcm = pydicom.dcmread(dataset_img_path)
        # print(ct_dcm.pixel_array)
        num_mag = int(ct_dcm.pixel_array.shape[1] / imgs_arr.shape[2])
        ct_img = ct_dcm.pixel_array[::num_mag, ::num_mag] / 65535
        imgs_arr[i] = ct_img

    return imgs_arr


def transform_ctdata(img, windowWidth, windowCenter, CONVERT_DCM2GRAY=True):
    ct_slope = float(ct_dcm.RescaleSlope)
    ct_intercept = float(ct_dcm.RescaleIntercept)
    ct_img = ct_intercept + ct_dcm.pixel_array * ct_slope

    minWindow = - float(abs(windowCenter)) + 0.5*float(abs(windowWidth))
    new_img = (ct_img - minWindow) / float(windowWidth)
    # print(np.average(new_img))

    if np.average(new_img) > 1:  # if this img most of part are white
        try:
            minWindow = - float(abs(ct_dcm.WindowCenter)) + \
                0.5*float(abs(windowWidth))
        except TypeError:
            minWindow = - float(abs(ct_dcm.WindowCenter[0])) + \
                0.5*float(abs(windowWidth))
        new_img = (ct_img - minWindow) / float(windowWidth)

    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    if CONVERT_DCM2GRAY is True:
        new_img = (new_img * 255).astype('uint8')
    return new_img


def test(img):
    contours = get_contours_binary(img, whiteGround=False)
    new_contours = []
    contour_area_ls = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > (img.size * 0.1) and contour_area < (img.size * 0.9) and contour.size > 8:
            contour_area_ls.append(contour_area)
            new_contours.append(contour)

    if len(contour_area_ls) != 0:
        lung_contour = [
            new_contours[contour_area_ls.index(max(contour_area_ls))]]

    else:
        # need to fix : no contour fit the constrain
        lung_contour = []
        print(filename)

    img_crop = np.zeros(img.shape, dtype="uint8")
    img_crop = cv2.drawContours(
        img_crop.copy(), lung_contour, -1, 255, thickness=-1)
    img_crop = cv2.bitwise_and(img, img_crop)

    # img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2BGR)
    # img_lung = cv2.drawContours(
    #     img.copy(), lung_contour, -1, (0, 255, 0), thickness=-1)

    # return img_lung
    return img_crop


def get_dataset_paths(usr_imgs_path, NUM_DIVIDED=20):
    usr_img_paths = get_filenames(usr_imgs_path, 'dcm')
    num_img_ls = []
    for usr_img_path in usr_img_paths:
        num_img = int(usr_img_path.split('/')[-1][:-4])
        num_img_ls.append(num_img)
    num_img_ls.sort()
    # print(num_img_ls)

    dist = len(num_img_ls) / NUM_DIVIDED
    dataset_img_paths = []
    for i in range(NUM_DIVIDED):
        num_img_path = '{}/{}.dcm'.format(usr_imgs_path,
                                          num_img_ls[int(i * dist)])
        dataset_img_paths.append(num_img_path)

    return dataset_img_paths


if __name__ == "__main__":
    path = "Data/raw/train"
    file_extension = "dcm"
    df = pd.read_csv("doc/train.csv")
    usrs_id = list(set(df.iloc[:, 0]))
    problem_usrs_id = ["ID00419637202311204720264", "ID00105637202208831864134",
                       "ID00099637202206203080121", "ID00094637202205333947361"]
    # usrs_id = problem_usrs_id
    for problem_usr_id in problem_usrs_id:
        usrs_id.remove(problem_usr_id)

    filenames = []
    for usr_id in usrs_id:
        usr_imgs_path = ('{}/{}'.format(path, usr_id))
        dataset_paths = get_dataset_paths(usr_imgs_path, NUM_DIVIDED=30)
        filenames.extend(dataset_paths)

    # print(filenames)
    print('num of filename', len(filenames))

    # imgs_arr = np.zeros([20, 256, 256], dtype=np.float32)
    # imgs_arr = normalize_imgs(imgs_arr=imgs_arr, usr_imgs_path=path)

    try:
        for filename in filenames:
            ct_dcm = pydicom.dcmread(filename)

            ct_img = transform_ctdata(ct_dcm, -1500, -600)
            img_lung = test(ct_img)

            cv2.imshow("0", img_lung)
            # cv2.waitKey(0)
            # img_detect = cv2.cvtColor(img_lung, cv2.COLOR_BGR2GRAY)
            if img_lung[0, 0] < 50:
                print("background is black:", filename)
                cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                continue
    except KeyboardInterrupt:
        print(filename)
        cv2.imwrite('1.png', img_lung)
