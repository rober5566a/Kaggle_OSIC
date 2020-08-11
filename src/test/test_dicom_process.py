from Model.find_file_name import get_filenames
from Model.BoundaryDescriptor import *
from Model.dataset_img_process import *
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


def main():
    path = "Data/raw/train"
    file_extension = "dcm"
    df = pd.read_csv("doc/train.csv")
    usrs_id = list(set(df.iloc[:, 0]))
    problem_usrs_id = ['ID00026637202179561894768',
                       'ID00128637202219474716089', 'ID00298637202280361773446']
    # problem_usrs_id = ["ID00419637202311204720264", "ID00105637202208831864134",
    #                    "ID00099637202206203080121", "ID00094637202205333947361"]
    # ID00086637202203494931510 ID00082637202201836229724 ID00122637202216437668965 ID00283637202278714365037
    # ID00094637202205333947361/3
    # ID00026637202179561894768

    # usrs_id = problem_usrs_id
    # for problem_usr_id in problem_usrs_id:
    #     usrs_id.remove(problem_usr_id)

    filenames = []
    for usr_id in usrs_id:
        usr_imgs_path = ('{}/{}'.format(path, usr_id))
        dataset_paths = get_dataset_paths(usr_imgs_path, NUM_DIVIDED=30)
        filenames.extend(dataset_paths)

    # filenames = ['Data/raw/train/ID00105637202208831864134/1.dcm']
    # Data/raw/train/ID00094637202205333947361/3.dcm
    # Data/raw/train/ID00419637202311204720264/18.dcm
    # Data/raw/train/ID00419637202311204720264/20.dcm

    # print(filenames)
    print('num of filename', len(filenames))

    # imgs_arr = np.zeros([20, 256, 256], dtype=np.float32)
    # imgs_arr = normalize_imgs(imgs_arr=imgs_arr, usr_imgs_path=path)

    try:
        global filename
        for i, filename in enumerate(filenames):

            ct_dcm = pydicom.dcmread(filename)

            ct_img = transform_ctdata(ct_dcm, -1500, -600)
            cv2.imshow("0", ct_img)

            if np.average(ct_img) < 25 or np.average(ct_img) > 200 or (np.average(ct_img) == ct_img[:, :]).all():
                print('pass')
                i += 1
                ct_dcm = pydicom.dcmread(filenames[i])
                ct_img = transform_ctdata(ct_dcm, -1500, -600)
                cv2.imshow("0", ct_img)
                # output_img = ct_img
                # continue
            lung_img = get_lung_img(ct_img.copy(), isShow=True)
            output_img = get_square_img(lung_img)

            empty_img = np.ones(
                (output_img.shape[0]+2, output_img.shape[1]+2), dtype='uint8') * output_img[0, 0]
            empty_img[1:-1, 1:-1] = output_img
            output_img = empty_img

            lung_contour = get_biggest_countour(
                output_img, THRESH_VALUE=1, isShow=False)
            output_img = remove_img_nosie(
                output_img, lung_contour, meanBackGround=True)
            cv2.imshow("1", output_img)
            # cv2.waitKey(0)
            # img_detect = cv2.cvtColor(img_lung, cv2.COLOR_BGR2GRAY)
            if np.average(ct_img[:, :]) == np.average(output_img[:, :]):
                print("background is black:", filename)
                cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                continue
    except KeyboardInterrupt:
        print(filename)
        # cv2.imwrite('1.png', lung_img)


if __name__ == "__main__":

    main()
