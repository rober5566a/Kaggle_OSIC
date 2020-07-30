import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt
from Model.find_file_name import get_filenames


def normalize_imgs(imgs_arr, user_imgs_path):
    user_img_paths = get_filenames(user_imgs_path, 'dcm')
    num_img_ls = []
    for user_img_path in user_img_paths:
        num_img = int(user_img_path.split('/')[-1][:-4])
        num_img_ls.append(num_img)
    num_img_ls.sort()
    # print(num_img_ls)

    dist = len(num_img_ls) / imgs_arr.shape[0]
    dataset_img_paths = []
    for i in range(imgs_arr.shape[0]):
        num_img_path = '{}/{}.dcm'.format(user_imgs_path,
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


if __name__ == "__main__":
    path = "Data/raw/train/ID00014637202177757139317"
    file_extension = "dcm"
    filenames = get_filenames(path, file_extension, isImported=True)
    print(filenames)
    print(len(filenames))

    # imgs_arr = np.zeros([20, 256, 256], dtype=np.float32)
    # imgs_arr = normalize_imgs(imgs_arr=imgs_arr, user_imgs_path=path)

    for filename in filenames:
        ct_dcm = pydicom.dcmread(filename)
        # print(ct_dcm.pixel_array)
        ct_img = ct_dcm.pixel_array
        # plt.imshow(ct_img)
        # plt.show()
        # ct_img = cv2.imread(ct_dcm.pixel_array)
        ct_img = cv2.resize(ct_img, (256, 256))

        # ct_img1 = ct_dcm.pixel_array[::, ::]
        # print(ct_img1)
        # print(ct_img1.shape)

        cv2.imshow("0", ct_img)
        cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue
