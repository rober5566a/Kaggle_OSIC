import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_usr_info(df, usr_id, col_name):
    usr_info = []
    for i, df_id in enumerate(df.loc[:, col_id_name]):
        if df_id == usr_id:
            usr_info.append(df.loc[i, col_name])
            if type(df.loc[i, col_name]) is str:
                break

    return usr_info


def create_mean_df(df, df_ids, cols_name):
    usrs_id = list(set(df_ids))
    col_id = cols_name.pop(0)
    df_dict = {col_id: usrs_id}
    for col_name in cols_name:
        usrs_info = []
        for usr_id in usrs_id:
            usr_info = get_usr_info(df, usr_id, col_name)

            if len(usr_info) > 1:
                usr_info = np.average(usr_info)
            else:
                usr_info = usr_info[0]
            usrs_info.append(usr_info)

        df_dict.update({col_name: usrs_info})

    new_df = pd.DataFrame(df_dict)

    return new_df


if __name__ == "__main__":
    df = pd.read_csv('Data/raw/train.csv')
    col_id_name = 'Patient'

    cols_name = ['Patient', 'Weeks', 'FVC',
                 'Percent', 'Age', 'Sex', 'SmokingStatus']

    new_df = create_mean_df(df, df.loc[:, col_id_name], cols_name)

    new_df.to_csv("output/df_mean.csv")
    # print(new_df)
