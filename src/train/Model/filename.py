import os


def get_filenames(root_dir):
    output = []
    for user_id in os.listdir(root_dir):
        output.extend([os.path.join(root_dir, user_id, filename)
                       for filename in os.listdir(os.path.join(root_dir, user_id))])

    return output


if __name__ == '__main__':
    get_filenames('')
