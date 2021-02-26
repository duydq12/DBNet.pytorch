import glob
import re
from os import makedirs
from os.path import basename
from shutil import copy2

import magic
import tqdm
from sklearn.model_selection import train_test_split


def load_data(path_data="/Users/duydq/_data/plate_license/data_plate_detect/", type_label=""):
    _lst_img = glob.glob(path_data + "*.jpg")
    if type_label:
        _lst_label = [item.replace(".jpg", "_" + type_label + ".txt") for item in _lst_img]
    else:
        _lst_label = [item.replace(".jpg", ".txt") for item in _lst_img]
    return _lst_img, _lst_label


def process_data():
    annotation_str = "plate"
    lst_img, lst_label = load_data()
    for i, path_img in tqdm.tqdm(enumerate(lst_img)):
        path_label = lst_label[i]
        t = magic.from_file(path_img)
        w, h = re.findall('(\d+)x(\d+)', t)[-1]

        # I = cv2.imread(path_img)

        new_string = ""
        with open(path_label, 'r') as fread:
            content = fread.read()
            lst_row = content.split("\n")
            for row in lst_row:
                if not row:
                    continue
                data = row.strip().split(',')
                x_value = [int(float(_) * int(w)) for _ in data[1:5]]
                y_value = [int(float(_) * int(h)) for _ in data[5:9]]

                # cv2.line(I, (x_value[0], y_value[0]), (x_value[1], y_value[1]),
                #          (255, 255, 0), thickness=2)
                # cv2.line(I, (x_value[1], y_value[1]), (x_value[2], y_value[2]), (255, 255, 0),
                #          thickness=2)
                # cv2.line(I, (x_value[2], y_value[2]), (x_value[3], y_value[3]), (255, 255, 0),
                #          thickness=2)
                # cv2.line(I, (x_value[3], y_value[3]), (x_value[0], y_value[0]), (255, 255, 0),
                #          thickness=2)
                # cv2.imshow("Anh goc", I)
                # cv2.waitKey()

                for i, item in enumerate(x_value):
                    new_string += "{},{},".format(item, y_value[i])
                new_string += annotation_str
                new_string += "\n"
            new_string = new_string[:-1]

        with open(path_label[:-4] + "_int.txt", 'w') as fread:
            fread.write(new_string)


def split_data(path_out="/Users/duydq/_data/plate_license/data_dbnet_float/"):
    train_img_path = path_out + "train/img/"
    train_gt_path = path_out + "train/gt/"
    test_img_path = path_out + "test/img/"
    test_gt_path = path_out + "test/gt/"
    makedirs(train_img_path, exist_ok=True)
    makedirs(train_gt_path, exist_ok=True)
    makedirs(test_img_path, exist_ok=True)
    makedirs(test_gt_path, exist_ok=True)

    lst_img, lst_label = load_data(type_label="float")
    x_train, x_test, y_train, y_test = train_test_split(lst_img, lst_label, test_size=0.33,
                                                        random_state=42)

    str_train = ""
    str_test = ""
    for i, item in enumerate(x_train):
        path_img = "/content/data_dbnet/train/img/" + basename(item)
        path_label = "/content/data_dbnet/train/gt/" + basename(y_train[i])

        str_train += path_img + "\t" + path_label + "\n"
    str_train = str_train.strip()
    for i, item in enumerate(x_test):
        path_img = "/content/data_dbnet/test/img/" + basename(item)
        path_label = "/content/data_dbnet/test/gt/" + basename(y_test[i])

        str_test += path_img + "\t" + path_label + "\n"
    str_test = str_test.strip()
    with open(path_out + "train.txt", "w") as fwrite:
        fwrite.write(str_train)
    with open(path_out + "test.txt", "w") as fwrite:
        fwrite.write(str_test)

    for i, item in enumerate(x_train):
        copy2(item, train_img_path)
        copy2(y_train[i], train_gt_path)

    for i, item in enumerate(x_test):
        copy2(item, test_img_path)
        copy2(y_test[i], test_gt_path)


if __name__ == '__main__':
    split_data()
