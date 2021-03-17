import os
import time

import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm

from src.post_processing import get_post_processing
from src.utils import get_file_list

ort_session = onnxruntime.InferenceSession("/home/rabiloo/project/alpr/DBNet.pytorch/weights/"
                                           "model_best_dynamic.onnx")


def preprocess_input(img_ori, short_size):
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img_resize = resize_image(img, short_size)

    img = img_resize.astype(np.float32) / 255
    img = img - [0.485, 0.456, 0.406]
    img = img / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img.astype(np.float32), 0), img_resize


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


def warp_perspective_img(img_, rect):
    rect = rect.astype(np.float32)
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)

    plate_img = cv2.warpPerspective(img_, matrix, (max_width, max_height), borderValue=0)
    return plate_img


def predict(short_size: int = 640):
    config = {'type': 'SegDetectorRepresenter',
              'args': {'thresh': 0.3, 'box_thresh': 0.6, 'max_candidates': 1000,
                       'unclip_ratio': 1.5}}
    post_process = get_post_processing(config)
    input_folder = "/home/rabiloo/project/alpr/DBNet.pytorch/datasets/predict"
    for img_path in tqdm(get_file_list(input_folder, p_postfix=['.jpg'])):
        if img_path.find("1613959261.1915412.jpg") < 0:
            continue
        img_ori = cv2.imread(img_path)
        h, w = img_ori.shape[:2]
        batch = {'shape': [(h, w)]}

        input_predict, img_resize = preprocess_input(img_ori, short_size)
        start = time.time()

        ort_inputs = {ort_session.get_inputs()[0].name: input_predict}
        ort_outs = ort_session.run(None, ort_inputs)[0]

        box_list, score_list = post_process(batch, ort_outs,img_resize,  is_output_polygon=False)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
            box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []
        t = time.time() - start
        print(t)
        plate = warp_perspective_img(img_ori, box_list[0])
        # img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], box_list)
        cv2.imshow(os.path.basename(img_path), plate)
        cv2.imshow("origin", img_ori)
        cv2.waitKey(0)


# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
predict()
