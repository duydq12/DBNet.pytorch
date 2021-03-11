# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:54
# @Author  : zhoujun
import copy
import os
import pathlib

import scipy.io as sio
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.data_loader.modules import *
from src.utils import get_datalist, load, expand_polygon


class BaseDataSet(Dataset):

    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags,
                 transform=None,
                 target_transform=None):
        assert img_mode in ['RGB', 'BRG', 'GRAY']
        self.ignore_tags = ignore_tags
        self.data_list = self.load_data(data_path)
        item_keys = ['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags']
        for item in item_keys:
            assert item in self.data_list[0], 'data_list from load_data must contains {}'.format(
                item_keys)
        self.img_mode = img_mode
        self.filter_keys = filter_keys
        self.transform = transform
        self.target_transform = target_transform
        self._init_pre_processes(pre_processes)

    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def load_data(self, data_path: str) -> list:
        """
        把数据加载为一个list：
        :params data_path: 存储数据的文件夹或者文件
        return a dict ,包含了，'img_path','img_name','text_polys','texts','ignore_tags'
        """
        raise NotImplementedError

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):
        try:
            data = copy.deepcopy(self.data_list[index])
            im = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0)
            if self.img_mode == 'RGB':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            data['img'] = im
            data['shape'] = [im.shape[0], im.shape[1]]
            data = self.apply_pre_processes(data)

            if self.transform:
                data['img'] = self.transform(data['img'])
            data['text_polys'] = data['text_polys'].tolist()
            if len(self.filter_keys):
                data_dict = {}
                for k, v in data.items():
                    if k not in self.filter_keys:
                        data_dict[k] = v
                return data_dict
            else:
                return data
        except:
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


class ICDAR2015Dataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags,
                 transform=None, **kwargs):
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        data_list = get_datalist(data_path)
        t_data_list = []
        for img_path, label_path in data_list:
            data = self._get_annotation(label_path)
            # img = cv2.imread(img_path)
            # pts = data['text_polys'][0].astype(int)
            # cv2.line(img, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 2)
            # cv2.line(img, tuple(pts[1]), tuple(pts[2]), (0, 255, 0), 2)
            # cv2.line(img, tuple(pts[2]), tuple(pts[3]), (0, 255, 0), 2)
            # cv2.line(img, tuple(pts[3]), tuple(pts[0]), (0, 255, 0), 2)
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            if len(data['text_polys']) > 0:
                item = {'img_path': img_path, 'img_name': pathlib.Path(img_path).stem}
                item.update(data)
                t_data_list.append(item)
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes = []
        texts = []
        ignores = []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                pts = params[0:8]
                box = np.array([float(value) for value in pts]).reshape((4, 2))
                boxes.append(box)
                label = params[8]
                texts.append(label)
                ignores.append(label in self.ignore_tags)
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data


class DetDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, ignore_tags,
                 transform=None, **kwargs):
        self.load_char_annotation = kwargs['load_char_annotation']
        self.expand_one_char = kwargs['expand_one_char']
        super().__init__(data_path, img_mode, pre_processes, filter_keys, ignore_tags, transform)

    def load_data(self, data_path: str) -> list:
        """
        从json文件中读取出 文本行的坐标和gt，字符的坐标和gt
        :param data_path:
        :return:
        """
        data_list = []
        for path in data_path:
            content = load(path)
            for gt in tqdm(content['data_list'], desc='read file {}'.format(path)):
                img_path = os.path.join(content['data_root'], gt['img_name'])
                polygons = []
                texts = []
                illegibility_list = []
                language_list = []
                for annotation in gt['annotations']:
                    if len(annotation['polygon']) == 0 or len(annotation['text']) == 0:
                        continue
                    if len(annotation['text']) > 1 and self.expand_one_char:
                        annotation['polygon'] = expand_polygon(annotation['polygon'])
                    polygons.append(annotation['polygon'])
                    texts.append(annotation['text'])
                    illegibility_list.append(annotation['illegibility'])
                    language_list.append(annotation['language'])
                    if self.load_char_annotation:
                        for char_annotation in annotation['chars']:
                            if len(char_annotation['polygon']) == 0 or len(
                                    char_annotation['char']) == 0:
                                continue
                            polygons.append(char_annotation['polygon'])
                            texts.append(char_annotation['char'])
                            illegibility_list.append(char_annotation['illegibility'])
                            language_list.append(char_annotation['language'])
                data_list.append({'img_path': img_path, 'img_name': gt['img_name'],
                                  'text_polys': np.array(polygons),
                                  'texts': texts, 'ignore_tags': illegibility_list})
        return data_list


class SynthTextDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, pre_processes, filter_keys, transform=None,
                 **kwargs):
        self.transform = transform
        self.dataRoot = pathlib.Path(data_path)
        if not self.dataRoot.exists():
            raise FileNotFoundError('Dataset folder is not exist.')

        self.targetFilePath = self.dataRoot / 'gt.mat'
        if not self.targetFilePath.exists():
            raise FileExistsError('Target file is not exist.')
        targets = {}
        sio.loadmat(self.targetFilePath, targets, squeeze_me=True, struct_as_record=False,
                    variable_names=['imnames', 'wordBB', 'txt'])

        self.imageNames = targets['imnames']
        self.wordBBoxes = targets['wordBB']
        self.transcripts = targets['txt']
        super().__init__(data_path, img_mode, pre_processes, filter_keys, transform)

    def load_data(self, data_path: str) -> list:
        t_data_list = []
        for imageName, wordBBoxes, texts in zip(self.imageNames, self.wordBBoxes, self.transcripts):
            item = {}
            wordBBoxes = np.expand_dims(wordBBoxes, axis=2) if (
                    wordBBoxes.ndim == 2) else wordBBoxes
            _, _, numOfWords = wordBBoxes.shape
            text_polys = wordBBoxes.reshape([8, numOfWords], order='F').T  # num_words * 8
            text_polys = text_polys.reshape(numOfWords, 4, 2)  # num_of_words * 4 * 2
            transcripts = [word for line in texts for word in line.split()]
            if numOfWords != len(transcripts):
                continue
            item['img_path'] = str(self.dataRoot / imageName)
            item['img_name'] = (self.dataRoot / imageName).stem
            item['text_polys'] = text_polys
            item['texts'] = transcripts
            item['ignore_tags'] = [x in self.ignore_tags for x in transcripts]
            t_data_list.append(item)
        return t_data_list


if __name__ == '__main__':
    import anyconfig
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from src.utils import parse_config

    config = anyconfig.load('config/icdar2015_resnet18_FPN_DBhead_polyLR.yaml')
    config = parse_config(config)
    dataset_args = config['dataset']['train']['dataset']['args']
    # dataset_args.pop('data_path')
    # data_list = [(r'E:/zj/dataset/icdar2015/train/img/img_15.jpg', 'E:/zj/dataset/icdar2015/train/gt/gt_img_15.txt')]
    train_data = ICDAR2015Dataset(data_path=dataset_args.pop('data_path'),
                                  transform=transforms.ToTensor(),
                                  **dataset_args)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=0)
    for i, data in enumerate(tqdm(train_loader)):
        # img = data['img']
        # shrink_label = data['shrink_map']
        # threshold_label = data['threshold_map']
        #
        # print(threshold_label.shape, threshold_label.shape, img.shape)
        # show_img(img[0].numpy().transpose(1, 2, 0), title='img')
        # show_img((shrink_label[0].to(torch.float)).numpy(), title='shrink_label')
        # show_img((threshold_label[0].to(torch.float)).numpy(), title='threshold_label')
        # img = draw_bbox(img[0].numpy().transpose(1, 2, 0),np.array(data['text_polys']))
        # show_img(img, title='draw_bbox')
        # plt.show()
        pass
