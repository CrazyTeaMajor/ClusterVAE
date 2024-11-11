import os
import torch.utils.data as td
import numpy as np
from torchvision import transforms
from PIL import Image
import random
import torch


def read_one_img(img_path):
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512]),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    open_img = Image.open(img_path)
    img = open_img.convert("RGB")
    img = trans(np.array(img))
    return img


def self_data(data_list, sample_num, input_list):
    data1 = random.sample(data_list, sample_num)
    data2 = random.sample(data_list, sample_num)
    for i in data1:
        for j in data2:
            input_list.append(((read_one_img(i), read_one_img(j)), 0))


def diff_data(data_1_list, data_2_list, sample_num_1, sample_num_2, input_list):
    data_1 = random.sample(data_1_list, sample_num_1)
    data_2 = random.sample(data_2_list, sample_num_2)
    for i in data_1:
        for j in data_2:
            input_list.append(((read_one_img(i), read_one_img(j)), 1))


class CD_all_dataset:
    def __init__(self):
        cmd_path = os.getcwd()
        self.image_dog = []
        self.image_cat = []
        self.train_image = []
        self.train_label = []
        self.test_image = []
        self.test_label = []
        self.n_classes = 2
        img_list = os.listdir("{}/data/dog-cat/PetImages/Dog".format(cmd_path))
        for file_name in img_list:
            self.image_dog.append("{}/data/dog-cat/PetImages/Dog/{}".format(cmd_path, file_name))
        img_list = os.listdir("{}/data/dog-cat/PetImages/Cat".format(cmd_path))
        for file_name in img_list:
            self.image_cat.append("{}/data/dog-cat/PetImages/Cat/{}".format(cmd_path, file_name))
        len_cat = len(self.image_cat)
        # len_dog = len(self.image_dog)
        len_train = int(0.9 * len_cat)
        for i in range(int(0.1 * len_train)):
            self.train_image.append(self.image_cat[i])
            self.train_label.append(1)
            self.train_image.append(self.image_dog[i])
            self.train_label.append(0)
        for i in range(len_train, len_cat): 
            self.test_image.append(self.image_cat[i])
            self.test_label.append(1)
            self.test_image.append(self.image_dog[i])
            self.test_label.append(0)

    def get_train_test_data(self):
        return self.train_image, self.train_label, self.test_image, self.test_label

    def get_contrastive_learning_data(self, ratio=0.05, batch_size=16, num_workers=0, pin_memory=False):
        len_train = int(0.9 * len(self.image_cat))
        tot_len = len(self.image_cat)
        train_cross = []
        test_cross = []
        self_data(self.image_dog[:len_train], ratio * len_train, train_cross)
        self_data(self.image_cat[:len_train], ratio * len_train, train_cross)
        diff_data(self.image_dog[:len_train], self.image_cat[:len_train], 2 * ratio * len_train, 2 * ratio * len_train,
                  train_cross)
        len_test = tot_len - len_train
        self_data(self.image_dog[len_train: tot_len], ratio * len_test, test_cross)
        self_data(self.image_cat[len_train: tot_len], ratio * len_test, test_cross)
        diff_data(self.image_dog[len_train: tot_len], self.image_cat[len_train: tot_len], 2 * ratio * len_test,
                  2 * ratio * len_test, train_cross)
        train_loader = torch.utils.data.DataLoader(train_cross, batch_size=batch_size, shuffle=True,
                                                   num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(
            test_cross, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader


class CD_dataset(td.Dataset):
    def __init__(self, data_list, label_list):
        """
        cat dog dataset
        :param data_image:
        :param data_label:
        :param n_classes: the number of classes, 0 for regression task
        """
        self.image = data_list
        self.label = label_list
        self.n_classes = 2

    def readImg(self, img_path, index):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512]),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        open_img = Image.open(img_path)
        img = open_img.convert("RGB")
        img = trans(np.array(img))
        return img, self.label[index]

    def __getitem__(self, index):
        img_data, now_classes = self.readImg(self.image[index], index)
        # label_data = np.zeros(self.n_classes)
        # label_data[now_classes] = 1
        # return img_data, label_data
        return img_data, now_classes

    def __len__(self):
        return len(self.label)


def get_dataset(batch_size, is_contrastive):
    all_data = CD_all_dataset()
    if is_contrastive:
        train_data, test_data = all_data.get_contrastive_learning_data(batch_size=batch_size)
        return train_data, test_data, 2
    else:
        train_img, train_label, test_img, test_label = all_data.get_train_test_data()
        train_dataset = CD_dataset(train_img, train_label)
        test_dataset = CD_dataset(test_img, test_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=0, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        return train_loader, test_loader, 2
