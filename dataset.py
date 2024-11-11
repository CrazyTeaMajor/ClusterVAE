import os
from torchvision import transforms
import torch
from sklearn import datasets
import torchvision
import pandas as pd
import numpy as np
import random

cmd_path = os.getcwd()


def data_trans_to_input_image(iris_data, img_size=128, area_size=2):
    blank_pad = np.zeros((img_size, img_size))
    side = (img_size - area_size * 2 - 2) // 2
    blank_pad[side:side + area_size, side:side + area_size] = iris_data[0]
    blank_pad[side + area_size + 2:side + area_size * 2 + 2, side:side + area_size] = iris_data[1]
    blank_pad[side:side + area_size, side + area_size + 2:side + area_size * 2 + 2] = iris_data[2]
    blank_pad[side + area_size + 2:side + area_size * 2 + 2, side + area_size + 2:side + area_size * 2 + 2] = iris_data[
        3]
    blank_pad = blank_pad.reshape((1, 128, 128))
    return blank_pad


def self_data(data_list, sample_num, input_list):
    data1 = random.sample(data_list, sample_num)
    data2 = random.sample(data_list, sample_num)
    for i in data1:
        for j in data2:
            input_list.append(((i, j), 0))


def diff_data(data_1_list, data_2_list, sample_num_1, sample_num_2, input_list):
    data_1 = random.sample(data_1_list, sample_num_1)
    data_2 = random.sample(data_2_list, sample_num_2)
    for i in data_1:
        for j in data_2:
            input_list.append(((i, j), 1))


def get_dataset(data_type, batch_size=16, IMG_w=28, IMG_h=28, is_contrastive=False, cls='all', is_gray=False):
    random.seed(42)
    """
    get normal/contrastive learning format dataset
    :param is_contrastive: pair data or org-data
    :param IMG_h: img height
    :param IMG_w: img width
    :param data_type: MNIST, E-MNIST, Iris, Fashion-MNIST, (CIFAR-10)
    :return: train_set, test_set
    """
    num_workers = 0
    pin_memory = False
    img_trans = transforms.Compose([
        transforms.Resize((IMG_w, IMG_h)),
        transforms.ToTensor(),
    ])

    if data_type == 'MNIST':
        class_num = 10
        train_dataset = torchvision.datasets.MNIST(root='{}/data'.format(cmd_path), train=True, download=True,
                                                   transform=img_trans)
        test_dataset = torchvision.datasets.MNIST(root='{}/data'.format(cmd_path), train=False, transform=img_trans)
    elif data_type == 'Fashion-MNIST':
        class_num = 10
        train_dataset = torchvision.datasets.FashionMNIST(root='{}/data'.format(cmd_path), train=True, download=True,
                                                          transform=img_trans)
        test_dataset = torchvision.datasets.FashionMNIST(root='{}/data'.format(cmd_path), train=False, download=True,
                                                         transform=img_trans)
    elif data_type == 'E-MNIST':
        class_num = 26
        train_dataset = torchvision.datasets.EMNIST(root='{}/data'.format(cmd_path), train=True, download=True,
                                                    transform=img_trans, split="letters")
        test_dataset = torchvision.datasets.EMNIST(root='{}/data'.format(cmd_path), train=False, download=True,
                                                   transform=img_trans, split="letters")
        # poor performance/ wasted
    elif data_type == 'CIFAR-10':
        IMG_size = (32, 32)
        class_num = 10
        if is_gray:
            print("ASD")
            test_trans = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(IMG_size),
                transforms.ToTensor(),
            ])
            train_trans = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(IMG_size),
                transforms.ToTensor(),
            ])
        else:
            test_trans = transforms.Compose([
                transforms.Resize(IMG_size),
                transforms.ToTensor(),
            ])
            train_trans = transforms.Compose([
                transforms.Resize(IMG_size),
                transforms.ToTensor(),
            ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_trans)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_trans)
    elif data_type == 'Iris':
        class_num = 3
        iris_datas = datasets.load_iris()
        iris_df = pd.DataFrame(iris_datas.data,
                               columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
        data_iris = iris_df.values
        labels = iris_datas.target
        data_0 = []
        data_1 = []
        data_2 = []
        for i in range(150):
            if labels[i] == 0:
                data_0.append(data_trans_to_input_image(data_iris[i]))
            elif labels[i] == 1:
                data_1.append(data_trans_to_input_image(data_iris[i]))
            else:
                data_2.append(data_trans_to_input_image(data_iris[i]))
        train_list = []
        test_list = []
        for i in range(10):
            test_list.append((data_0[i], 0))
            test_list.append((data_1[i], 1))
            test_list.append((data_2[i], 2))
        for i in range(10, 50):
            train_list.append((data_0[i], 0))
            train_list.append((data_1[i], 1))
            train_list.append((data_2[i], 2))
        train_data = train_list
        test_data = test_list
        # train_loader = torch.utils.data.DataLoader(train_list, batch_size=batch_size, shuffle=True,
        #                                            num_workers=num_workers, pin_memory=pin_memory)
        # test_loader = torch.utils.data.DataLoader(test_list, batch_size=batch_size, shuffle=True,
        #                                           num_workers=num_workers, pin_memory=pin_memory)
        # dataset too small/ wasted
    else:
        class_num = -1
        print("Unknown dataset!")
        exit(0)

    if is_contrastive is True:
        if data_type == "Iris":
            data_c0 = []
            data_c1 = []
            self_data(data_0, 25, data_c0)
            self_data(data_1, 25, data_c0)
            self_data(data_2, 25, data_c0)
            diff_data(data_0, data_1, 18, 18, data_c1)
            diff_data(data_0, data_2, 18, 18, data_c1)
            diff_data(data_1, data_0, 18, 18, data_c1)
            diff_data(data_1, data_2, 18, 18, data_c1)
            diff_data(data_2, data_0, 18, 18, data_c1)
            diff_data(data_2, data_1, 18, 18, data_c1)
            random.shuffle(data_c0)
            random.shuffle(data_c1)
            train_data = []
            test_data = []
            for i in range(380):
                test_data.append(data_c0[i])
                test_data.append(data_c1[i])
            for i in range(380, len(data_c0)):
                train_data.append(data_c0[i])
            for i in range(380, len(data_c1)):
                train_data.append(data_c1[i])
            random.shuffle(train_data)
        else:
            #  30% 0-0 0
            #  10% 1-0 1
            train_list = {}
            for item in train_dataset:
                if cls == 'all':
                    if train_list.get(item[1]) is None:
                        train_list[item[1]] = list([item[0]])
                    else:
                        train_list[item[1]].append(item[0])
                elif item[1] in cls:
                    if train_list.get(item[1]) is None:
                        train_list[item[1]] = list([item[0]])
                    else:
                        train_list[item[1]].append(item[0])
            test_list = {}
            label_list = []
            for item in test_dataset:
                if cls == 'all':
                    if test_list.get(item[1]) is None:
                        label_list.append(item[1])
                        test_list[item[1]] = list([item[0]])
                    else:
                        test_list[item[1]].append(item[0])
                elif item[1] in cls:
                    if test_list.get(item[1]) is None:
                        label_list.append(item[1])
                        test_list[item[1]] = list([item[0]])
                    else:
                        test_list[item[1]].append(item[0])

            train_data = []
            test_data = []
            print(label_list)
            if cls == 'all':
                alpha = 0.03
                beta = 0.01
            else:
                if len(cls) == 2:
                    alpha = 0.05
                    beta = 0.05
                else:
                    alpha = 0.003
                    beta = 0.001
            for label in label_list:
                now_list = train_list[label]
                self_data(now_list, int(alpha * len(now_list)), train_data)
                now_test = test_list[label]
                self_data(now_test, int(alpha * len(now_test)), test_data)
                for label_2 in label_list:
                    if label == label_2:
                        continue
                    now_list_2 = train_list[label_2]
                    diff_data(now_list, now_list_2, int(beta * len(now_list)), int(beta * len(now_list_2)), train_data)
                    now_test_2 = test_list[label_2]
                    diff_data(now_test, now_test_2, int(beta * len(now_test)), int(beta * len(now_test_2)), test_data)
    else:
        if data_type == "Iris":
            train_data = train_list
            test_data = test_list
        else:
            if cls == 'all':
                train_data = train_dataset
                test_data = test_dataset
            else:
                train_data = []
                test_data = []
                cnt = [0 for _ in range(len(cls))]
                for item in train_dataset:
                    if item[1] in cls:
                        if cnt[item[1]] < 0.005 * len(train_dataset):
                            cnt[item[1]] += 1
                            train_data.append((item[0], cls.index(item[1])))
                for item in test_dataset:
                    if item[1] in cls:
                        test_data.append((item[0], cls.index(item[1])))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader, class_num
