from metric import parameters
import random
import torch
import train_test
import train_test_new
import train_test_dc
import compare_train
import compare_train_dc

para = parameters(batch_size=32, learning_rate=1e-4, alpha=0.00001, beta=0.99999, epoch_number=40)

data_name_list = ["MNIST", "Fashion-MNIST", "E-MNIST", "Iris", "CAT-DOG", "CIFAR-10"]
class_number_list = [10, 10, 26, 3, 2, 10]
labels_list = [[i for i in range(10)],
               ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
               [chr(i) for i in range(ord('a'), ord('z') + 1)], [str(i) for i in range(3)],
               ['dog', 'cat'],
               ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]]

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    print("Start train-test normal")
    for i in range(1):
        data_name = data_name_list[i]
        # para.class_num = class_number_list[i]
        # cls_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        cls_list = [0, 1]
        para.class_num = len(cls_list)
        compare_train_dc.main_trainer("CNN-std", "DOG-CAT", para.epoch_number, cls_list, para)
        # train_test.main_trainer("CNN", "MNIST", para.epoch_number, cls_list, para)
        # train_test_new.main_trainer("CNN", "CIFAR-10", para.epoch_number, cls_list, para)
        # train_test_dc.main_trainer("CNN", para.epoch_number, cls_list, para,
        #                            continue_train=True,
        #                            model_path='./result/result-CNN-dog-cat/vae-stage3/epoch_19_CNN_ACC-0.9024.pt')

