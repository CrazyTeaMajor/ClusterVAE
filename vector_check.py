import torch
from model_main import ClusterVAENet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from metric import normalize_tensor
from dc_dataset import get_dataset

# model_normal = torch.load('./result/result-CNN-MNIST/1v8/best_CNN_ACC-0.9991.pt', map_location='cpu')
# model_onlyO = torch.load('./result/result-CNN-MNIST/1v8onlyO/best_CNN_ACC-1.0000.pt', map_location='cpu')
# model_onlyVAE = torch.load('./result/result-CNN-MNIST/1v8onlyVAE/best_CNN_ACC-0.9858.pt', map_location='cpu')
# model_normal_abs = torch.load('./result/result-CNN-MNIST/1v8abs-0.1-0.9/best_CNN_ACC-0.9976.pt', map_location='cpu')
# model_normal_abs_cos_out = torch.load('./result/result-CNN-MNIST/1v8abs-cosout-0.01-0.99/best_CNN_ACC-0.9991.pt', map_location='cpu')
# model_all = torch.load('./result/result-CNN-MNIST/0-all-9-abs-cosout/best_CNN_ACC-0.9678.pt', map_location='cpu')
# model_all = torch.load('./result/result-CNN-MNIST/0-all-9-vae/best_CNN_ACC-0.7609.pt', map_location='cpu')
# model_all = torch.load('./result/result-CNN-MNIST/0-all-9-onlyO/best_CNN_ACC-0.8798.pt', map_location='cpu')
# wrong_model = torch.load('./result/result-CNN-MNIST/best_CNN_ACC-0.8751.pt', map_location='cpu')
# model_dog_cat = torch.load('./result/result-CNN-dog-cat/vae-stage3/epoch_19_CNN_ACC-0.9024.pt', map_location='cpu')
model_dog_cat = torch.load('./result/result-CNN-dog-cat/stage1/epoch_18_CNN_ACC-0.8592.pt', map_location='cpu')
now_id = 4


def check_vector(vector, vector_list):
    dot_sum = 0
    v2 = normalize_tensor(vector)
    for v in vector_list:
        v1 = normalize_tensor(v)
        dot_sum = dot_sum + torch.abs(torch.dot(v1[0], v2[0]))
    return dot_sum, len(vector_list)


def show_highD_data(vector_list, labels, title_name):
    tsne_data = TSNE().fit_transform(vector_list)
    f = plt.figure(figsize=(10, 10))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(tsne_data[:, 0], tsne_data[:, 1], c=labels)
    plt.title(title_name)
    plt.xlabel('t-SNE1')
    plt.ylabel('t-SNE2')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    ax.axis('tight')
    plt.savefig('./result/{}-vis.jpg'.format(title_name))
    plt.show()


col = ['blue', 'red', 'green', 'yellow', 'pink', 'cyan', 'peru', 'lime', 'purple', 'orange']


def analysis_model(model, title_name):
    vector_now = []
    labels = []
    cnt = -1
    print(model.total_class)
    for label in model.total_class:
        cnt += 1
        for vec in model.multi_class[label]:
            vector_now.append(np.array(vec[0].cpu()))
            labels.append(col[cnt])
    vector_now = np.array(vector_now)
    show_highD_data(vector_now, labels, title_name)


def check_o(model, title_name):
    same_sum = []
    diff_sum = []
    for label1 in model.total_class:
        for vec1 in model.multi_class[label1]:
            for label2 in model.total_class:
                dot_sum, leng = check_vector(vec1, model.multi_class[label2])
                if label1 == label2:
                    same_sum.append(dot_sum / leng)
                else:
                    diff_sum.append(dot_sum / leng)
    plt.subplot(1, 2, 1)
    plt.hist(same_sum)
    plt.subplot(1, 2, 2)
    plt.hist(diff_sum)
    plt.title(title_name)
    plt.savefig("./result/{}-hist-pos.jpg".format(title_name))
    plt.show()


def analysis_all(model, model_name, idx):
    train_dataset, test_dataset, _ = get_dataset(batch_size=1, is_contrastive=False)
    vector_train = []
    label_train = []
    vector_test = []
    label_test = []
    model.eval()
    with torch.no_grad():
        for input_data, target in train_dataset:
            vector_train.append(np.array(model.encoding(input_data)[0][0]))
            label_train.append(col[target.item()])
        for input_data, target in test_dataset:
            vector_test.append(np.array(model.encoding(input_data)[0][0]))
            label_test.append(col[target.item()])
    show_highD_data(np.array(vector_train), label_train, "Train-{}-{}".format(model_name, idx))
    show_highD_data(np.array(vector_test), label_test, "Test-{}-{}".format(model_name, idx))
    corr_matrix = np.zeros((2, 2))
    for cls_1 in range(2):
        for cls_2 in range(2):
            corr_matrix[cls_1][cls_2] = torch.abs(torch.dot(model.multi_class[cls_1][0], model.multi_class[cls_2][0]))
    # 绘制相关矩阵
    fig, ax = plt.subplots()
    # cmap = plt.cm.Blues
    cmap = plt.cm.Reds
    img = ax.imshow(corr_matrix, cmap=cmap)
    # 设置轴标签
    ax.set_xticks(np.arange(len(corr_matrix)))
    ax.set_yticks(np.arange(len(corr_matrix)))
    ax.set_xticklabels(np.arange(1, len(corr_matrix) + 1))
    ax.set_yticklabels(np.arange(1, len(corr_matrix) + 1))
    # 添加colorbar
    plt.colorbar(img)
    # 设置每个格子的标签
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            if i != j:
                color = 'b'
            else:
                color = 'w'
            _ = ax.text(j, i, "{:.4f}".format(corr_matrix[i, j]),
                           ha='center', va='center', color=color)
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig("./result/{}-corr-{}.jpg".format(model_name, idx))
    plt.show()


analysis_all(model_dog_cat, "dog-cat", now_id)
# analysis_model(model_normal, 'Both')
# analysis_model(model_onlyO, 'onlyO')
# analysis_model(model_onlyVAE, 'onlyVAE')
# analysis_model(model_normal_abs, 'abs')
# analysis_model(model_normal_abs_cos_out, 'model_normal_abs_cos_out')
# analysis_model(model_all, 'model_all_onlyO')
# analysis_model(wrong_model, 'wrong4-9')

# check_o(model_normal, 'Both')
# check_o(model_onlyO, 'onlyO')
# check_o(model_onlyVAE, 'onlyVAE')
# check_o(model_normal_abs, 'abs')
# check_o(model_normal_abs_cos_out, 'model_normal_abs_cos_out')
# check_o(model_all, 'model_all_onlyO')
# check_o(wrong_model, 'wrong4-9')
