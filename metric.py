import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch


# vae-loss:
eps = 1e-8


def ss_loss(org_img, output_img, loss_type='MSE'):
    """
    calc ss two img loss (2D img span to 1D vector)
    :param org_img: input img
    :param output_img: decoder img
    :param loss_type: loss func, KL, MSE, BCE
    :return: loss
    """
    if loss_type == 'KL':
        # KL散度输入需要归一化
        loss = F.kl_div(org_img, output_img, reduction='sum')
    elif loss_type == 'MSE':
        # loss = F.l1_loss(org_img, output_img)
        # print(org_img.shape, output_img.shape)
        loss = F.mse_loss(org_img, output_img, reduction='sum')
    else:
        loss = F.binary_cross_entropy(org_img, output_img, reduction='sum')
    return loss


def vae_loss(recon_x, x, mu, logvar, img_size=28 * 28):
    Loss = ss_loss(x.view(-1, img_size), recon_x.view(-1, img_size), 'MSE')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return Loss + KLD


# o-loss:


def normalize_array(arr):
    """
    将numpy数组模长归一化
    :param arr: 输入数组
    :return: 归一化后的
    """
    norm = np.linalg.norm(arr, ord=2, axis=-1, keepdims=True)
    normalized = arr / norm
    return normalized


def normalize_tensor(ts):
    """
    将一个batch中的所有向量进行模长归一化
    :param batch: 输入张量，形状为 (batch_size, num_features)
    :return: 归一化后的张量
    """
    norms = torch.norm(ts, dim=1, keepdim=True)  # 计算每个向量的模长
    normalized_ts = ts / (norms + eps)  # 进行归一化
    return normalized_ts


def cosine_loss(x, y):
    if torch.all(x == 0) or torch.all(y == 0):
        return 1.0
    cosine_similarity = torch.cosine_similarity(x, y, dim=0).mean()
    return 1 - cosine_similarity


def self_cosine_similarity(v1, v2):
    """
    v1, v2 cos sim (norm before input)
    :param v1
    :param v2
    :return: cos sim
    """
    return F.cosine_similarity(v1, v2, dim=1)
    # v1 = normalize_array(v1)
    # v2 = normalize_array(v2)
    # # 计算两个向量的内积
    # dot_product = np.dot(v1, v2)
    # # 计算两个向量的长度
    # # length_v1 = np.linalg.norm(v1)
    # # length_v2 = np.linalg.norm(v2)
    # # 计算余弦相似度
    # # similarity = dot_product / (length_v1 * length_v2)
    # # 归一化后不用再除以长度
    # similarity = dot_product
    # return similarity


def o_loss(v1, v2, is_same_type):
    """
    o loss and same loss
    :param v1: vector 1
    :param v2: vector 2
    :param is_same_type: 0 for same, 1 for diff
    :return: total loss
    """
    v1 = normalize_tensor(v1)
    v2 = normalize_tensor(v2)
    loss = torch.tensor(0.0, requires_grad=True)
    for i in range(is_same_type.size(0)):
        # if is_same_type[i] == 0:
        #     loss = loss + 1 - torch.abs((v1[i] * v2[i]).sum(dim=0)) + cosine_loss(v1[i], v2[i])
        # else:
        #     loss = loss + 1 + torch.abs((v1[i] * v2[i]).sum(dim=0)) - cosine_loss(v1[i], v2[i])
        if is_same_type[i] == 0:
            loss = loss + 1 - torch.abs((v1[i] * v2[i]).sum(dim=0))
        else:
            loss = loss + 1 + torch.abs((v1[i] * v2[i]).sum(dim=0))
    return loss
    # if is_same_type == 0:
    #     return 1 - (v1 * v2).sum(dim=1) + cosine_loss(v1, v2)
    # else:
    #     return 1 + (v1 * v2).sum(dim=1) - cosine_loss(v1, v2)


def o_new_loss(v, label, model):
    loss = torch.tensor(0.0, requires_grad=True)
    v = normalize_tensor(v)
    for i in range(label.size(0)):
        for cls in model.total_class:
            total_value = 0
            for item in model.multi_class[cls]:
                total_value = total_value + torch.abs(torch.dot(v[i], item))
            total_value = total_value / len(model.multi_class[cls])
            if label[i] == cls:
                loss = loss + 1 - total_value
            else:
                loss = loss + total_value
    return loss


# contrastive-loss:
"""
1. Contrastive Loss
output1: 第一个输入样本的特征表示。
output2: 第二个输入样本的特征表示。
label: 真实标签，通常为0或1，表示样本是否属于同一类（1为同类，0为不同类）。
margin: 分离样本的最小距离，当样本不属于同一类时，应该相隔的最小距离。
2. Triplet Loss
anchor: 锚点样本的特征表示。
positive: 与锚点样本属于同一类的样本特征表示。
negative: 与锚点样本不同类的样本特征表示。
margin: 期望的最小距离，确保锚点与正样本的距离小于锚点与负样本的距离。
3. SimCLR Loss (NT-Xent Loss)
z_i: 第一组增强样本的特征表示。
z_j: 第二组增强样本的特征表示。
temperature: 控制相似性分数的缩放因子，较低的温度会增强相似性评分的差异。
"""


def contrastive_loss(output1, output2, label, margin=1.0):
    distance = F.pairwise_distance(output1, output2)
    loss = label * torch.pow(distance, 2) + (1 - label) * torch.pow(F.relu(margin - distance), 2)
    return loss.mean()


def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    # Compute similarity
    sim_ij = F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2) / temperature
    sim_ji = F.cosine_similarity(z_j.unsqueeze(1), z_i.unsqueeze(0), dim=2) / temperature
    # Create labels
    labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0).to(z_i.device)
    # Compute loss
    loss_ij = F.cross_entropy(sim_ij, labels)
    loss_ji = F.cross_entropy(sim_ji, labels)
    return (loss_ij + loss_ji) / 2


def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_distance = F.pairwise_distance(anchor, positive)
    neg_distance = F.pairwise_distance(anchor, negative)
    loss = F.relu(pos_distance - neg_distance + margin)
    return loss.mean()


# other-metric-function:


def draw_roc(y_true, y_score, title_name):
    plt.figure(figsize=(14, 10), dpi=300)
    plt.rcParams['font.size'] = 12
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(14, 10), dpi=300)
    plt.plot(fpr, tpr, color='lime', linewidth=2,
             label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc="lower right")
    plt.savefig(title_name + "_roc.jpg")
    plt.close()


def draw_cm(y_true, y_predict, title_name, labels_name):
    plt.figure(figsize=(14, 10), dpi=300)
    plt.rcParams['font.size'] = max(7, int(30 / len(labels_name)))
    cm = confusion_matrix(y_true, y_predict)
    norm_cm = np.zeros_like(cm, dtype=float)
    total = []
    for j in range(len(cm)):
        now_total = 0
        for i in range(len(cm)):
            now_total += cm[j, i]
        total.append(now_total)
    for i in range(len(cm)):
        for j in range(len(cm)):
            norm_cm[j, i] = cm[j, i] / total[j]
    plt.figure(figsize=(10, 7))
    plt.imshow(norm_cm, interpolation='nearest', cmap="Blues")
    for i in range(len(norm_cm)):
        for j in range(len(norm_cm)):
            if norm_cm[j, i] >= 0.5:
                c = 'white'
            else:
                c = 'black'
            plt.annotate("{}\n({:.0%})".format(cm[j, i], norm_cm[j, i]), xy=(i, j), horizontalalignment='center',
                         verticalalignment='center', color=c)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)
    plt.yticks(num_local, labels_name)
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    plt.savefig("{}_cm.jpg".format(title_name))
    plt.close()


class parameters:
    def __init__(self, alpha=0.3, beta=0.7, epoch_number=50, learning_rate=1e-4, weight_decay=1e-5,
                 batch_size=2, eps=1e-5, pin_memory=False, num_workers=0, loss_func=None, class_num=10):
        self.alpha = alpha
        self.beta = beta
        self.class_num = class_num
        self.epoch_number = epoch_number
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.eps = eps
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if not use_cuda:
            self.pin_memory = False
            self.num_workers = 0
        assert (not self.pin_memory and self.num_workers == 0) or \
               (self.pin_memory and self.num_workers > 0), "pin memory setting error"
        if loss_func is None:
            self.loss_func = F.mse_loss
        else:
            self.loss_func = loss_func

    def show_all(self):
        print(f"alpha: {self.alpha}, beta: {self.beta}")
        print(f"batch-size: {self.batch_size}, epoch: {self.epoch_number}, eps: {self.eps}")
        print(f"device: {self.device}, weight_decay: {self.weight_decay}, num_worker: {self.num_workers}")
        print(f"class_num: {self.class_num}, pin_memory: {self.pin_memory}, learning_rate: {self.learning_rate}")
