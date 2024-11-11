import os.path
import torch.optim as optim
# from dataset import get_dataset
from dc_dataset import get_dataset
from metric import *
# from model_main import UnitBallNet
from model_CNN import CNNClassification


def train(model, epoch, data, optimizer, para):
    model.train()
    train_loss = 0
    for batch_idx, (input_data, target) in enumerate(data):
        input_data, target = input_data.to(para.device), target.to(para.device)
        output = model(input_data)
        loss = F.cross_entropy(output, target)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Train Epoch {epoch}: {batch_idx * len(input_data)}/{len(data.dataset)} loss:{loss.item()}")
    return train_loss / len(data.dataset)


def experiment(model, data, epoch, data_name, model_name, labels_list, para, vis=False):
    model.eval()
    correct_number = 0
    predict_label = []
    true_label = []
    predict_prob = []
    with torch.no_grad():
        for input_data, target in data:
            input_data, target = input_data.to(para.device), target.to(para.device)
            output = model(input_data)
            _, predict_index = torch.max(output, 1)
            correct_number += predict_index.eq(target.view_as(predict_index)).sum().item()
            for i in range(output.size(0)):
                predict_prob.append(np.array(output[i].cpu().squeeze()))
                predict_index_now = np.array(predict_index[i].cpu().squeeze())
                predict_label.append(predict_index_now)
                target_now = np.array(target[i].cpu())
                true_label.append(target_now)
    acc = correct_number / len(data.dataset)
    print(f"Test: acc={acc}\n")
    if vis:
        draw_cm(true_label, predict_label,
                "result/result-{}/{}_epoch-{}".format(data_name, model_name, epoch), labels_list)
    return acc


def main_trainer(model_name, data_name, epoch_number, labels_list, para):
    print("Using model: {}, Dataset: {}, Total epoch: {}".format(model_name, data_name, epoch_number))
    para.show_all()
    conv_layers = 3
    img_size = 512
    kernel_sizes = [3, 3, 3, 3, 3]
    input_channels = 3
    output_channels = [32, 64, 128, 256, 128]
    paddings = [1, 1, 1, 1, 1]
    strides = [1, 1, 1, 1, 1]
    pool_kernel_sizes = [2, 2, 2, 2, 2]
    num_classes = 2
    model = CNNClassification(conv_layers, img_size, kernel_sizes, input_channels, output_channels,
                                       paddings, strides, pool_kernel_sizes, num_classes)
    model = model.to(para.device)
    train_dataset, test_dataset, _ = get_dataset(batch_size=para.batch_size, is_contrastive=False)
    data_name = "{}-{}".format(model_name, data_name)
    if not os.path.exists("result/result-{}".format(data_name)):
        os.mkdir("result/result-{}".format(data_name))
    print("LEN: {} {}".format(len(train_dataset), len(test_dataset)))
    loss_list = []
    acc_list = []
    best_acc = 0
    patient = 0
    optimizer = optim.Adam(model.parameters(), lr=para.learning_rate, weight_decay=para.weight_decay, eps=para.eps)
    sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    for epoch in range(1, epoch_number + 1):
        loss = train(model, epoch, train_dataset, optimizer, para)
        sch.step()
        acc = experiment(model, test_dataset, epoch, data_name, model_name, labels_list, para, vis=True)
        patient += 1
        if acc > best_acc:
            patient = 0
            best_acc = acc
            torch.save(model, "result/result-{}/best_{}_ACC-{:.4f}.pt".format(data_name, model_name, acc))
        else:
            print("Patient: {}".format(patient))
        if patient >= epoch_number * 0.2:
            print("Patient up!")
            break
        loss_list.append(loss)
        acc_list.append(acc)
    plt.figure(figsize=(9, 9), dpi=300)
    plt.plot([i for i in range(len(loss_list))], loss_list, label='loss')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("result/result-{}/{}_result_loss.jpg".format(data_name, model_name))
    plt.show()
    plt.figure(figsize=(9, 9), dpi=300)
    plt.plot([i for i in range(len(acc_list))], acc_list, label='acc')
    plt.xlabel("epochs")
    plt.ylabel("acc")
    plt.savefig("result/result-{}/{}_result_acc.jpg".format(data_name, model_name))
    plt.show()
    torch.save(model, "result/result-{}/last_{}.pt".format(data_name, model_name))
    print("Best ACC: {:.4f}".format(best_acc))
