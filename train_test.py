import os.path
import torch.optim as optim
from dataset import get_dataset
from metric import *
from model_main import ClusterVAENet


def train(model, epoch, data, optimizer, para):
    model.train()
    train_loss = 0
    for batch_idx, (input_data, target) in enumerate(data):
        x1, x2 = input_data
        x1, x2, target = x1.to(para.device), x2.to(para.device), target.to(para.device)
        recon1, u1, logvar1 = model(x1)
        recon2, u2, logvar2 = model(x2)
        loss_1 = vae_loss(recon1, x1, u1, logvar1, img_size=3 * 32 * 32)
        loss_2 = vae_loss(recon2, x2, u2, logvar2, img_size=3 * 32 * 32)
        loss = para.alpha * 0.5 * (loss_1 + loss_2) + para.beta * o_loss(u1, u2, target)
        # loss = 0.5 * (loss_1 + loss_2)
        # loss = o_loss(u1, u2, target)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 2000 == 0:
            print(f"Train Epoch {epoch}: {batch_idx * len(x1)}/{len(data.dataset)} loss:{loss.item()}")
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
            output = model.inference(input_data)
            _, predict_index = torch.max(output, 0)
            correct_number += predict_index.eq(target.view_as(predict_index)).sum().item()
            predict_prob.append(np.array(output.cpu().squeeze()))
            predict_index = np.array(predict_index.cpu().squeeze())
            predict_label.append(predict_index)
            target = np.array(target.cpu())
            true_label.append(target)
    acc = correct_number / len(data.dataset)
    print(f"Test: acc={acc}\n")
    if vis:
        draw_cm(true_label, predict_label,
                "result/result-{}/{}_epoch-{}".format(data_name, model_name, epoch), labels_list)
    return acc


def refresh(model, dataset, para):
    print("Refresh model saving...")
    model.eval()
    model.total_class = []
    model.multi_class = {}
    cnt = np.zeros(para.class_num)
    total = 0
    with torch.no_grad():
        for input_data, target in dataset:
            input_data = input_data.to(para.device)
            v, _ = model.encoding(input_data)
            v = normalize_tensor(v)
            cls = target.item()
            if model.multi_class.get(cls) is None:
                model.total_class.append(cls)
                model.multi_class[cls] = list([v])
                cnt[cls] = 1
            else:
                cnt[cls] += 1
                if cnt[cls] <= 100:
                    model.multi_class[cls].append(v)
                elif cnt[cls] == 101:
                    total += 1
            if total > para.class_num:
                break
    print("Model saving has refreshed.")


def main_trainer(model_name, data_name, epoch_number, labels_list, para):
    print("Using model: {}, Dataset: {}, Total epoch: {}".format(model_name, data_name, epoch_number))
    para.show_all()
    model = ClusterVAENet(backbone=model_name, channels_img=3, img_size=32)
    model = model.to(para.device)
    train_dataset, _, _ = get_dataset(data_name, batch_size=para.batch_size, is_contrastive=True, cls=labels_list)
    single_train_dataset, test_dataset, _ = get_dataset(data_name, batch_size=1, is_contrastive=False, cls=labels_list)
    data_name = "{}-{}".format(model_name, data_name)
    if not os.path.exists("result/result-{}".format(data_name)):
        os.mkdir("result/result-{}".format(data_name))
    print("LEN: {} {} {}".format(len(train_dataset), len(single_train_dataset), len(test_dataset)))
    loss_list = []
    acc_list = []
    best_acc = 0
    patient = 0
    optimizer = optim.Adam(model.parameters(), lr=para.learning_rate, weight_decay=para.weight_decay, eps=para.eps)
    sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    for epoch in range(1, epoch_number + 1):
        loss = train(model, epoch, train_dataset, optimizer, para)
        refresh(model, single_train_dataset, para)
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
