import os.path
import torch.optim as optim
from dataset import get_dataset
from metric import *
from model_main import UnitBallNet


def train(model, epoch, data, optimizer, para, single_dataset):
    model.train()
    train_loss = 0
    for batch_idx, (input_data, target) in enumerate(data):
        input_data, target = input_data.to(para.device), target.to(para.device)
        recon, u, logvar = model(input_data)
        # print(recon.shape, input_data.shape)
        loss_vae = vae_loss(recon, input_data, u, logvar, img_size=3 * 32 * 32)
        loss = para.alpha * loss_vae + para.beta * o_new_loss(u, target, model)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            if batch_idx > 0:
                refresh(model, single_dataset, para)
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
    total_class = torch.zeros((para.class_num, 64))
    total_class = total_class.to(para.device)
    cnt = [0 for _ in range(para.class_num)]
    with torch.no_grad():
        for input_data, target in dataset:
            input_data = input_data.to(para.device)
            target = target.to(para.device)
            v, _ = model.encoding(input_data)
            cls = target.item()
            cnt[cls] += 1
            total_class[cls] = total_class[cls] + v
        for i in range(para.class_num):
            total_class[i] = total_class[i] / cnt[i]
        total_class = normalize_tensor(total_class)
        model.update(total_class, para.learning_rate)
    print("Model saving has refreshed.")


def old_refresh(model, dataset, para):
    print("Old-Refresh model saving...")
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
                if cnt[cls] <= 10:
                    model.multi_class[cls].append(v)
                elif cnt[cls] == 11:
                    total += 1
            if total > para.class_num:
                break
    print("Model saving has refreshed.")


def main_trainer(model_name, data_name, epoch_number, labels_list, para, continue_train=False, model_path=''):
    print("Using model: {}, Dataset: {}, Total epoch: {}".format(model_name, data_name, epoch_number))
    para.show_all()
    if continue_train is not False:
        print("Continue {}".format(model_path))
        model = torch.load(model_path)
    else:
        model = UnitBallNet(backbone=model_name, channels_img=3, img_size=32, cls_list=labels_list, device=para.device)
        model = model.to(para.device)
    train_dataset, _, _ = get_dataset(data_name, batch_size=para.batch_size, is_contrastive=False, cls=labels_list)
    single_train_dataset, test_dataset, _ = get_dataset(data_name, batch_size=1, is_contrastive=False, cls=labels_list)
    data_name = "{}-{}".format(model_name, data_name)
    if not os.path.exists("result/result-{}".format(data_name)):
        os.mkdir("result/result-{}".format(data_name))
    print("LEN: {} {}".format(len(single_train_dataset), len(test_dataset)))
    loss_list = []
    acc_list = []
    best_acc = 0
    patient = 0
    optimizer = optim.Adam(model.parameters(), lr=para.learning_rate, weight_decay=para.weight_decay, eps=para.eps)
    sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
    for epoch in range(1, epoch_number + 1):
        loss = train(model, epoch, train_dataset, optimizer, para, single_train_dataset)
        refresh(model, single_train_dataset, para)
        sch.step()
        acc = experiment(model, test_dataset, epoch, data_name, model_name, labels_list, para)
        patient += 1
        if acc > best_acc:
            patient = 0
            best_acc = acc
            experiment(model, test_dataset, epoch, data_name, model_name, labels_list, para, vis=True)
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


"""
4 - 32
        self.deconv1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 4, kernel_size=4, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, kernel_size=4, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=2)
        self.deconv5 = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1)
"""
