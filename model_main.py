import torch
import torch.nn as nn
from model_CNN import VAE_CNN
from model_ViT import VAE_ViT


class ClusterVAENet(nn.Module):
    def __init__(self, backbone='CNN', channels_img=1, img_size=28, latent_dim=64, cls_list=None, device='cpu'):
        super(ClusterVAENet, self).__init__()
        if backbone == 'CNN':
            in_channels = channels_img
            hidden_dim = 32
            latent_dim = 64
            self.net = VAE_CNN(img_size, in_channels, hidden_dim, latent_dim)
        elif backbone == 'ViT':
            patch_size = 4
            in_channels = channels_img
            latent_dim = 64
            num_heads = 12
            num_layers = 12
            self.net = VAE_ViT(img_size, patch_size, in_channels, latent_dim, num_heads, num_layers)
        self.multi_class = {}
        self.total_class = []
        if cls_list is not None:
            cnt = 0
            self.total_class = cls_list
            for cls in cls_list:
                now = torch.zeros(latent_dim)
                now[cnt] = 1
                now = now.unsqueeze(0)
                now = now.to(device)
                self.multi_class[cls] = now
                cnt += 1

    def forward(self, x):
        return self.net(x)

    def encoding(self, x):
        return self.net.encoder(x)

    def decoding(self, x):
        return self.net.decoder(x)

    def update(self, new_v, learning_rate):
        for cls in self.total_class:
            now = self.multi_class[cls]
            self.multi_class[cls] = now + (new_v[cls] - now) * learning_rate

    def inference(self, x):
        # single test! not batch!
        v, _ = self.net.encoder(x)
        prob = torch.zeros(len(self.total_class))
        for cls in self.total_class:
            total_value = 0
            for item in self.multi_class[cls]:
                total_value += torch.abs(torch.dot(v[0], item))
            total_value /= len(self.multi_class[cls])
            prob[cls] = total_value
        normalized_prob = (prob - prob.min()) / (prob.max() - prob.min())
        return normalized_prob
