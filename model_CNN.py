import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, img_size, in_channels, hidden_dim, latent_dim):
        super(CNNEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1)  # (batch_size, hidden_dim, H/2, W/2)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)  # (batch_size, hidden_dim*2, H/4, W/4)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=4, stride=2, padding=1)  # (batch_size, hidden_dim*4, H/8, W/8)
        self.conv4 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)  # (batch_size, hidden_dim*4, H/8, W/8)
        self.conv5 = nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=4, stride=2, padding=1)  # (batch_size, hidden_dim*4, H/8, W/8)
        self.fc_input_size = self._calculate_fc_input_size()
        self.fc_mu = nn.Linear(self.fc_input_size, latent_dim)  # Mean
        self.fc_logvar = nn.Linear(self.fc_input_size, latent_dim)  # Log variance

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def _calculate_fc_input_size(self):
        x = torch.randn(1, self.in_channels, self.img_size, self.img_size)
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
        fc_input_size = x.view(1, -1).size(1)
        return fc_input_size


class CNNDecoder(nn.Module):
    def __init__(self, img_size, latent_dim, hidden_dim, out_channels, output_size):
        super(CNNDecoder, self).__init__()
        self.img_size = img_size
        self.fc = nn.Linear(latent_dim, hidden_dim * 4 * 4 * 4)  # Latent to hidden
        self.deconv1 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 4, kernel_size=4, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, kernel_size=4, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=2)
        self.deconv5 = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), -1, 4, 4)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv4(z))
        x_recon = torch.sigmoid(self.deconv5(z))  # Output image
        return x_recon


class VAE_CNN(nn.Module):
    def __init__(self, img_size, in_channels, hidden_dim, latent_dim):
        super(VAE_CNN, self).__init__()
        self.encoder = CNNEncoder(img_size, in_channels, hidden_dim, latent_dim)
        self.decoder = CNNDecoder(img_size, latent_dim, hidden_dim, in_channels, self.encoder.fc_input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


class CNNClassification(nn.Module):
    """
    introduction: This class is for classification tasks.
    :param conv_layers: input the number of CNN layers.
    :param img_size: input the size of images (n means n x n image)
    :param kernel_sizes: input the kernel size of each convolution layer
    :param input_channels: input the original input channels of images,
                           1 means grey image and 3 means RGB image
    :param output_channels: input the output channels of each convolution layer
    :param paddings: input the paddings you need in each layer
    :param strides: input the strides you need in each layer
    :param pool_kernel_sizes: input the size of kernel of maxpooling
                              in each convolution layer after convolution
    :param num_classes: input the object number of classes

    example: conv_layers=2,
             img_size=28,
             kernel_sizes=[3, 3],
             input_channels=1,  # 因为MNIST是灰度图像，所以通道数为1
             output_channels=[32, 64],
             paddings=[1, 1],
             strides=[1, 1],
             pool_kernel_sizes=[2, 2]
             num_classes=10
             model = CNNClassification(conv_layers, img_size, kernel_sizes, input_channels, output_channels,
                                       paddings, strides, pool_kernel_sizes, num_classes)
    """

    def __init__(self, conv_layers, img_size, kernel_sizes, input_channels, output_channels,
                 paddings, strides, pool_kernel_sizes, num_classes):
        super(CNNClassification, self).__init__()

        self.conv_layers = conv_layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.img_size = img_size

        self.conv_blocks = nn.ModuleList()
        self.pooling = nn.ModuleList()

        in_channels = input_channels

        for i in range(conv_layers):
            conv_layer = nn.Conv2d(
                in_channels,
                output_channels[i],
                kernel_sizes[i],
                padding=paddings[i],
                stride=strides[i]
            )
            self.conv_blocks.append(conv_layer)

            pool_layer = nn.MaxPool2d(pool_kernel_sizes[i])
            self.pooling.append(pool_layer)

            in_channels = output_channels[i]

        self.fc_input_size = self._calculate_fc_input_size()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        for i in range(self.conv_layers):
            x = nn.functional.relu(self.conv_blocks[i](x))
            x = self.pooling[i](x)

        x = x.view(-1, self.fc_input_size)
        x = self.fc(x)

        return x

    def _calculate_fc_input_size(self):
        dummy_input = torch.randn(1, self.input_channels, self.img_size, self.img_size)
        with torch.no_grad():
            for i in range(self.conv_layers):
                dummy_input = self.conv_blocks[i](dummy_input)
                dummy_input = self.pooling[i](dummy_input)
        fc_input_size = dummy_input.view(1, -1).size(1)
        return fc_input_size


class CNNRegression(nn.Module):
    """
    introduction: This class is for regression tasks.
    :param conv_layers: input the number of CNN layers.
    :param input_dim: input the dimension of regression tasks
    :param kernel_sizes: input the kernel size of each convolution layer
    :param output_channels: input the output channels of each convolution layer
    :param paddings: input the paddings you need in each layer
    :param strides: input the strides you need in each layer
    :param pool_kernel_sizes: input the size of kernel of maxpooling
                              in each convolution layer after convolution
    :param output_dim: input the object dim of regression tasks

    example: conv_layers=2,
             input_dim=13,
             kernel_sizes=[3, 3],
             output_channels=[32, 64],
             paddings=[1, 1],
             strides=[1, 1],
             pool_kernel_sizes=[1, 1], #在回归任务里，一般情况必须让pool_kernel_sizes=1
             output_dim=1
             model = CNNRegression(conv_layers, input_dim, kernel_sizes, output_channels,
                                   paddings, strides, pool_kernel_sizes, output_dim)
    """

    def __init__(self, conv_layers, input_dim, kernel_sizes, output_channels,
                 paddings, strides, pool_kernel_sizes, output_dim):
        super(CNNRegression, self).__init__()

        self.conv_layers = conv_layers
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.conv_blocks = nn.ModuleList()
        self.pooling = nn.ModuleList()

        in_channels = input_dim

        for i in range(conv_layers):
            conv_layer = nn.Conv2d(
                in_channels,
                output_channels[i],
                kernel_sizes[i],
                padding=paddings[i],
                stride=strides[i]
            )
            self.conv_blocks.append(conv_layer)

            pool_layer = nn.MaxPool2d(pool_kernel_sizes[i], stride=1)
            self.pooling.append(pool_layer)

            in_channels = output_channels[i]

        self.fc_input_size = self._calculate_fc_input_size()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        for i in range(self.conv_layers):
            x = nn.functional.relu(self.conv_blocks[i](x))
            x = self.pooling[i](x)

        x = x.view(-1, self.fc_input_size)
        x = self.fc(x)

        return x

    def _calculate_fc_input_size(self):
        dummy_input = torch.randn(1, self.input_dim, 1, 1)
        with torch.no_grad():
            for i in range(self.conv_layers):
                dummy_input = self.conv_blocks[i](dummy_input)
                dummy_input = self.pooling[i](dummy_input)
        fc_input_size = dummy_input.view(1, -1).size(1)
        return fc_input_size
