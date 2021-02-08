import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_size = args.input_size[0] * args.input_size[1]
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, args.output_size)
        self.log_var = nn.Linear(256, args.output_size)

    def forward(self, input_image):
        r"""
        input_image: B * C * H * W
        """
        B, C, H, W = input_image.shape
        input_image = input_image.reshape(B, -1)  # Flatten
        hidden = self.net(input_image)
        mu, log_var = self.mu(hidden), self.log_var(hidden)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu), std, eps  # the sampled results


class VAEDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        output_size = args.output_size[0] * args.output_size[1]
        self.net = nn.Sequential(
            nn.Linear(args.input_size, 256),
            nn.Linear(256, 512),
            nn.Linear(512, output_size),
            nn.Sigmoid()
        )
        self.args = args
    
    def forward(self, input_z):
        return self.net(input_z)


def VAE_loss(recon_x, x, mu, log_var):
    # import pdb; pdb.set_trace()
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD
