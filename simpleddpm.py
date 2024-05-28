import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import math
from torch.optim import Adam


device = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 32
BATCH_SIZE = 8
T = 300


# ------------------ Loading CIFAR-100---------------------------------------------

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=data_transform)

    test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=data_transform)
    return torch.utils.data.ConcatDataset([train, test])


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# -------------------------------Forward Process----------------------------------------
# Define beta schedule
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alpha_bar = torch.cumprod(alphas, dim=-1)  # Returns the cumulative product of elements of input
alphas_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
betas_tilda = betas * (1. - alphas_bar_prev) / (1. - alpha_bar)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_bar = torch.sqrt(alpha_bar)
sqrt_one_minus_alphas_bar = torch.sqrt(1. - alpha_bar)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    time_indexes = out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    return time_indexes


def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_bar_t = get_index_from_list(sqrt_alphas_bar, t, x_0.shape)

    sqrt_one_minus_alphas_bar_t = get_index_from_list(sqrt_one_minus_alphas_bar, t, x_0.shape)

    # mean + variance
    x_t = sqrt_alphas_bar_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_bar_t.to(device) * noise.to(device)

    return x_t, noise.to(device)

# -------------------------------U-Net----------------------------------------

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """

    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1],
                                          time_emb_dim)
                                    for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i + 1],
                                        time_emb_dim, up=True)
                                  for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)


model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))


# -------------------------------Sampling----------------------------------------

# @torch.no_grad()
def sample_timestep():
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    # plt.figure(figsize=(1, 10))
    # plt.axis('off')
    img_size = IMG_SIZE
    x_T = torch.randn((1, 3, img_size, img_size), device=device)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        betas_t = get_index_from_list(betas, t, x_T.shape)
        sqrt_one_minus_alphas_bar_t = get_index_from_list(sqrt_one_minus_alphas_bar, t, x_T.shape)
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x_T.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (x_T - betas_t * model(x_T, t) / sqrt_one_minus_alphas_bar_t)

        variance_t = get_index_from_list(betas_tilda, t, x_T.shape)

        if t == 0:
            # The t's are offset from the t's in the paper
            img_gen = model_mean
        else:
            z = torch.randn_like(x_T)
            img_gen = model_mean + torch.sqrt(variance_t) * z

        img_gen_clipped = torch.clamp(img_gen, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            show_tensor_image(img_gen_clipped.detach().cpu())
        plt.show()

    return img_gen


# -------------------------------Train and Test----------------------------------------
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100  # Try more!
num_images = 10
stepsize = int(T / num_images)

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        img = batch[0]
        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        x_noisy, noise = forward_diffusion_sample(img, t, device)
        noise_pred = model(x_noisy, t)
        loss = F.l1_loss(noise, noise_pred)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 and step == 0:
            x_out = sample_timestep()
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
