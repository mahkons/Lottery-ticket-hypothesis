import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import plotly.graph_objects as go
import numpy as np
from PIL.ImageOps import grayscale

from VAE import VAE

BATCH_SIZE = 32
image_height = 84
image_width = 84
z_dim = 256
plot_data = list()

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3, required=False)
    parser.add_argument('--restart', type=lambda x: (str(x).lower() in ['true','1', 'yes']), default=False, required=False)
    parser.add_argument('--device', type=str, default='cpu', required=False)
    parser.add_argument('--lr', type=float, default=1e-3, required=False)
    return parser 


def train(epochs, restart, device, dataloader, lr):
    model = VAE(image_height=image_height, image_width=image_width, image_channels=1, z_dim=z_dim, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not restart:
        model.load_state_dict(torch.load("generated/vae.torch", map_location='cpu'))

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        pbar.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))

        for idx, (images, _) in tqdm(enumerate(dataloader)):
            images = images.to(device)
            recon_images, mu, logstd = model(images)

            loss = VAE.calculate_loss(recon_images, images, mu, logstd)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            plot_data.append(loss.item() / BATCH_SIZE)
            pbar.write("Loss: {:.3f}".format(loss.item() / BATCH_SIZE))

    model.save_model('generated/vae.torch')


if __name__ == "__main__":
    dataset = datasets.ImageFolder(root='generated/assault', transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    args = create_parser().parse_args()
    train(args.epochs, args.restart, torch.device(args.device), dataloader, args.lr)

    plot = go.Figure()
    plot.add_trace(go.Scatter(x=np.arange(len(plot_data)), y=np.array(plot_data)))
    plot.show() 


