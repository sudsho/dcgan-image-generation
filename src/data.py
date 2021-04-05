"""CelebA dataloader. Center-crop -> resize 64 -> normalize to [-1, 1]."""
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_transform(image_size=64):
    return transforms.Compose([
        transforms.CenterCrop(178),  # CelebA aligned faces are ~178x218
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_celeba_loader(root, image_size=64, batch_size=128, num_workers=4,
                     subset_size=None, shuffle=True):
    tfm = build_transform(image_size)
    # ImageFolder expects root/<class>/*.jpg, so we wrap CelebA dir
    dataset = datasets.ImageFolder(root=root, transform=tfm)
    if subset_size is not None and subset_size < len(dataset):
        dataset = Subset(dataset, list(range(subset_size)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return loader
