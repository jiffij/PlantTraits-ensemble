import torch
import torchvision.transforms as transforms
from plant_dataset import PlantTraitsDataset
from torch.utils.data import DataLoader

def get_train_valid_test_loader(config, train_X_img_df, train_X_data_df, valid_X_img_df, valid_X_data_df, test_X_img_df, test_X_data_df, train_y_df, valid_y_df, test_y_df, seed=1):
    image_dim = config['MODEL']['INPUT_DIM']
    mean = config['DATA']['mean']
    std = config['DATA']['std']
    img_dir = config['DATA']['img_dir']
    batch_size = config['TRAIN']['BATCH_SIZE']
    training_transformation = transforms.Compose([
        transforms.Resize((image_dim,image_dim)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),  # Rotate the image by a random angle between -10 and 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, saturation, and hue
        # transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10), # last added
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.RandomResizedCrop(image_dim, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Randomly crop and resize the image
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        # transforms.Normalize(training_set_mean, training_set_std)
    ])
    valid_transformation = transforms.Compose([
        transforms.Resize((image_dim,image_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        # transforms.Normalize(training_set_mean, training_set_std)
    ])
    test_transformation = transforms.Compose([
        transforms.Resize((image_dim,image_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        # transforms.Normalize(training_set_mean, training_set_std)
    ])
    train_set = PlantTraitsDataset(img_dir, train_X_img_df, train_X_data_df, train_y_df, transform=training_transformation)
    validation_set = PlantTraitsDataset(img_dir, valid_X_img_df, valid_X_data_df, valid_y_df, transform=valid_transformation)
    test_set = PlantTraitsDataset(img_dir, test_X_img_df, test_X_data_df, test_y_df, transform=test_transformation)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader