from DataLoader import DatasetLoader
from torch.utils.data import DataLoader, Dataset, random_split


def loading_datas():
    full_data = DatasetLoader("/home/prajin/Desktop/balot/train_set","/home/prajin/Desktop/balot/train_set.csv", train=True,transform=transform)
    train_size = int(0.9 * len(full_data))
    test_size = len(full_data) - train_size

    train_data, validation_data = random_split(full_data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)

    test_data = DatasetLoader("/home/prajin/Desktop/balot/test_set","/home/prajin/Desktop/balot/test_set.csv", train=True,transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    return train_data, train_loader, validation_data, validation_loader, test_data, test_loader

