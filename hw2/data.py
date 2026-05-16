from torch.utils.data import Dataset, DataLoader,Subset
from torchvision import datasets, transforms

def get_oxfordiiipets_classification(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    VAL_RATIO = 0.2

    trainval_dataset = datasets.OxfordIIITPet(root='./data', split='trainval', transform=transform, download=False)
    test_dataset = datasets.OxfordIIITPet(root='./data', split='test', transform=transform, download=False)

    val_dataset = Subset(trainval_dataset, range(int(len(trainval_dataset) * (1 - VAL_RATIO)), len(trainval_dataset)))
    train_dataset = Subset(trainval_dataset, range(int(len(trainval_dataset) * (1 - VAL_RATIO))))


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def transform_fn(img, mask):
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # mask transform
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.PILToTensor()
    ])
    img = img_transform(img)
    mask = mask_transform(mask).squeeze(0).long()

    mask -= 1

    return img, mask

class PetSegDataset(Dataset):
    def __init__(self, root, split):
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        img, mask = transform_fn(img, mask)
        return img, mask


def get_oxfordiiipets_segmentation(batch_size=64):
    VAL_RATIO = 0.2
    
    trainval_dataset = PetSegDataset('./data', 'trainval')
    train_dataset = Subset(trainval_dataset, range(int(len(trainval_dataset) * (1 - VAL_RATIO)), len(trainval_dataset)))
    val_dataset = Subset(trainval_dataset, range(int(len(trainval_dataset) * (1 - VAL_RATIO))))
    test_dataset = PetSegDataset('./data', 'test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    return train_loader, val_loader, test_loader