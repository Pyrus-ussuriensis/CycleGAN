import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split
from glob import glob
from PIL import Image
from pathlib import Path
import torch, random
from torchvision import transforms

def make_transform(size=256):
    return transforms.Compose([
        transforms.Resize(286),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),  # [0,1] -> [-1,1]
    ])

class Monet(Dataset):
    def __init__(self, dirA, dirB, tfm):
        pA, pB = Path(dirA), Path(dirB)
        exts = {".jpg",".jpeg",".png",".bmp"}
        self.A = sorted([p for p in pA.iterdir() if p.suffix.lower() in exts])
        self.B = sorted([p for p in pB.iterdir() if p.suffix.lower() in exts])
        self.tfm = tfm
    def __len__(self): return max(len(self.A), len(self.B))
    def __getitem__(self, i):
        a = Image.open(self.A[i % len(self.A)]).convert("RGB")
        b = Image.open(self.B[random.randrange(len(self.B))]).convert("RGB")
        return self.tfm(a), self.tfm(b)

class Monet2PhotoDM(L.LightningDataModule):
    def __init__(self, data_dir: str="data/monet2photo",
                 batch_size: int=1, size: int=256, num_workers: int=4):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.size = size
        self.num_workers = num_workers
        self.tfm = make_transform(size)

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            full = Monet(self.data_dir/"trainA", self.data_dir/"trainB", self.tfm)
            g = torch.Generator().manual_seed(42)
            self.train_set, self.val_set = random_split(full, [0.95, 0.05], generator=g)
        if stage in (None, "test"):
            self.test_set = Monet(self.data_dir/"testA", self.data_dir/"testB", self.tfm)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers>0, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers>0)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers>0)


if __name__ == "__main__":
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    import matplotlib.pyplot as plt
    class InverseNormalize(T.Normalize):
        def __init__(self, mean, std):
            inv_std = [1/s for s in std]
            inv_mean = [-m/s for m,s in zip(mean, std)]
            super().__init__(inv_mean, inv_std)

    denorm_tfm = InverseNormalize((0.5,)*3, (0.5,)*3)
    monet = Monet2PhotoDM(data_dir="data/monet2photo")
    monet.setup(stage="fit")
    loader = monet.train_dataloader()
    for x, y in loader:
        img = denorm_tfm(x[0]).permute(1,2,0).cpu().numpy()
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        #pil = TF.to_pil_image(denorm_tfm(x[0].squeeze(0).cpu()))
        #pil.show()
        break
