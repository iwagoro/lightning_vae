import torch
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch.loggers import TensorBoardLogger
from network import AutoEncoder
import matplotlib.pyplot as plt
import pytorch_lightning as pl
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = AutoEncoder()
model.to(device)

logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = pl.Trainer(
    accelerator="cuda",
    default_root_dir="./",
    logger=logger,
    max_epochs=100,
    strategy="ddp",
    devices=4,
    # check_val_every_n_epoch=1  # 毎エポックで検証
)
trainer.fit(model, train_loader,test_loader)    