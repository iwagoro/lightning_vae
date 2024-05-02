import torch
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import RichProgressBar , EarlyStopping
from network import AutoEncoder,Encoder,Decoder
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch import Trainer
from torchinfo import summary


transform = transforms.Compose([
    transforms.Pad(2),  # 2ピクセルのパディング
    transforms.ToTensor()  # PIL ImageをTensorに変換
])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True,num_workers=27)
test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False,num_workers=27)

torch.set_float32_matmul_precision('high')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoEncoder()
model.to(device)

logger = TensorBoardLogger("tb_logs", name="my_model")
strategy=DDPStrategy(find_unused_parameters=False)
progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green_yellow",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="cyan",
        processing_speed="#ff1493",
        metrics="#ff1493",
        metrics_text_delimiter="\n",
    )
)
trainer = Trainer(
    accelerator="cuda",
    default_root_dir="./",
    callbacks=[EarlyStopping(monitor="val_loss", mode="min"),progress_bar],
    logger=logger,
    max_epochs=100,
    # strategy=strategy,
    devices=1,
)
trainer.fit(model, train_loader,test_loader)    