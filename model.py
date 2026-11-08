import torch
from torch import nn
import torch.nn.functional as F

class CustomDefectCNN(nn.Module):
  def __init__(self):
    super(CustomDefectCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1)
    self.fc1 = nn.Linear(32 * 56 * 56, 128)
    self.dropout = nn.Dropout(p=0.5) # 50% forget
    self.fc2 = nn.Linear(128, 2)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.pool(F.relu((self.conv1(x))))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 32 * 56 * 56)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)  #for the purpose of preventing rote memorization
    x = self.fc2(x)
    return x


