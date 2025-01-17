import torch.nn as nn
import torch.nn.functional as F


class M5(nn.Module):
  def __init__(self, n_input=1, n_output=3, stride=16, n_channel=32):
    super().__init__()
    self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
    self.bn1 = nn.BatchNorm1d(n_channel)
    self.pool1 = nn.MaxPool1d(4)
    self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
    self.bn2 = nn.BatchNorm1d(n_channel)
    self.pool2 = nn.MaxPool1d(4)
    self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
    self.bn3 = nn.BatchNorm1d(2 * n_channel)
    self.pool3 = nn.MaxPool1d(4)
    self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
    self.bn4 = nn.BatchNorm1d(2 * n_channel)
    self.pool4 = nn.MaxPool1d(4)
    self.fc1 = nn.Linear(2 * n_channel, n_output)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(self.bn1(x))
    x = self.pool1(x)
    x = self.conv2(x)
    x = F.relu(self.bn2(x))
    x = self.pool2(x)
    x = self.conv3(x)
    x = F.relu(self.bn3(x))
    x = self.pool3(x)
    x = self.conv4(x)
    x = F.relu(self.bn4(x))
    x = self.pool4(x)
    x = F.avg_pool1d(x, x.shape[-1])
    x = x.permute(0, 2, 1)
    x = self.fc1(x)
    return F.log_softmax(x, dim=2)
  
  
class ResUnit(nn.Module):
  def __init__(self, input_channels, output_channels):
    super(ResUnit, self).__init__()
    self.conv1 = nn.Conv1d(input_channels, output_channels, 3, padding=1)
    self.bn1 = nn.BatchNorm1d(num_features=output_channels)
    self.conv2 = nn.Conv1d(output_channels, output_channels, 3, padding=1)
    self.bn2 = nn.BatchNorm1d(num_features=output_channels)
  def forward(self, x):
    x = self.bn1(self.conv1(x))
    x = F.relu(x)

    x = self.bn2(self.conv2(x))
    x = F.relu(x)

    return x
  
  
class ResBlock(nn.Module):
  def __init__(self, input_channels, output_channels, num_layers, is_pool=True):
    super(ResBlock, self).__init__()
    self.input_channels = input_channels
    self.output_channels = output_channels
    self.num_layers = num_layers
    self.pool =  nn.MaxPool1d(4)
    self.is_pool = is_pool
    
  def forward(self, x):
    for i in range(self.num_layers):
      conv = nn.Conv1d(x.shape[1], self.output_channels, 1)
      res = conv(x) 
      if i==0:
        unit = ResUnit(self.input_channels, self.output_channels)
      else:
        unit = ResUnit(self.output_channels, self.output_channels)
      x = unit(x)
      x = res + x
    if self.is_pool:
      x = self.pool(x)
    return x

  
class M34Res(nn.Module):
    def __init__(self, n_input, n_output):
        super(M34Res, self).__init__()
        self.n_output = n_output
        self.conv1 = nn.Conv1d(n_input, 48, 80, 4,padding=38)
        self.bn1 = nn.BatchNorm1d(48)
        self.pool1 = nn.MaxPool1d(4)

        self.block_1 = ResBlock(48, 48, 3)

        self.block_2 = ResBlock(48, 96, 4)

        self.block_3 = ResBlock(96, 192, 6)

        self.block_4 = ResBlock(192, 384, 3, is_pool=False)

        self.conv2 = nn.Conv1d(384, n_output, 3, padding=1)
        self.avgPool = nn.AvgPool1d(31) #input should be 512x30 so this outputs a 512x1

        
    def forward(self, x):
      
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.conv2(x)
        x = self.avgPool(x)
        x = x.view(-1, self.n_output)
        
        return x