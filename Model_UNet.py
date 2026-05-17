import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1, bias=False), #3,1,1寬高不變化
            nn.BatchNorm2d(out_channels), #讓訓練更快且不會overfiting，把out_channels當作feature
            nn.ReLU(inplace=True), #不會產生新的
            #nn.Dropout(dropout_prob),######## ((有效地減少過擬合
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)  
  
class UNet_Autoencoder(nn.Module):
    def __init__(self,  in_channels=3, out_channels=3,features=[64,128,256,512]):   ######  out_channels=3
        #RGB input=3  
        super().__init__()
        #UNet左半邊downs的部分
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for feature in features:#3轉64 轉128 轉 256 轉512(UNet左半邊down的部分)
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature 

        #UNet中間bottleneck  512轉1024
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)#(512,1024)

        #UNet右半邊ups的部分        
        for feature in reversed(features):#[512,256,128,64]
            self.ups.append(nn.ConvTranspose2d(feature*2,feature, kernel_size=2, stride=2)) #(1024,512)
            self.ups.append(DoubleConv(feature*2,feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)  # 確保輸出通道數為1 #(64,1,kenel=1)
        
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x) 
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)  
        x = self.bottleneck(x)
        skip_connections.reverse()  # 反轉 skip connections

        for i in range(0, len(self.ups), 2):  # 2個2個處理
            x = self.ups[i](x)  # 上採樣
            skip_connection = skip_connections[i // 2]  
            
            # 確保skip_connection和x的尺寸一樣
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            concat = torch.cat((skip_connection, x), dim=1)  # 拼接
            x = self.ups[i + 1](concat)  # 第二個卷積
        return self.final_conv(x)

