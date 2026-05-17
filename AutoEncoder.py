import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

from Model_UNet import UNet_Autoencoder

data = './DRIVE'

class Readimage(Dataset):
    def __init__(self, data, subset='training'):
        super().__init__()
        self.data = data
        self.subset = subset
        self.image_dir = os.path.join(self.data, subset, 'images')  
        self.mask_dir = os.path.join(self.data, subset, 'mask')  
        self.manual_dir = os.path.join(self.data, subset, '1st_manual')    
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.tif')]

        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]  
        img_path = os.path.join(self.image_dir, img_name)   

        img = Image.open(img_path)
        img = np.array(img)

        img = img / 255.0
        
        img = np.transpose(img, (2, 0, 1))  # H x W x C -> C x H x W
        img = torch.from_numpy(img).float()
        #print(img.shape)

        file_id = img_name.split('_')[0]
        manual_name = f"{file_id}_manual1.gif"
        mask_name = f"{file_id}_{self.subset}_mask.gif"

        manual_path = os.path.join(self.manual_dir, manual_name)
        manual = Image.open(manual_path).convert('L')  # L: 灰度模式
        manual = np.array(manual)
        manual = manual / 255.0  
        manual = torch.from_numpy(manual).long()

        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')  # L: 灰度模式
        mask = np.array(mask)
        mask = torch.from_numpy(mask).float()  
        mask = mask / 255.0 
        mask = mask.unsqueeze(0)  # 增加通道維度，使其形狀為 (1, H, W)
        
        return img, mask, manual
    
batch_size = 1
epoch = 50

train_dataset = Readimage(data, subset='training')
test_dataset = Readimage(data, subset='test')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# def visualize_data(dataset, indices):
#     plt.figure(figsize=(50, 18 * len(indices)))
    
#     for idx, index in enumerate(indices):
#         img, mask, manual = dataset[index]  
#         img = img.numpy()
#         mask = mask.numpy()
#         manual = manual.numpy()
#         if img.ndim == 3 and img.shape[0] == 3:
#             img = img.transpose(1, 2, 0)
#         img = (img * 255).astype(np.uint8)  

#         plt.subplot(len(indices), 3, idx * 3 + 1)
#         plt.imshow(img)
#         plt.axis('off')
#         plt.text(-1, 0.5, f'Image {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

#         plt.subplot(len(indices), 3, idx * 3 + 2)
#         plt.imshow(mask.squeeze().astype(np.uint8), cmap='gray')
#         plt.axis('off')
#         plt.text(-1, 0.5, f'Mask {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

#         plt.subplot(len(indices), 3, idx * 3 + 3)
#         plt.imshow(manual.squeeze().astype(np.uint8), cmap='gray')
#         plt.axis('off')
#         plt.text(-1, 0.5, f'Manual {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)
#     plt.tight_layout()
#     plt.show()

# visualize_data(train_dataset, indices=[1,2,3,4,5,6,7,8,9,10])

'''模型初始化'''
model = UNet_Autoencoder()  
model = model.cuda() 

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print(device)

loss_function = nn.MSELoss()   #Autoencoder常見的loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

#print(model)


# 訓練過程
def train_model(model, dataloader, loss_function, optimizer, total_epochs):
    model.train()  
    for current_epoch in range(total_epochs):
        running_loss = 0.0
        batch_count = 0
        for images, _, _ in dataloader:
            images = images.cuda()
            #masks = masks.cuda()  
            #manuals = manuals.cuda() 

            optimizer.zero_grad() 
            outputs = model(images)  # 輸出重建圖像

            loss = loss_function(outputs, images)  
            running_loss += loss.item()
            loss.backward() 
            optimizer.step()  

            torch.cuda.empty_cache()  # 清理GPU記憶體
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{current_epoch + 1}/{total_epochs}], Loss: {avg_loss:.5f}')

# 訓練模型
train_model(model, train_dataloader, nn.MSELoss(), optimizer, epoch)

def calculate_psnr(original, reconstructed, max_value=1.0):
    original = original.clamp(0, max_value)
    reconstructed = reconstructed.clamp(0, max_value)
    # 計算 MSE (均方誤差)
    mse = F.mse_loss(reconstructed, original)
    if mse == 0:
        return float('inf')# 確保MSE不等於零避免無窮大
    psnr = 10 * np.log10((max_value ** 2) / mse.item())
    return psnr

psnr_values = []  
def evaluate_psnr(model, dataloader, max_value=1.0): 
    model.eval()
    psnr_list = []
    with torch.no_grad():
        for index, (images, _, _) in enumerate(dataloader):
            images = images.cuda()  
            outputs = model(images)  

            for i in range(images.size(0)): 
                psnr_value = calculate_psnr(images[i], outputs[i], max_value)
                psnr_list.append(psnr_value)
                print(f"PSNR for Image {index * images.size(0) + i + 1}: {psnr_value:.3f} dB")
                psnr_values.append(psnr_value)  
    avg_psnr = np.mean(psnr_list)
    print(f"Average PSNR: {avg_psnr:.3f} dB")

    pd.DataFrame(psnr_values, columns=['PSNR']).to_csv('./hw3_psnr.csv', index=False)
    
evaluate_psnr(model, test_dataloader, max_value=1.0)
####重建視網膜的影像
def visualize_reconstruction(model, dataloader, save_dir='./rebuilt'):
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for index, (images, _, _) in enumerate(dataloader):
            images = images.cuda() 
            outputs = model(images) 
            
            for i in range(images.size(0)):  # 批次大小
                original = images[i].cpu().numpy().transpose(1, 2, 0)
                reconstructed = outputs[i].cpu().numpy().transpose(1, 2, 0)

                original = np.clip(original, 0, 1)
                reconstructed = np.clip(reconstructed, 0, 1)
                combined_image = np.concatenate((original, reconstructed), axis=1)

                plt.figure(figsize=(10, 5))
                plt.imshow(combined_image)
                plt.title(f'Original and Reconstructed Image {index * images.size(0) + i + 1}')
                plt.axis('off')
                plt.show()

                combined_img_path = os.path.join(save_dir, f'pic_{index * images.size(0) + i + 1}.png')
                plt.imsave(combined_img_path, (combined_image * 255).astype(np.uint8))

visualize_reconstruction(model, test_dataloader, save_dir='./rebuilt')

# def train_model(model, dataloader, loss_function, optimizer, total_epochs):
#     model.train()  
#     for current_epoch in range(total_epochs):  
#         running_loss = 0.0
#         batch_count = 0
#         for images, masks, manuals in dataloader:
#             images = images.cuda()
#             masks = masks.cuda()  
#             manuals = manuals.cuda()

#             optimizer.zero_grad()
#             outputs = model(images)

#             loss = loss_function(outputs, images)  
#             running_loss += loss.item()
#             loss.backward()
#             optimizer.step()

#         avg_loss = running_loss / len(dataloader)
#         print(f'Epoch [{current_epoch + 1}/{total_epochs}], Loss: {avg_loss:.5f}')

# train_model(model, train_dataloader, loss_function, optimizer, epoch)


# def evaluate_model(model, dataloader):
#     model.eval()  
#     total_loss = 0.0
#     batch_count = 0
    
#     with torch.no_grad():
#         for index, (images, _, _) in enumerate(dataloader):
#             images = images.cuda()

#             outputs = model(images)
#             loss = nn.MSELoss()(outputs, images)
#             total_loss += loss.item()
#             batch_count += 1
        
#         avg_loss = total_loss / batch_count
#         print(f'\nMean Loss: {avg_loss:.4f}')

# evaluate_model(model, test_dataloader)

# def calculate_psnr(original, reconstructed, max_value=255.0):
#     # 計算MSE
#     mse = F.mse_loss(reconstructed, original)
#     # 計算PSNR
#     psnr = 10 * np.log10((max_value ** 2) / mse.item())
#     return psnr

# # 範例：計算重建影像的PSNR
# # 假設output是模型的輸出影像，target是原始影像
# output = torch.randn(1, 3, 256, 256)  # 模擬重建影像
# target = torch.randn(1, 3, 256, 256)  # 模擬原始影像

# psnr_value = calculate_psnr(target, output)
# print(f"PSNR: {psnr_value:.2f} dB")
