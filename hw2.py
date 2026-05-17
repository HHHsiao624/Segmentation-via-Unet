import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from UNet_model import UNet

data = './DRIVE'
#test_data = './DIRVE_TestingSet'

class Readimage(Dataset):
    def __init__(self, data, subset='training'):
        super().__init__()
        self.data = data
        self.subset = subset
        self.image_dir = os.path.join(self.data, subset, 'images')  
        self.mask_dir = os.path.join(self.data, subset, 'mask')  
        self.manual_dir = os.path.join(self.data, subset, '1st_manual')    
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.tif')]
        #print(f'Loaded {len(self.image_files)} images from {self.image_dir}')

        # 數據增強
        self.transform = transforms.Compose([
            #transforms.RandomRotation(10),      # 旋轉圖像
            #transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 改變亮度與對比度
            #transforms.RandomHorizontalFlip(p=0.5),  # 50% 機率翻轉圖像
            #transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1), 
            transforms.ColorJitter(brightness=0.2, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),               # 轉換為 Tensor
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]  
        img_path = os.path.join(self.image_dir, img_name)   

        img = Image.open(img_path)
        img = np.array(img)

        img = img / 255.0
        
        # 將圖像從 H x W x C 轉為 C x H x W
        img = np.transpose(img, (2, 0, 1))  # img形狀 (C, H, W)
        img = torch.from_numpy(img).float()
        #print(f'Image range: Min: {img.min()}, Max: {img.max()}')

        file_id = img_name.split('_')[0]
        
        manual_name = f"{file_id}_manual1.gif"
        mask_name = f"{file_id}_{self.subset}_mask.gif"

        manual_path = os.path.join(self.manual_dir, manual_name)
        manual = Image.open(manual_path).convert('L')  # L: 灰度模式
        manual = np.array(manual)
        manual = manual / 255.0  
        manual = torch.from_numpy(manual).long()

        mask_name = f"{file_id}_{self.subset}_mask.gif"
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')  # L: 灰度模式

        mask = np.array(mask)
        mask = torch.from_numpy(mask).float()  
        mask = mask / 255.0 
        mask = mask.unsqueeze(0)  # 添加通道維度，使其形狀為 (1, H, W)
        #print(f'Mask range: Min: {mask.min()}, Max: {mask.max()}')
        return img, mask, manual
    
#dataset = Readimage(data)
#dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

batch_size = 1
epoch = 150

train_dataset = Readimage(data, subset='training')
test_dataset = Readimage(data, subset='test')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def visualize_data(dataset, indices):
    plt.figure(figsize=(50, 18 * len(indices)))
    
    for idx, index in enumerate(indices):
        img, mask, manual = dataset[index]  
        img = img.numpy()
        mask = mask.numpy()
        manual = manual.numpy()
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)  

        plt.subplot(len(indices), 3, idx * 3 + 1)
        #plt.title(f'Image {index}')
        plt.imshow(img)
        plt.axis('off')
        plt.text(-1, 0.5, f'Image {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

        plt.subplot(len(indices), 3, idx * 3 + 2)
        #plt.title(f'Mask {index}')
        plt.imshow(mask.squeeze().astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.text(-1, 0.5, f'Mask {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

        plt.subplot(len(indices), 3, idx * 3 + 3)
        #plt.title(f'Manual {index}')
        plt.imshow(manual.squeeze().astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.text(-1, 0.5, f'Manual {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()

visualize_data(train_dataset, indices=[1,2,3,4,5,6,7,8,9,10])

'''模型'''
model = UNet()
model = model.cuda() 
#output = model(input)
#print("Output shape:", output.shape)

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print(device)

loss_function = nn.BCEWithLogitsLoss()
#loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)#,weight_decay=1e-4)


def calculate_iou_and_f1(pred, Mask):
    pred = pred.squeeze().cpu().numpy()
    Mask = Mask.squeeze().cpu().numpy()
    pred = (pred > 0.5).astype(np.float32)
    Mask = Mask.astype(np.float32)
    intersection = np.sum(pred * Mask)
    union = np.sum(pred) + np.sum(Mask) - intersection
    # 避免除零
    iou = intersection / union if union != 0 else 0
    tp = intersection
    fp = np.sum(pred) - tp
    fn = np.sum(Mask) - tp
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
    return iou, f1

def train_model(model, dataloader, loss_function, optimizer, total_epochs):
    model.train()  
    for current_epoch in range(total_epochs):  
        running_loss = 0.0
        total_iou, total_f1 = 0.0, 0.0
        batch_count = 0
        for images, masks, manuals in dataloader:
            images = images.cuda()
            masks = masks.cuda()  
            manuals = manuals.cuda()
            manuals = manuals.unsqueeze(1)
            # 清空梯度
            optimizer.zero_grad()
            # 前向傳播
            outputs = model(images)

            loss = loss_function(outputs, manuals.float())
            running_loss += loss.item()
            # 反向傳播
            loss.backward()
            optimizer.step()

            predictions = torch.sigmoid(outputs)
            predictions = (predictions > 0.5).float()
            
            iou, f1 = calculate_iou_and_f1(predictions, manuals)
            total_iou += iou
            total_f1 += f1
            batch_count += 1

        avg_loss = running_loss / len(dataloader)
        avg_iou = total_iou / batch_count
        avg_f1 = total_f1 / batch_count

        print(f'Epoch [{current_epoch + 1}/{total_epochs}], Loss: {avg_loss:.5f}, IoU: {avg_iou:.5f}, f1: {avg_f1:.5f}')

train_model(model, train_dataloader, loss_function, optimizer, epoch)


def evaluate_model(model, dataloader):
    model.eval()  
    total_iou, total_f1 = 0.0, 0.0
    batch_count = 0
    results = []  
    
    with torch.no_grad():
        for index, (images, masks, manuals) in enumerate(dataloader):
            images = images.cuda()
            masks = masks.cuda()
            manuals =manuals.cuda()

            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            predictions = (predictions > 0.5).float()
            
            iou, f1 = calculate_iou_and_f1(predictions, manuals)  
            total_iou += iou
            total_f1 += f1
            batch_count += 1

            results.append({'Image Index': index + 1, 'IoU': iou, 'F1': f1})
            print(f'Image {index + 1}: IoU = {iou:.4f}, F1 = {f1:.4f}')
        
        # MIOU&平均F1
        miou = total_iou / batch_count
        avg_f1 = total_f1 / batch_count
        print(f'\nMean IoU (MIOU): {miou:.4f}')
        print(f'Average F1: {avg_f1:.4f}')
        
        results.append({'Image Index': 'Mean', 'IoU': miou, 'F1': avg_f1})
        
        results_df = pd.DataFrame(results)
        print("\nResults Table:")
        print(results_df)
        results_df.to_csv("./results/test6.csv", index=False)
    
    return miou, avg_f1, results_df

evaluate_model(model, test_dataloader)


def visualize_predictions(model, test_dataset, indices=None, save_dir='./predictions/test6'):
    model.eval()
    if indices is None:
        indices = range(len(test_dataset))
    plt.figure(figsize=(50, 18 * len(indices)))
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, index in enumerate(indices):
            img, _, manual = test_dataset[index]  
            img = img.unsqueeze(0).cuda()
            
            outputs = model(img)
            predictions = torch.sigmoid(outputs)
            predictions = (predictions > 0.5).float()  
            
            img = img.squeeze().cpu().numpy()  
            manual = manual.numpy()
            pred = predictions.squeeze().cpu().numpy() 
            
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            
            combined_width = img.shape[1] * 3 + 20 * 2  # 空白
            combined_image = np.zeros((img.shape[0], combined_width, img.shape[2]), dtype=np.uint8)  
            combined_image[:, :img.shape[1], :] = img  
            combined_image[:, img.shape[1] + 20:img.shape[1] * 2 + 20, :] = np.stack([pred.astype(np.uint8) * 255] * 3, axis=-1)  
            combined_image[:, img.shape[1] * 2 + 40:, :] = np.stack([manual.squeeze().astype(np.uint8) * 255] * 3, axis=-1)  

            output_path = os.path.join(save_dir, f'test6_{index+1}.png')
            plt.imsave(output_path, combined_image)

            # 原始的眼球圖像
            plt.subplot(len(indices), 3, idx * 3 + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.text(-1, 0.5, f'Original Eye {index+1}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

            # 預測的血管
            plt.subplot(len(indices), 3, idx * 3 + 2)
            plt.imshow(pred.squeeze().astype(np.uint8), cmap='gray')
            plt.axis('off')
            plt.text(-1, 0.5, f'Predicted Blood Vessels {index+1}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

            # 血管分割的正解
            plt.subplot(len(indices), 3, idx * 3 + 3)
            plt.imshow(manual.squeeze().astype(np.uint8), cmap='gray')
            plt.axis('off')
            plt.text(-1, 0.5, f'Manual Blood Vessels {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()

visualize_predictions(model, test_dataset)
'''
#原版
def visualize_predictions(model, test_dataset, indices=None, save_dir='./predictions/test2'):
    model.eval()
    if indices is None:
        indices = range(len(test_dataset))
    plt.figure(figsize=(50, 18 * len(indices)))
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, index in enumerate(indices):
            img, mask, manual = test_dataset[index]  # 取原始圖像、遮罩和 manual 圖
            img = img.unsqueeze(0).cuda()  # 增加批次維度並移到GPU
            
            outputs = model(img)  # 獲取模型的輸出
            predictions = torch.sigmoid(outputs)
            predictions = (predictions > 0.5).float()  # 根據閾值進行二值化
            
            # 將圖像轉換回原始範圍 [0, 255]
            img = img.squeeze().cpu().numpy()  # 移除多餘的維度
            mask = mask.numpy()
            manual = manual.numpy()
            pred = predictions.squeeze().cpu().numpy()  # 移除多餘的維度

            if img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)

            # 創建合併圖像（每張圖像之間留出一些空白）
            combined_width = img.shape[1] * 3 + 20 * 2  # 兩側的空白
            combined_image = np.zeros((img.shape[0], combined_width, img.shape[2]), dtype=np.uint8)  # 使用計算的寬度
            combined_image[:, :img.shape[1], :] = img  # 原始圖像
            combined_image[:, img.shape[1] + 20:img.shape[1] * 2 + 20, :] = np.stack([mask.squeeze().astype(np.uint8) * 255] * 3, axis=-1)  # 遮罩
            combined_image[:, img.shape[1] * 2 + 40:, :] = np.stack([pred.squeeze().astype(np.uint8) * 255] * 3, axis=-1)  # 預測結果

            # 保存合併圖像
            output_path = os.path.join(save_dir, f'test1_{index}.png')
            plt.imsave(output_path, combined_image)

            # 可視化原始圖像
            plt.subplot(len(indices), 3, idx * 3 + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.text(-1, 0.5, f'Image {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

            # 可視化遮罩
            plt.subplot(len(indices), 3, idx * 3 + 2)
            plt.imshow(mask.squeeze().astype(np.uint8), cmap='gray')
            plt.axis('off')
            plt.text(-1, 0.5, f'Mask {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

            # 可視化預測結果
            plt.subplot(len(indices), 3, idx * 3 + 3)
            plt.imshow(pred.squeeze().astype(np.uint8), cmap='gray')
            plt.axis('off')
            plt.text(-1, 0.5, f'Prediction {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show() 

# 使用模型預測整個測試資料集的結果並保存合併圖像
visualize_predictions(model, test_dataset)
'''

'''
# 更新的預測和可視化函數
def visualize_predictions(model, test_dataset, indices):
    model.eval()  # 設置模型為評估模式
    plt.figure(figsize=(50, 18 * len(indices)))  # 根據需要的行數調整圖形大小
    
    with torch.no_grad():
        for idx, index in enumerate(indices):
            img, mask, manual = test_dataset[index]  # 取原始圖像、遮罩和 manual 圖
            img = img.unsqueeze(0).cuda()  # 增加批次維度並移到GPU
            
            outputs = model(img)  # 獲取模型的輸出
            predictions = torch.sigmoid(outputs)
            predictions = (predictions > 0.5).float()  # 根據閾值進行二值化
            
            # 將圖像轉換回原始範圍 [0, 255]
            img = img.squeeze().cpu().numpy()  # 移除多餘的維度
            mask = mask.numpy()
            manual = manual.numpy()
            pred = predictions.squeeze().cpu().numpy()  # 移除多餘的維度

            if img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)

            # 可視化原始圖像
            plt.subplot(len(indices), 3, idx * 3 + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.text(-1, 0.5, f'Image {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

            # 可視化遮罩
            plt.subplot(len(indices), 3, idx * 3 + 2)
            plt.imshow(mask.squeeze().astype(np.uint8), cmap='gray')
            plt.axis('off')
            plt.text(-1, 0.5, f'Mask {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

            # 可視化預測結果
            plt.subplot(len(indices), 3, idx * 3 + 3)
            plt.imshow(pred.squeeze().astype(np.uint8), cmap='gray')
            plt.axis('off')
            plt.text(-1, 0.5, f'Prediction {index}', fontsize=12, ha='left', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

# 使用模型預測測試資料集的結果並保存合併圖像
visualize_predictions(model, test_dataset,indices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
'''


#print(torch.__version__)
#print(torch.cuda.is_available())
''' 



'''
'''
def train(model, epoch, train_dataloader, optimizer, print_every=30):
  for epoch in range(epoch):
    running_loss = 0.0
    for count, (x, y, manual, width, height) in enumerate(train_dataloader):
      model.train()
      x = x.to(device)
      y = y.to(device)
      # 檢查 y 的類別範圍
      print("Unique values in y:", y.unique())
      print("Min value in y:", y.min().item(), "Max value in y:", y.max().item())

      # 確保 y 的數據類型為 Long
      y = y.long()
      #y = y.to(device).unsqueeze(1).float()  # 添加這一行，將目標的形狀變為 (4, 1, 584, 565)
      #print("Unique values in y:", y.unique())
      #y = (y > 0).float()  # 確保 y 的值在 [0, 1] 之間
      out = model(x)
      loss = loss_function(out, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      running_loss += loss.item()
      
      if count % print_every == 0:
          print(f"Epoch[{epoch + 1}]: Loss: {running_loss / (count + 1):.4f}")
          eval(model, test_dataloader, epoch)

def eval(model, test_dataloader, epoch):
    model.eval()
    num_correct = 0
    num_pixels = 0
    with torch.no_grad():
        for x, y, manual, width, height in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            out_img = model(x)
            probability = torch.sigmoid(out_img)
            predictions = (probability > 0.5).float() 

            num_correct += (predictions == y).sum()

            #batch_pixels = int(width.sum() * height.sum())  
            #num_pixels += batch_pixels 
            batch_size = x.shape[0]  # 獲取當前 batch 的大小
            pixels_per_image = x.shape[2] * x.shape[3]  # 獲取每張圖像的總像素數
            batch_pixels = batch_size * pixels_per_image  # 當前 batch 總像素數
            num_pixels += batch_pixels


    if num_pixels > 0:
        accuracy = num_correct / num_pixels
    else:
        accuracy = 0.0

    print(f'Epoch[{epoch + 1}]: Acc: {accuracy.item()}')


def create_model(out_channels):
    model = UNet(in_channels=3, out_channels=out_channels)#, base_c=32)
    return model

def train_one_epoch(model, optimizer, train_loader, device, epoch, loss_function):
    model.train()
    running_loss = 0.0
    for count, (x, y, manual) in enumerate(train_loader):
      x = x.to(device)
      y = y.to(device).long()  # 將 y 轉換為 Long 型
      #y = filter_labels(y)

      optimizer.zero_grad()
      out = model(x)
      loss = loss_function(out, y)
      loss.backward()
      optimizer.step()
        
    running_loss += loss.item()
    
    if count % 10 == 0:
        print(f"Epoch[{epoch + 1}]: Step[{count}]: Loss: {running_loss / (count + 1):.4f}")

    return running_loss / len(train_loader)


def eval(model, val_loader, device):
    model.eval()
    num_correct = 0
    num_pixels = 0
    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(device), y.to(device)

            out_img = model(x)
            predictions = torch.argmax(out_img, dim=1)

            num_correct += (predictions == y).sum().item()
            num_pixels += y.numel()

    accuracy = num_correct / num_pixels if num_pixels > 0 else 0.0
    print(f'Accuracy: {accuracy:.4f}')
    torch.cuda.empty_cache()

def main(data_path, out_channels, batch_size, epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = Readimage(data_path, subset='training')
    val_dataset = Readimage(data_path, subset='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = create_model(out_channels).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, loss_function)
        print(f"Epoch[{epoch + 1}] Training Loss: {train_loss:.4f}")
        eval(model, val_loader, device)   
 
if __name__ == '__main__':
    data_path = './DRIVE'
    out_channels = 2  # Assuming binary classification (1 class + background)
    batch_size = 1
    epochs = 50
    lr = 0.0001

    main(data_path, out_channels, batch_size, epochs, lr)
'''