import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # identity mapping

    def forward(self, x):
        identity = self.identity(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # add the resiual connection
        return self.relu(out)

 
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = ResidualBlock(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        # Output layer: num_classes channels
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)  # Output logits
        return out#torch.sigmoid(out)  # Return logits without softmax
# Check if MPS is available and set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



train_loc = '/Users/joppekietselaer/Desktop/coding/python/ACDC_dataChallenge/myDataset/training'
val_loc = '/Users/joppekietselaer/Desktop/coding/python/ACDC_dataChallenge/myDataset/validation'
#train_loc = (r'C:\Users\odemirel\Desktop\joppe_tutorial\ACDC\database\training_burak')

class ACDCDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        # Load the image with shape NxMx2
        image = np.load(img_path, allow_pickle=True)  
        # Check if the shape is correct
        if image.ndim != 3 or image.shape[0] != 2:
            raise ValueError("Expected image to have shape NxMx2")

        # Separate the anatomical image and segmentation mask
        anatomical_image = image[0, :, :].astype(np.float32)  # First channel
        #segmentation_mask = image[:, :, 1].astype(np.compat.long)   # Second channel
        segmentation_mask = image[1, :, :].astype(np.int64)

        # Pad zeros to reach 256x256
        def pad_to_target(arr, target_size=(256, 256)):
            height, width = arr.shape
            pad_height = max(0, target_size[0] - height)
            pad_width = max(0, target_size[1] - width)
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            return np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
        
        def crop_center(arr, target_size=(256, 256)):
            height, width = arr.shape
            start_y = height // 2 - target_size[0] // 2
            start_x = width // 2 - target_size[1] // 2
            return arr[start_y:start_y + target_size[0], start_x:start_x + target_size[1]]

        def normalize_img(img):
            img = img.copy().astype(np.float32)
            img -= np.mean(img)
            img /= np.std(img)
            return img
                
        # Pad and crop the images
        anatomical_image = pad_to_target(anatomical_image)
        anatomical_image = crop_center(anatomical_image)
        segmentation_mask = pad_to_target(segmentation_mask)
        segmentation_mask = crop_center(segmentation_mask)
        
        anatomical_image = normalize_img(anatomical_image)

        if self.transform:
            anatomical_image = self.transform(anatomical_image)
            segmentation_mask = self.transform(segmentation_mask)

        return anatomical_image, segmentation_mask


image_dir = train_loc
    
# Transformations for augmentations (if needed)
transform = transforms.Compose([
    transforms.ToTensor()
])

# I already split the dataset
'''
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
'''
# Create datasets for training and validation
train_dataset = ACDCDataset(image_dir=train_loc, transform=transform)
val_dataset = ACDCDataset(image_dir=val_loc, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False)


# Test the loader
for batch_idx, (images, masks) in enumerate(train_loader):
    print(f"Batch {batch_idx+1}, Image shape: {images.shape}, Mask shape: {masks.shape}")
    break

import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Example initialization
in_channels = 1  # Change based on input image channels
num_classes = 4   # RV, LV, myocardium
model = UNet(in_channels, num_classes).to(device)
num_epochs = 125

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation
# Joppe's assigment to learn what is this criteria!
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# Specify the directory to save figures
save_dir = ('/Users/joppekietselaer/Desktop/coding/python/ACDC_dataChallenge/myDataset/training_joppe_images_lr=1e-4_125_tmax=10_resnet')
os.makedirs(save_dir, exist_ok=True)

# Initialize lists to store loss values
train_loss_values = []
validation_loss_values = []


# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, (images, masks_in) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        input_image = images.to(device)  # Anatomical image
        masks = masks_in.to(device)  # Segmentation masks

        #main part, model is running
        outputs = model(input_image)

        loss = criterion(outputs, masks.squeeze(1).long())
        #Joppe's 2nd assignment! Why we use masks.squeeze(1).long())? and what this loss means and how it works?
        #Hint: Criterian is a Cross entropy and in torch, cross entrop is already defined with softmax, use the softmax to understand
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Print the loss at each iteration
        if batch_idx % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Save figures and display every 20th iteration
        #Joppe's 3rd assignment! What is happening here?
        if batch_idx % 50 == 0:
            fig, axs = plt.subplots(1, 7, figsize=(15, 8))

            # Display the input image
            axs[0].imshow(input_image[0, 0].detach().cpu().numpy(), cmap='gray')
            axs[0].set_title('Input Image')
            axs[0].axis('off')

            out = torch.softmax(outputs, dim=1)
            # Display the output images
            for i in range(4):
                axs[i + 1].imshow(out[0,i].detach().cpu().numpy(), cmap='gray')
                axs[i + 1].set_title(f'Output Mask {i + 1}')
                axs[i + 1].axis('off')

            #Joppe's 4th assignment! What does the line below doing?
            out_comb = torch.argmax(torch.softmax(outputs, dim=1), dim=1).unsqueeze(1)
            axs[5].imshow(out_comb[0, 0].detach().cpu().numpy(), cmap='gray')
            axs[5].set_title('Predicted Mask')
            axs[5].axis('off')

            # Display the segmentation mask
            axs[6].imshow(masks[0, 0].detach().cpu().numpy(), cmap='gray')
            axs[6].set_title('Segmentation Mask')
            axs[6].axis('off')

            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'epoch{epoch + 1}_batch{batch_idx + 1}.png'))
            plt.close(fig)

    # now call the dataloader but with validation
    # change your model.train() to model.eval()
    # and run all validation data with the current trained model
    # use the similar saving part to save some images from the validation to a validation_images folder
    
    # Validation Phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (images, masks_in) in enumerate(tqdm(val_loader)):
            input_image = images.to(device)
            masks = masks_in.to(device)

            # Forward pass (no gradient calculation)
            outputs = model(input_image)
            loss = criterion(outputs, masks.squeeze(1).long())
            val_loss += loss.item()

            if batch_idx % 20 == 0:
                fig, axs = plt.subplots(1, 7, figsize=(15, 8))
                axs[0].imshow(input_image[0, 0].detach().cpu().numpy(), cmap='gray')
                axs[0].set_title('Validation Input Image')
                axs[0].axis('off')

                out = torch.softmax(outputs, dim=1)
                for i in range(4):
                    axs[i + 1].imshow(out[0, i].detach().cpu().numpy(), cmap='gray')
                    axs[i + 1].set_title(f'Validation Output Mask {i + 1}')
                    axs[i + 1].axis('off')

                out_comb = torch.argmax(torch.softmax(outputs, dim=1), dim=1).unsqueeze(1)
                axs[5].imshow(out_comb[0, 0].detach().cpu().numpy(), cmap='gray')
                axs[5].set_title('Validation Predicted Mask')
                axs[5].axis('off')

                axs[6].imshow(masks[0, 0].detach().cpu().numpy(), cmap='gray')
                axs[6].set_title('Validation Segmentation Mask')
                axs[6].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'validation_epoch{epoch + 1}_batch{batch_idx + 1}.png'))
                plt.close(fig)

    # Calculate average validation loss epoch
    val_loss /= len(val_loader)
    validation_loss_values.append(val_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

    
    # Calculate average epoch loss
    epoch_loss /= len(train_loader)
    train_loss_values.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}')
    


    
    # Save the model every 20th epoch
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch{epoch + 1}.pth'))

    # Update learning rate scheduler
    scheduler.step()

plt.plot(train_loss_values, label='Training Loss')
plt.plot(validation_loss_values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training/Validation Loss Curve')

# Add legend to distinguish between training validation 
plt.legend()
plt.savefig(os.path.join(save_dir, 'loss.png'))
plt.close()

torch.save(model.state_dict(), os.path.join(save_dir, f'last_model_epoch.pth'))