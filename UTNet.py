import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class ResDoubleConv(nn.Module):
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

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

        # Initialize with a placeholder size; actual sizes are updated dynamically
        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * 1 - 1) * (2 * 1 - 1), num_heads) * 0.02
        )

        self.relative_position_index = None
        self.H, self.W = None, None  # Track last used height and width

    def update_bias(self, H, W):
        # Update relative position index and bias table only if H or W changes
        if self.H != H or self.W != W:
            self.H, self.W = H, W
            coords_h = torch.arange(H)
            coords_w = torch.arange(W)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, H, W
            coords_flatten = torch.flatten(coords, 1)  # 2, HW

            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += H - 1
            relative_coords[:, :, 1] += W - 1
            relative_coords[:, :, 0] *= 2 * H - 1
            self.relative_position_index = relative_coords.sum(-1)  # HW, HW

            # Dynamically adjust bias table size if required
            self.relative_position_bias_table.data = torch.randn(
                (2 * H - 1) * (2 * W - 1), self.num_heads
            ).to(self.relative_position_bias_table.device) * 0.02

    def forward(self, H, W):
        # Dynamically update the bias if dimensions change
        self.update_bias(H, W)
        # Debugging
        print(f"relative_position_bias_table shape: {self.relative_position_bias_table.shape}")
        print(f"H: {H}, W: {W}")
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(H, W, H * W, -1)  # H, W, HW, nH

        # Expand biases for larger grid sizes
        relative_position_bias_expand_h = torch.repeat_interleave(
            relative_position_bias, H // self.H, dim=0
        )
        relative_position_bias_expanded = torch.repeat_interleave(
            relative_position_bias_expand_h, W // self.W, dim=1
        )  # HW, HW, nH

        relative_position_bias_expanded = (
            relative_position_bias_expanded.view(H * W, H * W, self.num_heads)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
        )

        return relative_position_bias_expanded

# Relative Position Embedding
class RelativePositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.rel_height = None
        self.rel_width = None

    def update_embeddings(self, H, W):
        self.H = H
        self.W = W
        self.rel_height = nn.Parameter(torch.randn(2 * H - 1, self.dim) * 0.02)
        self.rel_width = nn.Parameter(torch.randn(2 * W - 1, self.dim) * 0.02)

    def forward(self, H, W):
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)

        rel_coords_h = coords_h[:, None] - coords_h[None, :]
        rel_coords_w = coords_w[:, None] - coords_w[None, :]

        rel_coords_h += self.H - 1
        rel_coords_w += self.W - 1

        rel_emb_h = self.rel_height[rel_coords_h]
        rel_emb_w = self.rel_width[rel_coords_w]

        bias = rel_emb_h[:, :, None, :] + rel_emb_w[None, :, :, :]
        return bias.permute(2, 0, 1, 3).contiguous().view(self.dim, H * W, H * W).unsqueeze(0)

    def relative_logits_1d(self, q, rel_k, case):
        B, Nh, H, W, dim = q.shape

        rel_logits = torch.einsum("bhxyd,md->bhxym", q, rel_k)  # B, Nh, H, W, 2*shape-1

        if W != self.shape:
            relative_index = torch.repeat_interleave(self.relative_position_index, W // self.shape, dim=0)  # W, shape
        relative_index = relative_index.view(1, 1, 1, W, self.shape)
        relative_index = relative_index.repeat(B, Nh, H, 1, 1)

        rel_logits = torch.gather(rel_logits, 4, relative_index)  # B, Nh, H, W, shape
        rel_logits = rel_logits.unsqueeze(3)
        rel_logits = rel_logits.repeat(1, 1, 1, self.shape, 1, 1)

        if case == "w":
            rel_logits = rearrange(rel_logits, "b heads H h W w -> b heads (H W) (h w)")

        elif case == "h":
            rel_logits = rearrange(rel_logits, "b heads W w H h -> b heads (H W) (h w)")

        return rel_logits



class DownSample_Trans(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, embed_dim, reduce_size):
        super().__init__()
        self.conv = ResDoubleConv(in_channels, out_channels)
        
        # Is this correct?  
        self.transform = TransEncoder(
            in_channels=out_channels,  
            embed_dim=embed_dim, 
            num_heads=num_heads,
            reduce_size=reduce_size
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)   # Apply convolution
        down = self.transform(down)  # Apply transformer
        p = self.pool(down)  # Downsample via pooling
        return down, p

class UpSample_Trans(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, embed_dim, reduce_size):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResDoubleConv(in_channels, out_channels)
        
        # Pass out_channels to match TransDecoder's arguments
        self.transform = TransDecoder(
            embed_dim=embed_dim, 
            out_channels=out_channels, 
            num_heads=num_heads,
            reduce_size=reduce_size
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)  # Upsample x1
        x = torch.cat([x1, x2], dim=1)  # Concatenate with skip connection
        x = self.conv(x)  # Apply convolution
        x = self.transform(x, x2)  # Apply transformer, using x2 as encoder output
        return x
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ResDoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

# Encoder with Positional Encoding
class TransEncoder(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, reduce_size):
        super().__init__()
        
        # 1x1 Convs for Query, Key, and Value
        self.query_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

        # Relative Positional Bias
        self.relative_position_bias =  RelativePositionBias(num_heads)

        # Normalization and Feed-Forward
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):
        # Input shape: B, C, H, W
        B, C, H, W = x.shape
        
        # Generate Queries, Keys, Values
        Q = self.query_conv(x).view(B, -1, H * W).permute(1, 0, 2)  # (HW, B, C)
        K = self.key_conv(x).view(B, -1, H * W).permute(1, 0, 2)
        V = self.value_conv(x).view(B, -1, H * W).permute(1, 0, 2)

        # Add Relative Positional Bias
        rel_bias = self.relative_position_bias(H, W)  # Shape: (1, num_heads, HW, HW)

        # Compute Scaled Dot-Product Attention with Bias
        QK = torch.matmul(Q.transpose(0, 1), K.transpose(0, 1).transpose(-2, -1))  # Shape: (B, num_heads, HW, HW)
        QK = QK / (C ** 0.5)  # Scale by sqrt(C)
        QK += rel_bias.squeeze(0)  # Add relative positional bias
        attn_weights = torch.softmax(QK, dim=-1)  # Apply softmax

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V.transpose(0, 1))  # Shape: (B, HW, C)
        attn_output = attn_output.permute(1, 0, 2).view(B, C, H, W)  # Reshape back

        # Normalization + Residual
        out = self.norm1((attn_output + x).view(B, -1)).view(B, C, H, W)

        # Feed-Forward + Residual
        out = self.norm2(self.ffn(out.view(B, -1)).view(B, C, H, W) + out)

        return out

class TransDecoder(nn.Module):
    def __init__(self, embed_dim, out_channels, num_heads, reduce_size):
        super().__init__()

        # 1x1 Convs for Query, Key, and Value
        self.query_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # what is a Sub-Sample?
        self.key_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # Attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

        # Relative Positional Bias
        self.relative_position_bias = RelativePositionBias(num_heads)

        # Normalization and Feed-Forward
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Output Convolution
        self.output_conv = nn.Conv2d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x, enc_out):
        # Input shape: B, C, H, W
        B, C, H, W = x.shape

        # Generate Queries, Keys, Values
        Q = self.query_conv(x).view(B, -1, H * W).permute(2, 0, 1)  # (HW, B, C)
        K = self.key_conv(enc_out).view(B, -1, H * W).permute(2, 0, 1)
        V = self.value_conv(enc_out).view(B, -1, H * W).permute(2, 0, 1)

        # Add Relative Positional Bias
        rel_bias = self.relative_position_bias(H, W)

        # Multi-Head Attention
        attn_output, _ = self.attn(Q, K, V, attn_mask=None, key_padding_mask=None, attn_bias=rel_bias)
        attn_output = attn_output.permute(1, 2, 0).view(B, -1, H, W)

        # Normalization + Residual
        out = self.norm1((attn_output + x).view(B, -1)).view(B, C, H, W)

        # Feed-Forward + Residual
        out = self.norm2(self.ffn(out.view(B, -1)).view(B, C, H, W) + out)

        # Final Convolution to Reduce Channels
        out = self.output_conv(out)

        return out
class UTNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_heads, embed_dim, reduce_size):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        
        # Halve height and width after each downsampling
        self.down_convolution_2 = DownSample_Trans(64, 128, num_heads=num_heads, embed_dim=embed_dim, reduce_size=reduce_size) 
        self.down_convolution_3 = DownSample_Trans(128, 256, num_heads=num_heads, embed_dim=embed_dim, reduce_size=reduce_size)
        self.down_convolution_4 = DownSample_Trans(256, 512, num_heads=num_heads, embed_dim=embed_dim, reduce_size=reduce_size)

        self.bottle_neck = ResDoubleConv(512, 1024)

        # Pass dimensions for upsampling
        self.up_convolution_1 = UpSample_Trans(1024, 512, num_heads=num_heads, embed_dim=embed_dim, reduce_size=reduce_size)
        self.up_convolution_2 = UpSample_Trans(512, 256, num_heads=num_heads, embed_dim=embed_dim, reduce_size=reduce_size)
        self.up_convolution_3 = UpSample_Trans(256, 128, num_heads=num_heads, embed_dim=embed_dim, reduce_size=reduce_size)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

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
        return out  # Return logits without softmax

# Check if MPS is available and set the device
device = torch.device("cpu") #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)



train_loader_iter = iter(train_loader)

# Get the first batch
images,masks = next(train_loader_iter)


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
num_heads = 4
embed_dim = 16
reduce_size = 16 # I don't know what this does?
model = UTNet(in_channels, num_classes, num_heads, embed_dim, reduce_size).to(device)
num_epochs = 75

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class segmentation
# Joppe's assigment to learn what is this criteria!
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# Specify the directory to save figures
save_dir = ('/Users/joppekietselaer/Desktop/coding/python/ACDC_dataChallenge/myDataset/training_joppe_images')
os.makedirs(save_dir, exist_ok=True)

# Initialize lists to store loss values
train_loss_values = []
validation_loss_values = []

def dice_loss(predicted_mask, segmentation_mask, epsilon=1e-6):
    """
    Calculate Dice loss between predicted and segmentation masks.
    Args:
        predicted_mask (tensor): Predicted binary mask.
        segmentation_mask (tensor): Ground truth binary mask.
        epsilon (float): Small constant to avoid division by zero.
    Returns:
        float: Dice loss value.
    """
    intersection = torch.sum(predicted_mask * segmentation_mask)
    predicted_sum = torch.sum(predicted_mask)
    segmentation_sum = torch.sum(segmentation_mask)

    # Compute Dice coefficient and Dice loss
    dice_coeff = (2 * intersection + epsilon) / (predicted_sum + segmentation_sum + epsilon)
    return 1 - dice_coeff



# Initialize lists to store loss values
train_loss_values = []
dice_loss_values = []

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
        if batch_idx % 5 == 0:
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

# Plot the loss curve
plt.plot(train_loss_values, validation_loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training/validation Loss Curve')
plt.savefig(os.path.join(save_dir, 'loss.png'))
plt.close()

torch.save(model.state_dict(), os.path.join(save_dir, f'last_model_epoch.pth'))