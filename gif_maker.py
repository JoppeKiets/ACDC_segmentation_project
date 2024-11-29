import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

class DoubleConv(nn.Module):
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
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

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

        self.bottle_neck = DoubleConv(512, 1024)

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





import os
import numpy as np
import torch
from PIL import Image, ImageDraw
import imageio
from tqdm import tqdm

# Assuming your UNet model and other necessary classes are defined above
# Define your UNet model again here or import if it's in another module

# Model and device setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_path = '/Users/joppekietselaer/Desktop/coding/python/ACDC_dataChallenge/myDataset/training_joppe_images_lr=1e-4_125_tmax=10_resnet/model_epoch60.pth'

# Initialize model
in_channels = 1
num_classes = 4
model = UNet(in_channels, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Paths
input_dir = '/Users/joppekietselaer/Desktop/coding/python/ACDC_dataChallenge/myDataset/training'
output_dir = '/Users/joppekietselaer/Desktop/coding/python/ACDC_dataChallenge/gif_outputs'
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Function to pad and crop images to the target size
def pad_and_crop(arr, target_size=(256, 256)):
    height, width = arr.shape
    pad_height = max(0, target_size[0] - height)
    pad_width = max(0, target_size[1] - width)
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    # Pad the array
    padded_arr = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    
    # Crop the array
    start_y = padded_arr.shape[0] // 2 - target_size[0] // 2
    start_x = padded_arr.shape[1] // 2 - target_size[1] // 2
    return padded_arr[start_y:start_y + target_size[0], start_x:start_x + target_size[1]]

# Function to overlay segmentation on anatomical image
def overlay_segmentation(anatomical_img, segmentation):
    colored_segmentation = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    colors = {
        1: (255, 0, 0),  # Class 1 (e.g., RV)
        2: (0, 255, 0),  # Class 2 (e.g., LV)
        3: (0, 0, 255)   # Class 3 (e.g., Myocardium)
    }
    for class_id, color in colors.items():
        colored_segmentation[segmentation == class_id] = color

    overlay = Image.fromarray(anatomical_img).convert("RGB")
    overlay_img = Image.fromarray(colored_segmentation, mode="RGB")
    overlay = Image.blend(overlay, overlay_img, alpha=0.5)
    return overlay

# Process slices for each patient and create GIFs
patient_gif_dict = {}  
for filename in tqdm(sorted(os.listdir(input_dir))):
    if filename.endswith('.npy'):
        patient_id = '_'.join(filename.split('_')[:3])        
        slice_path = os.path.join(input_dir, filename)
        
        # Load slice data
        slice_data = np.load(slice_path, allow_pickle=True)
        anatomical_image = slice_data[0, :, :].astype(np.float32)

        # Preprocess anatomical image (pad and crop)
        anatomical_image = pad_and_crop(anatomical_image)

        # Normalize the image
        anatomical_image -= np.mean(anatomical_image)
        anatomical_image /= np.std(anatomical_image)

        # Convert to tensor and add batch/channel dimensions
        anatomical_image_tensor = torch.tensor(anatomical_image).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(anatomical_image_tensor)
            predicted_segmentation = torch.argmax(output, dim=1).cpu().numpy()[0]

        # Convert anatomical image to 8-bit for overlay
        anatomical_image_8bit = ((anatomical_image - anatomical_image.min()) / 
                                 (anatomical_image.max() - anatomical_image.min()) * 255).astype(np.uint8)

        # Overlay segmentation and save the overlay image
        overlay_img = overlay_segmentation(anatomical_image_8bit, predicted_segmentation)

        # Store the overlay image in the patient's list
        if patient_id not in patient_gif_dict:
            patient_gif_dict[patient_id] = []
        patient_gif_dict[patient_id].append(overlay_img)

# Create GIFs for each patient
for patient_id, images in patient_gif_dict.items():
    gif_filename = f"{patient_id}.gif"
    gif_path = os.path.join(output_dir, gif_filename)

    imageio.mimsave(gif_path, images, duration=0.5)  

print(f"GIFs saved in {output_dir}")
