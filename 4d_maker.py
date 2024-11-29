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


import torch
import os
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_path = '/Users/joppekietselaer/Desktop/coding/python/ACDC_dataChallenge/myDataset/training_joppe_images_lr=1e-4_125_tmax=10_resnet/model_epoch60.pth'

# Initialize model
in_channels = 1
num_classes = 4
model = UNet(in_channels, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

input_dir = '/Users/joppekietselaer/Desktop/coding/python/ACDC_dataChallenge/4d_outputs'

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



# Process each file in the input directory
patient_gif_dict = {}  

for filename in tqdm(sorted(os.listdir(input_dir))):
    if filename.endswith('.npy'):
        patient_id = filename
        file_path = os.path.join(input_dir, filename)
        
        # Load 4D data array and process each frame
        data = np.load(file_path, allow_pickle=True)
        for frame_idx in range(data.shape[3]):  # Loop over frames
            processed_frame = []
            for slice_idx in range(data.shape[2]):  # Loop over slices
                anatomical_image = data[:, :, slice_idx, frame_idx].astype(np.float32)

                # Preprocess anatomical image (pad and crop)
                anatomical_image = pad_and_crop(anatomical_image)

                # Normalize the image
                anatomical_image -= np.mean(anatomical_image)
                anatomical_image /= np.std(anatomical_image)

                # Convert to tensor and add batch/channel dimensions
                anatomical_image_tensor = torch.tensor(anatomical_image).unsqueeze(0).unsqueeze(0).to(device)

                # Pass through model and store output
                with torch.no_grad():
                    output = model(anatomical_image_tensor)
                    predicted_segmentation = torch.argmax(output, dim=1).cpu().numpy()[0]
                    processed_frame.append(predicted_segmentation)
                    
            # Store processed slices in dictionary for each patient and frame
            if patient_id not in patient_gif_dict:
                patient_gif_dict[patient_id] = []
            patient_gif_dict[patient_id].append(np.stack(processed_frame, axis=0))  # Stack slices
        break

for patient_id, images in patient_gif_dict.items():
    # Ensure data is 4D: (num_frames, num_slices, height, width)
    patient_data = np.array(images)
    print(f"Shape of data for {patient_id}: {patient_data.shape}")  # Should confirm (frames, slices, height, width) or desired shape

    save_path = f"/Users/joppekietselaer/Desktop/coding/python/ACDC_dataChallenge/{patient_id}_segmentation_data.npy"
    np.save(save_path, patient_data)
    print(f"Data for {patient_id} saved at {save_path}")
    break

for patient_id, frames in patient_gif_dict.items():
    # Number of slices in each frame and the number of frames (time steps)
    num_slices = len(frames[0])
    num_frames = len(frames)

    fig = go.Figure(
        # Create frames for each time step, adjusting the color of all slices in each frame
        frames=[
            go.Frame(
                data=[
                    go.Surface(
                        z=slice_idx * np.ones_like(frames[time_step][slice_idx]),
                        surfacecolor=frames[time_step][slice_idx],
                        colorscale="Viridis",
                        showscale=(slice_idx == 0),  # Only show color scale on the first slice
                        opacity=0.5
                    )
                    for slice_idx in range(num_slices)
                ],
                name=f"time_{time_step}"
            )
            for time_step in range(num_frames)
        ]
    )

    # Set up the initial 3D stack of slices for the first time step
    fig.add_traces([
        go.Surface(
            z=slice_idx * np.ones_like(frames[0][slice_idx]),
            surfacecolor=frames[0][slice_idx],
            colorscale="Viridis",
            showscale=(slice_idx == 0),  # Show color scale only on the first slice
            opacity=0.5
        )
        for slice_idx in range(num_slices)
    ])

    # Configure layout with play/pause and time slider
    fig.update_layout(
        title=f'4D Visualization for Patient {patient_id}',
        width=700,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90),
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Slice Index"
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 5, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "fromcurrent": True}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )

    # Add a slider for time steps
    fig.update_layout(
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Time Step:",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {"args": [[f"time_{time_step}"], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
                 "label": str(time_step),
                 "method": "animate"}
                for time_step in range(num_frames)
            ]
        }]
    )

    fig.show()

    # Break after one patient for example
    break
