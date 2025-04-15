import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from vit_pytorch.vit_3d import ViT
from src.resnet3d import ResNet, generate_model


class fmriEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        
        self.volume_encoder = ViT3DEncoder(config)
        self.temporal_transformer = TemporalTransformer(config)
        self.projection = ProjectionHead(config)
        self.to(self.device)  # Move entire model to device at once

        # Gradients and activations tracking
        self.gradients = {}
        self.activations = {}
        self.register_hooks() 

    def forward(self, fmri):
        # fmri is tensor of shape (batch_size, 90, 90, 90, 140) = (B, H, W, D, T)
        B, H, W, D, T = fmri.shape
        volumes = fmri.reshape(B * T, H, W, D)
        volumes_encoding = self.volume_encoder(volumes) # [B, 1024]
        volumes_encoding = volumes_encoding.reshape(B, T, -1) # [B, T, 1024]

        fmri_encodings = self.temporal_transformer(volumes_encoding)  # Temporal transformer [B, T, 1024] -> [B, T, 1024]
        fmri_encodings = fmri_encodings.mean(dim=1) # [B, 1024]
        # fmri_encoding = fmri_encodings[:, 0, :] # [B, 1024]   # CLS token approach 

        fmri_predictions = self.projection(fmri_encodings)        # Linear projection [B, 1024] -> [B, 3] 3 for classification
        return fmri_predictions

    def register_hooks(self):
        # Get the last attention layer
        last_attention = self.volume_encoder.vit3d.transformer.layers[-1][0].norm

        def forward_hook(module, input, output):
            self.activations = output.detach().cpu() # [1, 1001, 1024]
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach().cpu() # [1, 1001, 1024]
        
        # Register hooks
        self.forward_handle = last_attention.register_forward_hook(forward_hook)
        self.backward_handle = last_attention.register_backward_hook(backward_hook)

    def get_attention_map(self, x):

        # Forward pass to get target class
        output = self.forward(x)
        class_idx = output.argmax(dim=1)
        # class_idx = torch.tensor([1])
        print(f"Class index: {class_idx}")  
        
        # Create one-hot vector for target class
        one_hot = torch.zeros_like(output)
        one_hot[torch.arange(output.size(0)), class_idx] = 1
        print(f"One-hot vector: {one_hot}") 
        
        # Backward pass to get gradients and activations from hooks
        output.backward(gradient=one_hot, retain_graph=True) 
        gradients = self.gradients      # [1, 1001, 1024] between -8.5e07 and 9.7e07 
        activations = self.activations  # [1, 1001, 1024] between -3 and 3

        # 1. Compute importance weights (global average pooling of gradients)
        # weights = gradients.mean(dim=2, keepdim=True)         # weights are [1, 1001, 1]
        # weights = gradients.abs().mean(dim=2, keepdim=True)
        weights = gradients.mean(dim=2, keepdim=True)[0] 
        # weights = F.relu(gradients).mean(dim=2, keepdim=True) 

        # 2. Weight activations by importance and sum all features
        cam = (weights * activations).sum(dim=2)  # [1, 1001, 1024] -> [1, 1001]
        
        # 3. Remove CLS token and process patches only
        cam = cam[:, 1:]  # [1, 1000]
        
        # 4. Reshape to 3D patch grid (10x10x10)
        cam = cam.reshape(1, 10, 10, 10) 
        
        # 5. Normalize cam
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) # [0, 1]
        
        # 6. Upsample to original size
        cam_3d = F.interpolate(
            cam.unsqueeze(0),  # [10, 10, 10]
            size=(90, 90, 90),
            mode='trilinear',
            align_corners=False
        ).squeeze()
        
        return cam_3d.detach().cpu().numpy(), class_idx
    
    def visualize_slice(self, cam_3d, original_volume, slice_dim=0, slice_idx=None, save_path='./gradcam_visualization.png'):
        """Improved visualization with better error handling"""
        
        # Check if CAM is computed
        if cam_3d is None:
            print("Error: No CAM computed")
            return
        
        # Process original volume
        original = original_volume.squeeze().permute(2, 0, 1)
        original = original.detach().cpu().numpy()
        
        # Verify shapes
        if original.ndim != 3 or cam_3d.ndim != 3:
            print(f"Shape mismatch: original {original.shape}, CAM {cam_3d.shape}")
            return
        
        # Default to middle slice
        if slice_idx is None:
            slice_idx = original.shape[slice_dim] // 2
        slice_idx = max(0, min(slice_idx, original.shape[slice_dim] - 1))
        
        # Select slice
        try:
            if slice_dim == 0:  # Axial
                img = original[slice_idx]
                attn = cam_3d[slice_idx]
            elif slice_dim == 1:  # Coronal
                img = original[:, slice_idx]
                attn = cam_3d[:, slice_idx]
            else:  # Sagittal
                img = original[:, :, slice_idx]
                attn = cam_3d[:, :, slice_idx]
        except IndexError:
            print(f"Slice {slice_idx} out of bounds for dim {slice_dim}")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Overlay
        ax.imshow(img, cmap='gray')
        heatmap = ax.imshow(attn, cmap='jet', alpha=0.4)
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Grad-CAM Attention')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Visualization saved to {save_path}")

        return img, attn
       
class TemporalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8, batch_first=True) # input is [batch, n_volume, 1024]
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6).to(self.device)

    def forward(self, x):
        # x is a tensor of shape (batch_size, n_volume, 1024)
        logits = self.transformer(x)  # output is [batch_size, n_volume, 1024]
        return logits

class ViT3DEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.dropout = config["dropout"]

        self.vit3d = ViT(
            frames=90,
            image_size=90,
            channels=1,
            frame_patch_size=18,
            image_patch_size=18,
            num_classes=1024,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=self.dropout,
            emb_dropout=self.dropout
        ).to(self.device)

    def forward(self, x):
        # x is a tensor of shape (batch_size, 90, 90, 90) = (B, H, W, D) timepoint
        # ViT3D expects (batch_size, channels, frames, height, width)
        timepoint = x.to(self.device)
        timepoint = timepoint.permute(0, 3, 1, 2)  # (B, H, W, D) -> (B, D, H, W)
        timepoint = timepoint.unsqueeze(1)        # (B, 1, D, H, W) for channels

        encoding = self.vit3d(timepoint) # output is [batch, 1024]
        return encoding

class ResnetVideo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.resnet_video = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True).eval().to(self.device)
        self.resnet_blocks = self.resnet_video.blocks[:-1]

    def forward(self, x):

        if len(x.shape) == 4:
            x = x.unsqueeze(1)                  # Add channel dimension ([batch_size, 1, 91, 90, 90])
        
        x = x.permute(0, 1, 4, 2, 3)           # ([batch_size, C, F, H, W])
        x = x.repeat(1, 3, 1, 1, 1)            # Shape (batch_size, 3, 91, 90, 90)

        # Process through each ResNet block manually
        for block in self.resnet_blocks:
            x = block(x)
        
        # Apply adaptive pooling to get a fixed-size output
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.shape[0], -1)  # Flatten to [batch_size, 2048]
        return x

class Resnet3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.resnet = generate_model(50, n_classes=1024)

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)                  # Add channel dimension ([batch_size, 1, 91, 90, 90])

        x = x.repeat(1, 3, 1, 1, 1)

        x = self.resnet(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.dropout = config["dropout"]
        
        # First average across timepoints to get (batch_size, 1024)
        # Then project to 2 classes
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 2)  # 2 classes classification
        ).to(self.device)

        self.projection2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 2)  # 2 classes classification
        ).to(self.device)

        self.projection3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 3)  # 2 classes classification
        ).to(self.device) 

    def forward(self, x):
        # x is a tensor of shape (batch_size, 1024)
        logits = self.projection3(x)  # output is [batch, 2]
        return logits

