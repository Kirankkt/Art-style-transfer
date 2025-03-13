import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# ------------------------
# TransformerNet Architecture (from fast neural style)
# ------------------------

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return residual + out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=self.upsample, mode='nearest')
        x = self.reflection_pad(x)
        x = self.conv2d(x)
        return x

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Downsampling
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        # Residual blocks
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)

    def forward(self, x):
        y = F.relu(self.in1(self.conv1(x)))
        y = F.relu(self.in2(self.conv2(y)))
        y = F.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = F.relu(self.in4(self.deconv1(y)))
        y = F.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

# ------------------------
# Model Loading with Key-Stripping Fix
# ------------------------

@st.cache_resource
def load_model(style_name):
    """
    Load the TransformerNet state_dict from the given style's .pth file,
    stripping out any running_mean and running_var keys to avoid errors.
    """
    model_path = f"{style_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = TransformerNet()
        state_dict = torch.load(model_path, map_location=device)
        # Remove unwanted running stats keys
        for key in list(state_dict.keys()):
            if "running_mean" in key or "running_var" in key:
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading the model for {style_name}: {e}")
        return None

# ------------------------
# Image Processing Functions
# ------------------------

def preprocess_image(image):
    transform_pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    image_tensor = transform_pipeline(image).unsqueeze(0)
    return image_tensor

def tensor_to_image(tensor):
    tensor = tensor.clone().detach().squeeze(0)
    tensor = tensor.cpu().clamp(0, 255).numpy()
    tensor = tensor.transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(tensor)

def stylize_image(model, input_tensor, device):
    with torch.no_grad():
        output_tensor = model(input_tensor.to(device))
    return output_tensor

# ------------------------
# Main Streamlit App
# ------------------------

def main():
    st.title("AI-Powered Art Style Transformation")
    st.write("Upload your image and choose one of the four styles below to transform it!")
    
    # Available styles (the .pth filenames without extension)
    style_options = ["candy", "mosaic", "rain_princess", "udnie"]
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    selected_style = st.selectbox("Choose an art style", style_options)
    
    if st.button("Transform Image"):
        if uploaded_file is None:
            st.error("Please upload an image first!")
            return
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)
        
        input_tensor = preprocess_image(image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(selected_style)
        if model is None:
            st.error("Model could not be loaded. Please check the model file.")
            return
        
        output_tensor = stylize_image(model, input_tensor, device)
        output_image = tensor_to_image(output_tensor)
        st.image(output_image, caption=f"Transformed with {selected_style} style", use_container_width=True)

if __name__ == '__main__':
    main()
