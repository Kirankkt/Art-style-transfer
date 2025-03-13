import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# ------------------------
# Helper Functions
# ------------------------

@st.cache(allow_output_mutation=True)
def load_model(style_name):
    """
    Load the model for the given style from the current directory.
    The model file should be named exactly as <style_name>.pth.
    Example: If style_name = "candy", then the model file is "candy.pth" in the root folder.
    """
    model_path = f"{style_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load the entire model checkpoint
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading the model for {style_name}: {e}")
        return None

def preprocess_image(image):
    """
    Convert an image to a tensor and apply necessary transformations.
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    image_tensor = transform_pipeline(image).unsqueeze(0)
    return image_tensor

def tensor_to_image(tensor):
    """
    Convert a tensor output by the model into a PIL image.
    """
    tensor = tensor.clone().detach().squeeze(0)
    tensor = tensor.cpu().clamp(0, 255).numpy()
    tensor = tensor.transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(tensor)

def stylize_image(model, input_tensor, device):
    """
    Run the style transfer model on the input tensor.
    """
    with torch.no_grad():
        output_tensor = model(input_tensor.to(device))
    return output_tensor

# ------------------------
# Main Streamlit App
# ------------------------

def main():
    st.title("AI-Powered Art Style Transformation")
    st.write("Upload your drawing or photo and choose one of the four styles below to transform it!")

    # Define available styles (matching your model filenames without .pth)
    style_options = ["candy", "mosaic", "rain_princess", "udnie"]
    
    # File uploader for user image
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    # Style selection
    selected_style = st.selectbox("Choose an art style", style_options)
    
    # Button to apply the style transformation
    if st.button("Transform Image"):
        if uploaded_file is None:
            st.error("Please upload an image first!")
            return
        
        # Display the original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Preprocess the image
        input_tensor = preprocess_image(image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the corresponding style model
        model = load_model(selected_style)
        if model is None:
            st.error("Model could not be loaded. Please check the model file.")
            return
        
        # Apply style transformation
        output_tensor = stylize_image(model, input_tensor, device)
        output_image = tensor_to_image(output_tensor)
        
        # Display the transformed image
        st.image(output_image, caption=f"Transformed with {selected_style} style", use_column_width=True)

if __name__ == '__main__':
    main()
