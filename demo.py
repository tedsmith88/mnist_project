import streamlit as st
import torch
import numpy as np
from streamlit_drawable_canvas import st_canvas
from mnist import MNIST_CNN
from PIL import Image

model = MNIST_CNN()
state_dict = torch.load('mnist_cnn.pth')
model.load_state_dict(state_dict)
model.eval()

st.title("MNIST Digit Recognizer")
st.write("Draw a digit (0-9) below")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=6,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True,
)

if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    img_resized = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img_resized) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()

    st.write(f"### Prediction: {prediction}")
    st.bar_chart(probabilities.squeeze().numpy())