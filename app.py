import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("mobilenet_banh_model_best.keras")

# Danh sách nhãn tương ứng với output
class_names = ['Bánh bông lan', 'Bánh cupcake', 'Bánh mì', 'Bánh tart']  # thứ tự phải đúng

# Hàm xử lý ảnh
def preprocess_image(img):
    img = img.resize((150, 150))  # kích thước đúng như khi train
    img = np.array(img) / 255.0  # chuẩn hóa
    return np.expand_dims(img, axis=0)

# Giao diện
st.title("🍰 Nhận Diện 4 Loại Bánh")

uploaded_file = st.file_uploader("Tải ảnh bánh lên", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh bạn vừa chọn", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("🔍 Kết quả dự đoán:")
    st.write(f"👉 Đây là **{predicted_class}**")
