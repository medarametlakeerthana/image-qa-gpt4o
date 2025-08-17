import streamlit as st
import base64
from PIL import Image
from io import BytesIO
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Image QA", page_icon="🖼️")
st.title("Image Question Answering with GPT-4o 🧠📷")

# Upload image
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)

    image_prompt = st.text_input("Ask a question about this image:")

    if image_prompt:
        # Convert image to base64
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:{image_file.type};base64,{base64_image}"

        # GPT-4o model
        vision_model = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.2,
            max_tokens=1000
        )

        # Send image + text in correct format
        response = vision_model.invoke([{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": image_prompt}
            ]
        }])

        # Extract answer
        st.subheader("Answer")
        st.write(response.content if isinstance(response.content, str) else str(response.content))
