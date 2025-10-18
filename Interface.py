import streamlit as st
import time
from PIL import Image
import io
from chatbot import App
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="MediVision",
    layout="wide"
)

st.title("Medi-Vision")

if "app" not in st.session_state:
    st.session_state.app = App()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Upload an image and ask a question."} # have to change
    ]

# Initialize the uploader_key for the "key-changing trick"
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- Sidebar for Image Upload ---
with st.sidebar:
    st.header("Upload an Image")
    
    # Use the dynamic key from session state
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["png", "jpg", "jpeg"],
        key=st.session_state.uploader_key  # This is the trick!
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Image to be analyzed")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the message has an image, display it
        if "image" in message:
            st.image(message["image"], width=200)

# --- Chat Input and Logic ---
if prompt := st.chat_input("What would you like to know?"):
    
    # --- Handle User Input ---
    user_message = {"role": "user", "content": prompt}
    image_to_process = None
    display_image = None
    
    # We can check the 'uploaded_file' variable defined in the sidebar
    if uploaded_file is not None:
        # Read the image bytes
        image_bytes = uploaded_file.getvalue()
        
        # This is the image that will be "processed" by the bot
        image_to_process = Image.open(io.BytesIO(image_bytes))
        
        base_filename = "img"
        
        # --- OPTION A: Save as PNG (Recommended) ---
        save_format = "PNG"
        save_extension = ".png"
        save_path = os.path.join(base_filename + save_extension)
        # Save the Pillow image object in the new format
        image_to_process.save(save_path, format=save_format)

        # This is the image that will be displayed in the chat
        display_image = image_bytes
        
        # Add the image to the user_message dictionary
        user_message["image"] = display_image

    # Add the complete user message to the chat history
    st.session_state.messages.append(user_message)
    
    # Display the user's message in the chat
    with st.chat_message("user"):
        st.markdown(prompt)
        if display_image:
            st.image(display_image, width=200)
            
    # --- Generate Bot Response (Mock) ---
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):            
            response_content = ""
            
            if image_to_process:
                # This is where you would call your multimodal AI model
                response_content = st.session_state.app.invoke_app_for_img(prompt, st.session_state.app.predict("img.png"))
            else:
                # This is where you would call your text-only AI model
                response_content = st.session_state.app.invoke_app(prompt)

            st.markdown(response_content)
            
            # Add the bot's response to the chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response_content}
            )

    # If a file was just processed, increment the key and rerun.
    if uploaded_file is not None:
        st.session_state.uploader_key += 1
        st.rerun()