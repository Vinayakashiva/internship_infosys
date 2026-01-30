# 1. Update your Imports (Remove Tkinter)
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
from ultralytics import YOLO
import plotly.express as px

# 2. Update Configuration (Use relative paths or UI inputs)
# On Streamlit Cloud, D:\ does not exist. 
# It's better to let the user define this or use a folder inside your repo.
TEMPLATE_DIR = "templates" 

if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

# 3. Remove the select_folder function entirely.
# Instead, use a text input for the local folder path.
# Note: Local folder browsing only works if you run this on your OWN machine.
# On the Cloud, users MUST use the "File Upload" method.

# ==========================================
# 4. UPDATED UI LOGIC
# ==========================================
st.title("🔍 Universal PCB Diagnostic Hub")

col_input, col_status = st.columns([2, 1])

with col_input:
    # We remove the "Browse" button and rely on text input or upload
    input_method = st.radio("Select Input:", ["File Upload", "Local Folder Path"], horizontal=True)
    files_to_process = []

    if input_method == "File Upload":
        uploaded_files = st.file_uploader("Upload Mixed PCB Batch", type=['png', 'jpg', 'jpeg'],
                                          accept_multiple_files=True)
        if uploaded_files:
            files_to_process = [(f.name, f, 'streamlit_upload') for f in uploaded_files]
    else:
        # Instead of a popup, we use a text box for the path
        folder_path = st.text_input("Enter Local Folder Path:", placeholder="/mount/src/your_app/data")
        if folder_path and os.path.exists(folder_path):
            local_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            files_to_process = [(f, os.path.join(folder_path, f), 'local_path') for f in local_files]
        elif folder_path:
            st.error("Folder path not found.")
