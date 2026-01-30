import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
from ultralytics import YOLO
# import tkinter as tk  <-- REMOVED
# from tkinter import filedialog <-- REMOVED
import plotly.express as px

# ... (Helper functions remain exactly the same) ...

# REMOVED: select_folder() function entirely

# ==========================================
# 3. UI & INITIALIZATION
# ==========================================
st.sidebar.title("⚙️ Configuration")
# Recommendation: Change this path to a relative path if you upload the model to GitHub
model_path = st.sidebar.text_input("Model Path",
                                   r"C:\Users\vinay\PycharmProjects\internship_infosys\pcb_yolo_results\run_high_acc\weights\best.pt")
model = load_yolo_model(model_path)

# Load ALL templates
template_library = load_all_templates(TEMPLATE_DIR)
st.sidebar.success(f"Loaded {len(template_library)} reference templates from database.")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.title("🔍 Universal PCB Diagnostic Hub")
st.markdown("### Auto-Matching Mode Enabled")
st.caption("Upload mixed batches. The system will automatically find the correct reference template for each board.")

if 'selected_folder_path' not in st.session_state:
    st.session_state['selected_folder_path'] = ''

col_input, col_status = st.columns([2, 1])

with col_input:
    input_method = st.radio("Select Input:", ["File Upload", "Local Folder"], horizontal=True)
    files_to_process = []

    if input_method == "File Upload":
        uploaded_files = st.file_uploader("Upload Mixed PCB Batch", type=['png', 'jpg', 'jpeg'],
                                          accept_multiple_files=True)
        if uploaded_files:
            files_to_process = [(f.name, f, 'streamlit_upload') for f in uploaded_files]
    else:
        # REPLACED: Removed the Browse button logic, used text_input to feed path
        path_input = st.text_input("Enter Folder Path:", st.session_state['selected_folder_path'])
        if path_input:
            st.session_state['selected_folder_path'] = path_input
            
        if st.session_state['selected_folder_path'] and os.path.exists(st.session_state['selected_folder_path']):
            local_files = [f for f in os.listdir(st.session_state['selected_folder_path']) if
                           f.lower().endswith(('png', 'jpg', 'jpeg'))]
            files_to_process = [(f, os.path.join(st.session_state['selected_folder_path'], f), 'local_path') for f in
                                local_files]
        elif st.session_state['selected_folder_path']:
            st.error("Folder path not accessible on the server.")

# ... (Rest of your batch processing and plotting logic remains exactly the same) ...
