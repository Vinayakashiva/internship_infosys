import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
import io
from ultralytics import YOLO
from PIL import Image

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
st.set_page_config(page_title="Batch PCB Diagnostic System", layout="wide")

# Path to your local template folder
TEMPLATE_DIR = "D:\PCB_DATASET\PCB_USED"

if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)


# ==========================================
# 2. HELPER FUNCTIONS (Your Original Logic)
# ==========================================

@st.cache_resource
def load_yolo_model(path):
    try:
        return YOLO(path)
    except:
        return None


def analyze_pcb(template_img, test_img, model, conf_threshold=0.5):
    """
    STRICT PRESERVATION OF YOUR LOGIC:
    Module 1: CV Subtraction (Localizes differences)
    Module 2: YOLO Classification (Identifies defect types)
    """
    # --- Module 1: Localization ---
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Dimensional Alignment
    if gray_template.shape != gray_test.shape:
        gray_test = cv2.resize(gray_test, (gray_template.shape[1], gray_template.shape[0]))
        test_img = cv2.resize(test_img, (template_img.shape[1], template_img.shape[0]))

    # Compute Difference & Threshold
    diff = cv2.absdiff(gray_template, gray_test)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological Noise Removal
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_output = test_img.copy()
    detections = []

    # --- Module 2: Classification ---
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50: continue

        x, y, w, h = cv2.boundingRect(cnt)
        pad = 10
        h_img, w_img = test_img.shape[:2]
        y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
        x1, x2 = max(0, x - pad), min(w_img, x + w + pad)
        roi = test_img[y1:y2, x1:x2]

        if roi.size == 0: continue

        label, confidence, color = "Unknown", 0.0, (0, 0, 255)

        if model:
            results = model(roi, verbose=False)
            if results and results[0].probs is not None:
                top1 = results[0].probs.top1
                confidence = results[0].probs.top1conf.item()
                label = results[0].names[top1]

                # Color coding based on your logic
                if label == "mouse_bite":
                    color = (255, 0, 0)
                elif label == "missing_hole":
                    color = (0, 255, 255)
                elif label == "spur":
                    color = (255, 0, 255)
                elif label == "open_circuit":
                    color = (0, 165, 255)

        cv2.rectangle(final_output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(final_output, f"{label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        detections.append({
            "Type": label,
            "Confidence": round(confidence, 4),
            "Area": area,
            "Location": f"({x}, {y})"
        })

    return final_output, diff, mask, detections


# ==========================================
# 3. UI: SIDEBAR
# ==========================================
st.sidebar.title("⚙️ System Controls")
template_files = [f for f in os.listdir(TEMPLATE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not template_files:
    st.sidebar.error(f"Please add a template image to: /{TEMPLATE_DIR}")
    selected_template = None
else:
    selected_template = st.sidebar.selectbox("Active Reference Template", template_files)

model_path = st.sidebar.text_input("Model Path", r"C:\Users\vinay\PycharmProjects\internship_infosys\pcb_yolo_results\run_high_acc\weights\best.pt")
model = load_yolo_model(model_path)
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.title("🔍 Automated PCB Diagnostic Hub")

if selected_template:
    t_path = os.path.join(TEMPLATE_DIR, selected_template)
    template_img = cv2.imread(t_path)

    col_ref, col_up = st.columns([1, 2])
    with col_ref:
        st.subheader("Reference Board")
        st.image(template_img, channels="BGR", use_container_width=True, caption=selected_template)

    with col_up:
        st.subheader("Batch Test Upload")
        uploaded_files = st.file_uploader("Upload Defective Boards", type=['png', 'jpg', 'jpeg'],
                                          accept_multiple_files=True)

    if uploaded_files and st.button("🚀 Start Batch Analysis"):
        master_log = []
        start_time = time.perf_counter()

        for uploaded_file in uploaded_files:
            # Convert upload to OpenCV
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            test_img = cv2.imdecode(file_bytes, 1)

            # PROCESS (Binary Mask and Diff Map are returned here)
            res_img, diff_map, binary_mask, detections = analyze_pcb(template_img, test_img, model, conf_threshold)

            # Save to log
            df_curr = pd.DataFrame(detections)
            df_curr['Filename'] = uploaded_file.name
            master_log.append(df_curr)

            # --- DISPLAY RESULTS IN TABS ---
            with st.expander(f"📋 Results for: {uploaded_file.name}", expanded=True):
                tab1, tab2, tab3 = st.tabs(["🖼️ Final Annotation", "⚫ Binary Mask", "📉 Difference Map"])

                with tab1:
                    st.image(res_img, channels="BGR", use_container_width=True)
                    if detections:
                        st.dataframe(df_curr.drop(columns=['Filename']))
                    else:
                        st.success("No defects found in this board.")

                with tab2:
                    st.image(binary_mask, caption="Module 1: Identified ROI Regions", use_container_width=True)

                with tab3:
                    st.image(diff_map, caption="Pixel Difference Intensity", use_container_width=True)

        # Performance Summary
        duration = time.perf_counter() - start_time
        st.divider()
        st.info(f"Batch completed: {len(uploaded_files)} images in {duration:.2f} seconds.")

        # Final Export
        if master_log:
            final_report = pd.concat(master_log)
            csv = final_report.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Master CSV Report", csv, "pcb_batch_report.csv", "text/csv")
else:
    st.warning("No template selected. Check your 'pcb_templates' folder.")

