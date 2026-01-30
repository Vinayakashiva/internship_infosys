import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
from ultralytics import YOLO
import plotly.express as px

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
st.set_page_config(page_title="Batch PCB Diagnostic System", layout="wide")

# HIGHLIGHTED TEMPLATE PATH
# Note: On Streamlit Cloud, change this to a relative path like "PCB_USED" 
# and upload the folder to your GitHub repo.
TEMPLATE_DIR = r"PCB_USED"

if not os.path.exists(TEMPLATE_DIR):
    try:
        os.makedirs(TEMPLATE_DIR)
    except:
        pass


# ==========================================
# 2. HELPER FUNCTIONS (Logic Unchanged)
# ==========================================

@st.cache_resource
def load_yolo_model(path):
    try:
        return YOLO(path)
    except:
        return None


@st.cache_resource
def load_all_templates(template_dir):
    """Loads all valid template images into a dictionary {filename: image_data}"""
    templates = {}
    if not os.path.exists(template_dir):
        return templates

    valid_exts = ('.png', '.jpg', '.jpeg')
    for f in os.listdir(template_dir):
        if f.lower().endswith(valid_exts):
            path = os.path.join(template_dir, f)
            img = cv2.imread(path)
            if img is not None:
                templates[f] = img
    return templates


def find_best_matching_template(test_img, template_dict):
    best_score = -1
    best_name = None
    best_img = None

    test_thumb = cv2.resize(test_img, (200, 200))
    test_gray = cv2.cvtColor(test_thumb, cv2.COLOR_BGR2GRAY)

    for name, tmpl_img in template_dict.items():
        tmpl_thumb = cv2.resize(tmpl_img, (200, 200))
        tmpl_gray = cv2.cvtColor(tmpl_thumb, cv2.COLOR_BGR2GRAY)
        score = cv2.matchTemplate(test_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)[0][0]

        if score > best_score:
            best_score = score
            best_name = name
            best_img = tmpl_img

    return best_name, best_img, best_score


def determine_qc_status(detections):
    if not detections: return "PASS", "✅", "#28a745"
    CRITICAL = ["open_circuit", "short", "missing_hole"]
    if any(d['Type'] in CRITICAL for d in detections):
        return "SCRAP", "❌", "#dc3545"
    return "REWORK", "⚠️", "#ffc107"


def generate_heatmap(template_shape, detections_subset):
    heatmap_mask = np.zeros((template_shape[0], template_shape[1]), dtype=np.uint8)
    for det in detections_subset:
        try:
            loc = det['Location'].replace('(', '').replace(')', '').split(',')
            x, y = int(loc[0]), int(loc[1])
            cv2.circle(heatmap_mask, (x, y), 40, (50), -1)
            cv2.circle(heatmap_mask, (x, y), 20, (100), -1)
            cv2.circle(heatmap_mask, (x, y), 5, (255), -1)
        except:
            continue
    heatmap_color = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_JET)
    return heatmap_color


def analyze_pcb(template_img, test_img, model):
    gray_template = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    if gray_template.shape != gray_test.shape:
        test_img = cv2.resize(test_img, (template_img.shape[1], template_img.shape[0]))
        gray_test = cv2.resize(gray_test, (template_img.shape[1], template_img.shape[0]))

    diff = cv2.absdiff(gray_template, gray_test)
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_output = test_img.copy()
    detections = []

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
                
                colors = {"mouse_bite": (255, 0, 0), "missing_hole": (0, 255, 255), 
                          "spur": (255, 0, 255), "open_circuit": (0, 165, 255)}
                color = colors.get(label, (0, 0, 255))

        cv2.rectangle(final_output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(final_output, f"{label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        detections.append({"Type": label, "Confidence": round(confidence, 4), "Area": area, "Location": f"({x}, {y})"})

    return final_output, detections

# ==========================================
# 3. UI & INITIALIZATION
# ==========================================
st.sidebar.title("⚙️ Configuration")
model_path = st.sidebar.text_input("Model Path", r"best.pt")
model = load_yolo_model(model_path)

template_library = load_all_templates(TEMPLATE_DIR)
st.sidebar.success(f"Loaded {len(template_library)} reference templates.")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.title("🔍 Universal PCB Diagnostic Hub")
st.markdown("### Auto-Matching Mode Enabled")

if 'selected_folder_path' not in st.session_state:
    st.session_state['selected_folder_path'] = ''

col_input, col_status = st.columns([2, 1])

with col_input:
    input_method = st.radio("Select Input:", ["File Upload", "Local Folder"], horizontal=True)
    files_to_process = []

    if input_method == "File Upload":
        uploaded_files = st.file_uploader("Upload Mixed PCB Batch", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        if uploaded_files:
            files_to_process = [(f.name, f, 'streamlit_upload') for f in uploaded_files]
    else:
        # Replaced Tkinter popup with a direct text input for the folder path
        user_path = st.text_input("Enter Local Folder Path:", st.session_state['selected_folder_path'])
        if user_path:
            st.session_state['selected_folder_path'] = user_path
            if os.path.exists(user_path):
                local_files = [f for f in os.listdir(user_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
                files_to_process = [(f, os.path.join(user_path, f), 'local_path') for f in local_files]
            else:
                st.error("Path not found.")

# --- BATCH PROCESSING ---
if files_to_process and st.button("🚀 Start Universal Scan"):
    if not template_library:
        st.error(f"No templates found in {TEMPLATE_DIR}.")
        st.stop()

    master_log, all_global_detections = [], []
    progress_bar = st.progress(0)
    results_container = st.container()

    for i, (fname, f_source, f_type) in enumerate(files_to_process):
        if f_type == 'streamlit_upload':
            file_bytes = np.asarray(bytearray(f_source.read()), dtype=np.uint8)
            test_img = cv2.imdecode(file_bytes, 1)
            f_source.seek(0)
        else:
            test_img = cv2.imread(f_source)

        if test_img is not None:
            matched_name, matched_tmpl, score = find_best_matching_template(test_img, template_library)
            if score < 0.5:
                st.warning(f"No match for {fname}. Skipping.")
                continue

            res_img, detections = analyze_pcb(matched_tmpl, test_img, model)
            qc_status, qc_icon, qc_color = determine_qc_status(detections)

            df_curr = pd.DataFrame(detections)
            df_curr['Filename'], df_curr['Matched_Template'], df_curr['QC_Decision'] = fname, matched_name, qc_status
            master_log.append(df_curr)
            for d in detections:
                d['Matched_Template'] = matched_name
                all_global_detections.append(d)

            with results_container:
                st.markdown(f'<div style="border: 1px solid #444; padding: 10px; border-radius: 5px; margin-bottom: 5px; background-color: #262730;"><b>{fname}</b><span style="float:right; background-color:{qc_color}; padding: 2px 8px; border-radius: 4px; color:white;">{qc_icon} {qc_status}</span></div>', unsafe_allow_html=True)
                with st.expander("Show Analysis"):
                    c1, c2 = st.columns(2)
                    c1.image(res_img, channels="BGR", caption="Defect Map")
                    c2.dataframe(df_curr)
        progress_bar.progress((i + 1) / len(files_to_process))

    if master_log:
        st.divider()
        final_report = pd.concat(master_log)
        st.markdown("## 📊 Manufacturing Insights Dashboard")
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            unique_boards = final_report[['Filename', 'QC_Decision']].drop_duplicates()
            fig_status = px.bar(unique_boards['QC_Decision'].value_counts().reset_index(), x='QC_Decision', y='count', color='QC_Decision', 
                                color_discrete_map={"PASS": "#28a745", "REWORK": "#ffc107", "SCRAP": "#dc3545"}, title="Batch Yield")
            st.plotly_chart(fig_status, use_container_width=True)
        with c_m2:
            fig_pie = px.pie(final_report['Type'].value_counts().reset_index(), values='count', names='Type', title='Defect Distribution', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("🔥 Root Cause Maps")
        for template_name in final_report['Matched_Template'].unique():
            subset = [d for d in all_global_detections if d.get('Matched_Template') == template_name]
            if subset and template_name in template_library:
                ref_img = template_library[template_name]
                overlay = cv2.addWeighted(ref_img, 0.7, generate_heatmap(ref_img.shape, subset), 0.6, 0)
                with st.expander(f"Heatmap: {template_name}", expanded=True):
                    st.image(overlay, channels="BGR", use_container_width=True)

        st.download_button("📥 Download Report (CSV)", final_report.to_csv(index=False).encode('utf-8'), "report.csv", "text/csv")
