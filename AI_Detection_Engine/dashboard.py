import streamlit as st
import cv2
import pandas as pd
import time
from detection_pipeline import DetectionPipeline
import os
from PIL import Image
import datetime

# Set page config
st.set_page_config(
    page_title="VisionGuard | AI Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create storage directories
SNAPSHOT_DIR = "snapshots"
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stAlert {
        border-radius: 10px;
    }
    .alert-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: white;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("VisionGuard: AI-Powered Semantic Surveillance")

# --- Session State Initialization ---
if 'pipeline' not in st.session_state:
    with st.spinner("Initializing AI Detection Engine (CLIP ViT-B/32)..."):
        st.session_state.pipeline = DetectionPipeline()
    st.session_state.alerts_history = []
    st.session_state.is_monitoring = False

# --- Sidebar Configuration ---
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=150)
else:
    st.sidebar.image("https://img.icons8.com/fluency/96/security-shield.png", width=80)
st.sidebar.header("System Controls")

source_option = st.sidebar.selectbox("Video Source", ["Webcam", "Local File"])
video_source = 0 if source_option == "Webcam" else st.sidebar.text_input("File Path", "demo.mp4")

sample_rate = st.sidebar.slider("Detection Interval (s)", 0.5, 5.0, 1.5, help="Seconds between AI analysis frames")
threshold_mod = st.sidebar.slider("Sensitivity Offset", -0.05, 0.05, 0.0, step=0.01, help="Adjust global detection sensitivity")

# New Option: Save Snapshots
save_snapshots = st.sidebar.checkbox("Save Incident Snapshots", value=True)

st.sidebar.divider()
system_status = st.sidebar.empty()
if st.session_state.is_monitoring:
    system_status.error("● System Active")
else:
    system_status.success("○ System Ready")

# --- Dashboard Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Monitoring Feed")
    video_placeholder = st.empty()
    if not st.session_state.is_monitoring:
        video_placeholder.info("Click 'Start Monitoring' in the sidebar to begin.")

with col2:
    st.subheader("Real-time Alerts")
    alerts_placeholder = st.empty()
    if not st.session_state.alerts_history:
        alerts_placeholder.caption("No alerts detected yet.")

# --- Monitoring Logic ---
log_file = "event_log.csv"

start_btn = st.sidebar.button("Start Monitoring", type="primary", use_container_width=True)
stop_btn = st.sidebar.button("Stop Monitoring", use_container_width=True)

if start_btn:
    st.session_state.is_monitoring = True
    st.rerun()

if stop_btn:
    st.session_state.is_monitoring = False
    st.rerun()

if st.session_state.is_monitoring:
    cap = cv2.VideoCapture(video_source)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    frame_interval = int(fps * sample_rate)
    frame_count = 0
    
    try:
        while cap.isOpened() and st.session_state.is_monitoring:
            ret, frame = cap.read()
            if not ret:
                st.error("End of video stream or camera disconnected.")
                st.session_state.is_monitoring = False
                break
            
            # AI Detection Step
            if frame_count % frame_interval == 0:
                # Perform detection
                new_alerts = st.session_state.pipeline.detect_scenarios(frame)
                
                if new_alerts:
                    # Update local history and save snapshots if enabled
                    for alert in new_alerts:
                        if save_snapshots:
                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            snapshot_path = f"{SNAPSHOT_DIR}/alert_{timestamp_str}_{alert['scenario_key']}.jpg"
                            cv2.imwrite(snapshot_path, frame)
                            alert['snapshot'] = snapshot_path
                            
                        st.session_state.alerts_history.insert(0, alert)
                    
                    # Log to CSV
                    df_new = pd.DataFrame(new_alerts)
                    df_new.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
                    
                    # Keep history manageable
                    st.session_state.alerts_history = st.session_state.alerts_history[:20]

            # UI Updates
            # 1. Update Video Feed
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # 2. Update Alerts Panel
            with alerts_placeholder.container():
                if st.session_state.alerts_history:
                    for alert in st.session_state.alerts_history:
                        st.markdown(f"""
                        <div class="alert-card">
                            <strong>{alert['scenario']}</strong><br>
                            <small>Time: {alert['timestamp']} | Score: {alert['score']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                        if 'snapshot' in alert and os.path.exists(alert['snapshot']):
                            st.image(alert['snapshot'], use_container_width=True)
                else:
                    st.caption("Monitoring active... no threats detected.")
            
            frame_count += 1
            time.sleep(0.01) # Yield for UI responsiveness
            
    finally:
        cap.release()
