import streamlit as st
import cv2
import pandas as pd
import time
from detection_pipeline import DetectionPipeline
import os
from PIL import Image
import datetime
import threading
import queue

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

# Custom Styling (Professional Maroon Theme)
st.markdown("""
    <style>
    /* Global App Background: Deep Maroon with Subtle Gradient */
    .stApp {
        background: linear-gradient(180deg, #5A0F1B 0%, #4A0C16 100%);
        color: #ffffff;
    }
    
    /* Sidebar Background: Lighter Maroon Contrast */
    [data-testid="stSidebar"] {
        background-color: #F5B5B5 !important;
        border-right: 1px solid rgba(255,255,255,0.1) !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.2);
    }
    
    /* Text overrides to ensure pure readability and clean aesthetics */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2 {
        color: #f8fafc !important;
        font-weight: 500 !important;
    }
    
    .stApp p, .stApp h1, .stApp h2, .stApp h3 {
        color: #ffffff !important;
    }

    /* Override Streamlit Orange Sliders natively -> Deep Red/Maroon */
    div[data-baseweb="slider"] div[data-testid="stTickBar"] > div { background-color: rgba(255,255,255,0.4) !important; }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #B3202E !important; 
        border: 2px solid #ffffff !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    }
    div[data-baseweb="slider"] div[data-testid="stSliderTickBar"] > div { background-color: rgba(255,255,255,0.4) !important; }
    div[data-baseweb="slider"] > div > div > div:nth-child(2) { 
        background-color: #B3202E !important; /* The filled rail track */
    }

    /* Professional Glassmorphism Cards (Alerts, Expanders) */
    div[data-testid="stExpander"], .stAlert, .glass-card {
        background-color: rgba(90, 15, 27, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        color: #ffffff !important;
    }
    
    div[data-testid="stExpander"] details summary p {
        color: #f8fafc !important;
        font-weight: 600;
    }

    /* Professional Buttons */
    .stButton > button {
        background-color: #9E1B28 !important;
        color: white !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #B3202E !important;
        border-color: rgba(255,255,255,0.4) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }

    /* Clean Dashboard Title */
    h1 {
        font-weight: 700 !important;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Clean Stats Cards */
    .stat-card {
        padding: 1.5rem;
        background-color: rgba(122, 31, 42, 0.5); /* Sidebar color, but transparent */
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stat-card h3 {
        color: #e2e8f0 !important; 
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        opacity: 0.9;
    }
    .stat-card .value {
        color: #ffffff !important; 
        font-size: 2.5rem;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'alerts_history' not in st.session_state:
    st.session_state.alerts_history = []
if 'is_monitoring' not in st.session_state:
    st.session_state.is_monitoring = False

st.title("VisionGuard: AI-Powered Semantic Surveillance")

# --- Statistics Cards ---
st.markdown("""
<div style="display: flex; gap: 1rem; margin-bottom: 2rem;">
    <div class="stat-card" style="flex: 1;">
        <h3>Total Alerts Recorded</h3>
        <div class="value">{}</div>
    </div>
    <div class="stat-card" style="flex: 1;">
        <h3>System Status</h3>
        <div class="value" style="color: {};">{}</div>
    </div>
</div>
""".format(
    len(st.session_state.alerts_history),
    "#86efac" if st.session_state.is_monitoring else "#94a3b8",
    "Active" if st.session_state.is_monitoring else "Standby"
), unsafe_allow_html=True)

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

# --- Background Save Thread Initialization ---
if 'save_queue' not in st.session_state:
    st.session_state.save_queue = queue.Queue()
    
    def background_saver(q):
        while True:
            item = q.get()
            if item is None:
                break
            
            # Unpack task
            frame, alert, save_snapshots, log_file = item
            
            # Save Snapshot
            if save_snapshots:
                cv2.imwrite(alert['snapshot'], frame)
            
            # Append to CSV
            df_new = pd.DataFrame([alert])
            if not os.path.exists(log_file):
                df_new.to_csv(log_file, mode='w', header=True, index=False)
            else:
                df_new.to_csv(log_file, mode='a', header=False, index=False)
            
            q.task_done()
            
    # Start thread
    t = threading.Thread(target=background_saver, args=(st.session_state.save_queue,), daemon=True)
    t.start()
    st.session_state.saver_thread = t

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

if not st.session_state.is_monitoring:
    if st.sidebar.button("Start Monitoring", type="primary", use_container_width=True):
        if st.session_state.pipeline is None:
            with st.spinner("Initializing AI Detection Engine (CLIP ViT-B/32)..."):
                st.session_state.pipeline = DetectionPipeline()
        st.session_state.is_monitoring = True
        st.rerun()
else:
    if st.sidebar.button("Stop Monitoring", type="primary", use_container_width=True):
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
            if int(frame_count) % int(max(1, frame_interval)) == 0:
                # Perform detection
                new_alerts = st.session_state.pipeline.detect_scenarios(frame)
                
                if new_alerts:
                    for alert in new_alerts:
                        # Pre-compute paths and structure
                        if save_snapshots:
                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            snapshot_path = f"{SNAPSHOT_DIR}/alert_{timestamp_str}_{alert['scenario_key']}.jpg"
                            alert['snapshot'] = snapshot_path
                            
                        st.session_state.alerts_history.insert(0, alert)
                    
                        # Queue the heavy I/O tasks to the background thread
                        st.session_state.save_queue.put((frame.copy(), alert, save_snapshots, log_file))
                    
                    # Keep history manageable
                    st.session_state.alerts_history = st.session_state.alerts_history[:20]

            # Throttled UI Updates (Targeting ~10 FPS max for the browser UI)
            current_time = time.time()
            if not hasattr(st.session_state, 'last_ui_update'):
                st.session_state.last_ui_update = 0
                
            if current_time - st.session_state.last_ui_update >= 0.1: # 0.1s = 10 updates per second max
                # 1. Update Video Feed
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # 2. Update Alerts Panel
                with alerts_placeholder.container():
                    if st.session_state.alerts_history:
                        for alert in st.session_state.alerts_history:
                            with st.expander(f"⚠️ {alert['scenario']} - {alert['timestamp']}", expanded=False):
                                st.markdown(f"**Confidence Score:** `{alert['score']}`")
                                if 'snapshot' in alert and os.path.exists(alert['snapshot']):
                                    st.image(alert['snapshot'], use_container_width=True)
                    else:
                        st.caption("Monitoring active... no threats detected.")
                
                st.session_state.last_ui_update = current_time
            
            frame_count += 1
            time.sleep(0.01) # Yield for internal responsiveness
            
    finally:
        cap.release()
