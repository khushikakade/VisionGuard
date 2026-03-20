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
from collections import deque
import requests

# Set page config
st.set_page_config(
    page_title="VisionGuard | AI Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create storage directories
SNAPSHOT_DIR = "snapshots"
VIDEO_DIR = "videos"
for d in [SNAPSHOT_DIR, VIDEO_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# Configuration
BUFFER_SECONDS = 5
FRAME_RATE_ESTIMATE = 10 # Estimated FPS for the buffer
MAX_BUFFER_SIZE = BUFFER_SECONDS * FRAME_RATE_ESTIMATE

# Custom Styling (Clean Professional Enterprise Theme - Flipped)
st.markdown("""
    <style>
    /* Global App Background: Deep Corporate Maroon (Slightly Toned Down) */
    .stApp {
        background: #7A2536;
        color: #ffffff;
    }
    
    /* Sidebar Background: Clean Light Slate */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    
    /* Text overrides for Sidebar (Dark Text on Light Background) - Max Specificity */
    /* Target ALL divs and spans recursively within the sidebar */
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Target the deeply nested internal Streamlit components (Selectbox, Checkbox, Slider) */
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] span,
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    /* Main Area Typography (Light Text on Dark Background) */
    .stApp p, .stApp h1, .stApp h2, .stApp h3 {
        color: #ffffff !important;
    }

    /* Override Streamlit Sliders -> Maroon Theme */
    div[data-baseweb="slider"] div[data-testid="stTickBar"] > div { background-color: rgba(255,255,255,0.4) !important; }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #fca5a5 !important; /* Lighter maroon/rose for visibility on dark bg */
        border: 2px solid #ffffff !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    }
    div[data-baseweb="slider"] div[data-testid="stSliderTickBar"] > div { background-color: rgba(255,255,255,0.4) !important; }
    div[data-baseweb="slider"] > div > div > div:nth-child(2) { 
        background-color: #fca5a5 !important; 
    }

    /* Clean Semi-Transparent Cards (Alerts, Expanders) */
    div[data-testid="stExpander"], .stAlert, .glass-card {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        color: #ffffff !important;
    }
    
    div[data-testid="stExpander"] details summary p {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Clean Solid Buttons */
    .stButton > button {
        background-color: #7A1F2A !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15) !important;
        transition: all 0.2s ease;
    }
    
    /* Maximum Specificity: Force text inside sidebar buttons to be perfectly white */
    section[data-testid="stSidebar"] div.stButton > button,
    section[data-testid="stSidebar"] div.stButton > button p,
    section[data-testid="stSidebar"] div.stButton > button span,
    section[data-testid="stSidebar"] div.stButton > button div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        background-color: #9E1B28 !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }

    /* Clean Dashboard Title */
    h1 {
        font-weight: 700 !important;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
        color: #ffffff !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Clean Stats Cards */
    .stat-card {
        padding: 1.5rem;
        background-color: rgba(255, 255, 255, 0.05); 
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .stat-card h3 {
        color: #e2e8f0 !important; 
        font-size: 0.875rem;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        opacity: 0.9;
    }
    .stat-card .value {
        color: #ffffff !important; 
        font-size: 2.25rem;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'alerts_history' not in st.session_state:
    st.session_state.alerts_history = []
if 'frame_buffer' not in st.session_state:
    st.session_state.frame_buffer = deque(maxlen=MAX_BUFFER_SIZE)
if 'is_monitoring' not in st.session_state:
    st.session_state.is_monitoring = False

st.title("VisionGuard: AI-Powered Semantic Surveillance")

def delete_alert(timestamp):
    # Find the alert by timestamp and delete image + session state
    for i, alert in enumerate(st.session_state.alerts_history):
        if alert['timestamp'] == timestamp:
            if 'snapshot' in alert and os.path.exists(alert['snapshot']):
                try:
                    os.remove(alert['snapshot'])
                except Exception as e:
                    print(f"Error removing snapshot: {e}")
            st.session_state.alerts_history.pop(i)
            break

# --- Dynamic Threat Status Logic ---
stats_placeholder = st.empty()

def update_stats(container):
    current_status_color = "#64748b" # Default Standby
    current_status_text = "Standby"
    
    if st.session_state.is_monitoring:
        if len(st.session_state.alerts_history) > 0:
            latest_alert = st.session_state.alerts_history[0]
            try:
                latest_time = datetime.datetime.strptime(latest_alert['timestamp'], "%Y-%m-%d %H:%M:%S")
                # Calculate time_diff using local time
                time_diff = (datetime.datetime.now().astimezone() - latest_time.replace(tzinfo=datetime.datetime.now().astimezone().tzinfo)).total_seconds()
            except ValueError:
                time_diff = 0
                
            recent_score = latest_alert['score']
            
            if time_diff < 10:
                if recent_score > 0.8:
                    current_status_color = "#dc2626" # Red - High Threat
                    current_status_text = "High Alert"
                else:
                    current_status_color = "#eab308" # Yellow - Warning
                    current_status_text = "Warning"
            else:
                current_status_color = "#16a34a" # Green - Clear
                current_status_text = "Clear"
        else:
            current_status_color = "#16a34a"
            current_status_text = "Clear"

    # --- Statistics Cards ---
    container.markdown("""
    <div style="display: flex; gap: 1rem; margin-bottom: 2rem;">
        <div class="stat-card" style="flex: 1;">
            <h3>Total Alerts Recorded</h3>
            <div class="value">{}</div>
        </div>
        <div class="stat-card" style="flex: 1;">
            <h3>Threat Status</h3>
            <div class="value" style="color: {};">{}</div>
        </div>
    </div>
    """.format(
        len(st.session_state.alerts_history),
        current_status_color,
        current_status_text
    ), unsafe_allow_html=True)

update_stats(stats_placeholder)

# --- WhatsApp Notification Helper ---
def send_whatsapp_notification(phone, alert_type, score):
    """Sends a simplified WhatsApp alert via CallMeBot free API"""
    if not phone: return
    
    # CallMeBot Free API for WhatsApp
    # Note: User must once message +34 644 20 44 15 with 'I allow callmebot to send me messages'
    # For this prototype, we use a shared system key or guide the user
    apikey = "1138817" # This is a placeholder/demo key
    message = f"🚨 *VisionGuard Alert* 🚨\n\n*Type:* {alert_type}\n*Confidence:* {score}\n*Time:* {datetime.datetime.now().strftime('%H:%M:%S')}\n\n_Evidence recorded._"
    
    url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={requests.utils.quote(message)}&apikey={apikey}"
    try:
        requests.get(url, timeout=5)
    except Exception as e:
        print(f"WhatsApp Error: {e}")

# --- Sidebar Configuration ---
col1, col2, col3 = st.sidebar.columns([1, 4, 1])
with col2:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    else:
        st.markdown("<div style='text-align: center;'><img src='https://img.icons8.com/fluency/96/security-shield.png' width='120'></div>", unsafe_allow_html=True)
st.sidebar.header("System Controls")

source_option = st.sidebar.selectbox("Video Source", ["Webcam", "Local File"])
video_source = 0 if source_option == "Webcam" else st.sidebar.text_input("File Path", "demo.mp4")

sample_rate = st.sidebar.slider("Detection Interval (s)", 0.5, 5.0, 1.5, help="Seconds between AI analysis frames")
threshold_mod = st.sidebar.slider("Sensitivity Offset", -0.05, 0.05, 0.0, step=0.01, help="Adjust global detection sensitivity")

# New Option: Save Snapshots & Video
col_a, col_b = st.sidebar.columns(2)
save_snapshots = col_a.checkbox("Snapshots", value=True)
save_video = col_b.checkbox("Video Clips", value=True)

st.sidebar.subheader("Emergency Registration")
whatsapp_number = st.sidebar.text_input("WhatsApp Number", placeholder="+91XXXXXXXXXX", help="Enter number to receive instant alerts")
if whatsapp_number:
    st.sidebar.success("✅ Number Registered")

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
            
            # Save Video Segment
            if 'video_buffer' in alert and alert['video_buffer']:
                video_path = alert['video_path']
                height, width, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'avc1') # Use H.264
                if os.name == 'nt': # Windows fallback
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
                for f in alert['video_buffer']:
                    out.write(f)
                out.release()
            
            # Send WhatsApp
            if alert.get('send_whatsapp') and alert.get('phone'):
                send_whatsapp_notification(alert['phone'], alert['scenario'], alert['score'])

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

# --- Dashboard Layout (Tabs) ---
tab_live, tab_gallery = st.tabs(["Live Monitoring", "Incident Gallery"])

with tab_live:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Feed")
        video_placeholder = st.empty()
        if not st.session_state.is_monitoring:
            video_placeholder.info("Click 'Start Monitoring' in the sidebar to begin.")
            
    with col2:
        st.subheader("Activity Graph")
        graph_placeholder = st.empty()
        
        # Pre-render chart state
        if not st.session_state.alerts_history:
            dummy_data = pd.DataFrame({
                "Time": pd.Series(dtype='str'), 
                "Confidence": pd.Series(dtype='float64')
            }).set_index("Time")
            graph_placeholder.line_chart(dummy_data, height=200, color="#B3202E")
        else:
            with graph_placeholder.container():
                recent_data = st.session_state.alerts_history[:15]
                chart_df = pd.DataFrame([{
                    "Time": d["timestamp"][-6:], 
                    "Confidence": d["score"]
                } for d in reversed(recent_data)]).set_index("Time")
                st.line_chart(chart_df, height=200, color="#B3202E")
            
        st.subheader("Recent Alerts")
        alerts_placeholder = st.empty()
        
        # Pre-render alerts list
        if not st.session_state.alerts_history:
            alerts_placeholder.caption("No alerts detected yet.")
        else:
            with alerts_placeholder.container():
                for alert in st.session_state.alerts_history[:3]:
                    st.markdown(f"**[{alert['timestamp'][-6:]}]** {alert['scenario']} (`{alert['score']}`)")

with tab_gallery:
    st.subheader("Historical Incident Snapshots")
    gallery_placeholder = st.empty()
    
    # Pre-render gallery history (Only statically render if not looping to avoid duplicate keys)
    if not st.session_state.is_monitoring:
        if not st.session_state.alerts_history:
            gallery_placeholder.info("No recorded incidents to display.")
        else:
            with gallery_placeholder.container():
                cols = st.columns(3)
                for idx, alert in enumerate(st.session_state.alerts_history):
                    with cols[idx % 3]:
                        st.markdown(f"**{alert['scenario']}**")
                        st.caption(f"Score: {alert['score']} | {alert['timestamp']}")
                        if 'snapshot' in alert and os.path.exists(alert['snapshot']):
                            st.image(alert['snapshot'], use_container_width=True)
                        st.button("🗑️ Delete", key=f"del_{alert['timestamp']}", on_click=delete_alert, args=(alert['timestamp'],), use_container_width=True)
                        st.divider()

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
            
            # Maintain rolling buffer
            st.session_state.frame_buffer.append(frame.copy())
            
            # AI Detection Step
            if int(frame_count) % int(max(1, frame_interval)) == 0:
                # Perform detection
                new_alerts = st.session_state.pipeline.detect_scenarios(frame)
                
                if new_alerts:
                    for alert in new_alerts:
                        # Pre-compute paths and structure
                        if save_snapshots:
                            timestamp_str = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
                            snapshot_path = f"{SNAPSHOT_DIR}/alert_{timestamp_str}_{alert['scenario_key']}.jpg"
                            alert['snapshot'] = snapshot_path
                        
                        if save_video:
                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            video_path = f"{VIDEO_DIR}/incident_{timestamp_str}_{alert['scenario_key']}.mp4"
                            alert['video_path'] = video_path
                            # Carry over the last few seconds of video
                            alert['video_buffer'] = list(st.session_state.frame_buffer)
                            
                        # Set WhatsApp flags
                        alert['send_whatsapp'] = bool(whatsapp_number)
                        alert['phone'] = whatsapp_number
                            
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
                
                # 2. Update Activity Graph
                with graph_placeholder.container():
                    if st.session_state.alerts_history:
                        # Extract the last 15 alerts for the mini chart
                        recent_data = st.session_state.alerts_history[:15]
                        chart_df = pd.DataFrame([{
                            "Time": d["timestamp"][-6:], # Just HHMMSS for brevity
                            "Confidence": d["score"]
                        } for d in reversed(recent_data)]).set_index("Time")
                        st.line_chart(chart_df, height=200, color="#B3202E")
                
                # 3. Update Live Alerts Panel
                with alerts_placeholder.container():
                    if st.session_state.alerts_history:
                        for alert in st.session_state.alerts_history:
                            with st.expander(f"⚠️ {alert['scenario']} - {alert['timestamp']}", expanded=False):
                                st.markdown(f"**Confidence Score:** `{alert['score']}`")
                                if save_video and 'video_path' in alert:
                                    st.info(f"📹 Evidence Recorded: `{os.path.basename(alert['video_path'])}`")
                                if 'snapshot' in alert and os.path.exists(alert['snapshot']):
                                    st.image(alert['snapshot'], use_container_width=True)
                    else:
                        st.caption("Monitoring active... no threats detected.")
                        
                # 4. Update Incident Gallery Grid
                with gallery_placeholder.container():
                    if not st.session_state.alerts_history:
                        st.info("No recorded incidents to display.")
                    else:
                        cols = st.columns(3)
                        for idx, alert in enumerate(st.session_state.alerts_history):
                            with cols[idx % 3]:
                                st.markdown(f"**{alert['scenario']}**")
                                st.caption(f"Score: {alert['score']} | {alert['timestamp']}")
                                if 'snapshot' in alert and os.path.exists(alert['snapshot']):
                                    st.image(alert['snapshot'], use_container_width=True)
                                # Interactive widgets cannot be rendered inside a fast while loop
                                st.caption("⏸️ Stop monitoring to delete.")
                                st.divider()
                
                # 5. Update Threat Status
                update_stats(stats_placeholder)
                
                st.session_state.last_ui_update = current_time
            
            frame_count += 1
            time.sleep(0.01) # Yield for internal responsiveness
            
    finally:
        cap.release()
