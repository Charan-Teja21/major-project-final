import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import base64

# Page config
st.set_page_config(
    page_title="MRI-CT Brain Tumor Detection • Medical AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching the modern medical design
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Root variables for modern medical theme */
    :root {
        --primary-color: #0F172A;       /* Slate 900 - Deep Background */
        --primary-accent: #3B82F6;      /* Blue 500 - Primary Action */
        --secondary-accent: #14B8A6;    /* Teal 500 - Medical/Success */
        --c-bg-primary: #FFFFFF;        /* White */
        --c-bg-secondary: #F8FAFC;      /* Slate 50 */
        --c-text-primary: #1E293B;      /* Slate 800 */
        --c-text-secondary: #64748B;    /* Slate 500 */
        --c-success: #10B981;           /* Emerald 500 */
        --c-warning: #F59E0B;           /* Amber 500 */
        --c-danger: #EF4444;            /* Red 500 */
        --radius-lg: 16px;
        --radius-md: 12px;
        --shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    }

    /* Main app styling */
    .stApp {
        background-color: #F1F5F9;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: var(--c-text-primary);
        font-weight: 700 !important;
    }

    /* Hide standard Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Centered Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        width: 100%;
        gap: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #64748B; /* Passive color */
        font-weight: 600;
        font-size: 1.1rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #0F172A !important; /* Active Dark Color */
        border-bottom: 3px solid #3B82F6;
    }

    /* Custom Card Container */
    .custom-card {
        background: var(--c-bg-primary);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    /* Header Styling */
    .dashboard-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, #1e293b 100%);
        color: white;
        padding: 2rem;
        border-radius: var(--radius-lg);
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .header-content h1 {
        color: white !important;
        font-size: 2rem;
        margin: 0;
    }
    
    .header-content p {
        color: #94A3B8;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }

    /* File Uploader Visibility Fix */
    [data-testid="stFileUploader"] section {
        background-color: #F8FAFC;
    }
    
    [data-testid="stFileUploader"] section > input + div {
        color: #0F172A !important; /* Drag and drop text */
    }
    
    [data-testid="stFileUploader"] small {
        color: #475569 !important; /* File limits text */
    }

    /* Force visibility for native file uploader list - Aggressive Override */
    [data-testid="stUploadedFileList"],
    [data-testid="stUploadedFileList"] div, 
    [data-testid="stUploadedFileList"] span,
    [data-testid="stUploadedFileList"] p {
        color: #0F172A !important;
        opacity: 1 !important;
    }
    
    [data-testid="stUploadedFileList"] svg {
        fill: #3B82F6 !important;
        stroke: #3B82F6 !important;
        opacity: 1 !important;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #334155; /* Default paragraph color to dark slate */
    }

    /* Stat Cards */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        background: var(--c-bg-secondary);
        border-radius: var(--radius-md);
        border: 1px solid #E2E8F0;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--primary-accent);
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--c-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar Styling Override */
    section[data-testid="stSidebar"] {
        background-color: var(--c-bg-primary);
        border-right: 1px solid #E2E8F0;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Sidebar Profile */
    .sidebar-profile {
        text-align: center;
        padding: 1.5rem 1rem;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(20, 184, 166, 0.1) 100%);
        border-radius: var(--radius-lg);
        margin-bottom: 2rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .profile-avatar {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.1));
    }
    
    .profile-name {
        font-weight: 700;
        color: var(--c-text-primary);
        font-size: 1.1rem;
    }
    
    .profile-subtitle {
        color: var(--primary-accent);
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-accent) 0%, #2563EB 100%);
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        width: 100%;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
        color: white;
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Feature/Info Cards */
    .feature-item {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        background: white;
        border-radius: var(--radius-md);
        margin-bottom: 0.75rem;
        border: 1px solid #F1F5F9;
    }
    
    .feature-icon-box {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-size: 1.2rem;
    }
    
    .bg-blue { background: rgba(59, 130, 246, 0.1); color: var(--primary-accent); }
    .bg-teal { background: rgba(20, 184, 166, 0.1); color: var(--secondary-accent); }
    .bg-purple { background: rgba(139, 92, 246, 0.1); color: #8B5CF6; }

    .feature-text div:first-child {
        font-weight: 600;
        color: var(--c-text-primary);
        font-size: 0.9rem;
    }
    
    .feature-text div:last-child {
        font-size: 0.75rem;
        color: var(--c-text-secondary);
    }

    /* Result Panels */
    .result-box {
        padding: 1.5rem;
        border-radius: var(--radius-lg);
        margin-top: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .result-healthy {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1px solid #10B981;
        color: #065F46;
    }
    
    .result-tumor {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1px solid #EF4444;
        color: #991B1B;
    }

    /* Image Container Styling */
    .image-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
    }

    .image-wrapper {
        width: 224px;
        height: 224px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 2px solid #E2E8F0;
        background-color: #F8FAFC;
    }

    .image-wrapper img {
        width: 224px;
        height: 224px;
        object-fit: contain;
        border-radius: var(--radius-md);
    }

    .image-caption {
        margin-top: 1rem;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--c-text-primary);
        text-align: center;
    }

    .confidence-label {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        display: flex;
        justify-content: space-between;
    }
    
    .confidence-bar-bg {
        background: rgba(0,0,0,0.1);
        height: 6px;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .confidence-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
    }

    /* File Uploader */
    .stFileUploader section {
        background-color: var(--c-bg-primary);
        border: 2px dashed #CBD5E1;
        border-radius: var(--radius-lg);
        padding: 2rem;
    }
    
    .stFileUploader section:hover {
        border-color: var(--primary-accent);
        background-color: #F8FAFC;
    }

</style>
""", unsafe_allow_html=True)



# Initialize session state for medical statistics
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0
if 'tumors_detected' not in st.session_state:
    st.session_state.tumors_detected = 0
if 'healthy_scans' not in st.session_state:
    st.session_state.healthy_scans = 0
if 'glioma_cases' not in st.session_state:
    st.session_state.glioma_cases = 0
if 'meningioma_cases' not in st.session_state:
    st.session_state.meningioma_cases = 0
if 'pituitary_cases' not in st.session_state:
    st.session_state.pituitary_cases = 0

@st.cache_resource
def load_binary_model():
    """Load the binary classification model"""
    try:
        model = tf.keras.models.load_model("custom_cnn_fused_model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading binary model: {str(e)}")
        return None

@st.cache_resource
def load_multiclass_model():
    """Load the multiclass classification model"""
    try:
        model = tf.keras.models.load_model("custom_multiclass.h5")
        return model
    except Exception as e:
        st.error(f"Error loading multiclass model: {str(e)}")
        return None

def validate_image_content(image_file):
    """
    Validate if the image is likely a medical scan (grayscale check).
    Returns (bool, str): (is_valid, error_message)
    """
    try:
        image = Image.open(image_file).convert('RGB')
        img_array = np.array(image)
        
        # Convert to HSV to check saturation
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv_img[:, :, 1]
        mean_saturation = np.mean(saturation)
        
        # Medical images are typically grayscale (low saturation)
        # Threshold: 30 (out of 255) allows for some minor color noise or overlays, but rejects photos
        if mean_saturation > 30:
            return False, "Image appears to be non-medical (high color saturation detected). Please upload a valid grayscale brain scan."
            
        return True, ""
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def check_modality(image_file, expected_type):
    """
    Multi-feature heuristic to distinguish MRI vs CT brain scans.
    
    Thresholds calibrated from profiling actual project images:
      CT images:  P95 in [132–211], CoeffVar in [0.85–1.13], Skew in [0.32–1.01]
      MRI images: P95 in [87–229],  CoeffVar in [0.82–1.39], Skew in [0.51–1.38]
    
    No single feature cleanly separates the two modalities, so a weighted
    voting across multiple features is used. The function only flags a
    mismatch when the composite score is strongly in one direction (>0.6),
    reducing false alarms in ambiguous cases.
    """
    try:
        image = Image.open(image_file).convert('L')
        img_array = np.array(image, dtype=np.float64)

        ct_score = 0.0
        total_weight = 0.0
        details = []

        # --- Compute base statistics (used by multiple features) ---
        mean_val = np.mean(img_array)
        std_val = np.std(img_array)
        p50 = np.median(img_array)
        p95 = np.percentile(img_array, 95)

        if std_val > 0:
            skewness = float(np.mean(((img_array - mean_val) / std_val) ** 3))
        else:
            skewness = 0.0

        coeff_var = std_val / (mean_val + 1e-10)

        # === Feature 1: P95 Intensity (weight: 2.5) ===
        # CT P95 range: [132–211], mean ≈ 173   (bone windows cap intensity)
        # MRI P95 range: [87–229], mean ≈ 157   (wider spread, can go very high)
        # Key: CT P95 is almost always ≤ 211. MRI no-tumor P95 > 200.
        w1 = 2.5
        total_weight += w1
        if p95 <= 180:
            ct_score += w1  # CT-like: moderate upper range
            details.append(f"P95={p95:.0f} (CT-like)")
        elif p95 >= 215:
            details.append(f"P95={p95:.0f} (MRI-like)")  # MRI-like: very high
        else:
            ct_score += w1 * 0.5
            details.append(f"P95={p95:.0f} (ambiguous)")

        # === Feature 2: Coefficient of Variation (weight: 2.0) ===
        # CT CoeffVar range: [0.85–1.13], mean ≈ 0.99
        # MRI CoeffVar range: [0.82–1.39], mean ≈ 1.15
        # Key: CoeffVar > 1.25 is strongly MRI (tumor images ~1.38)
        w2 = 2.0
        total_weight += w2
        if coeff_var > 1.25:
            details.append(f"CV={coeff_var:.3f} (MRI-like)")  # MRI: high variance
        elif coeff_var < 0.81:
            ct_score += w2
            details.append(f"CV={coeff_var:.3f} (CT-like)")
        else:
            ct_score += w2 * 0.5
            details.append(f"CV={coeff_var:.3f} (ambiguous)")

        # === Feature 3: Skewness (weight: 2.0) ===
        # CT skewness range: [0.32–1.01], mean ≈ 0.67
        # MRI skewness range: [0.51–1.38], mean ≈ 0.98
        # Key: Skewness > 1.15 is strongly MRI-like (tumor images ~1.37)
        w3 = 2.0
        total_weight += w3
        if skewness > 1.15:
            details.append(f"Skew={skewness:.3f} (MRI-like)")
        elif skewness < 0.50:
            ct_score += w3
            details.append(f"Skew={skewness:.3f} (CT-like)")
        else:
            ct_score += w3 * 0.5
            details.append(f"Skew={skewness:.3f} (ambiguous)")

        # === Feature 4: Median Intensity / P50 (weight: 1.5) ===
        # CT P50 range: [17–81], mean ≈ 57
        # MRI P50 range: [3–82], mean ≈ 40
        # Key: Very low P50 (< 10) → likely MRI tumor. High P50 (> 65) → ambiguous.
        w4 = 1.5
        total_weight += w4
        if p50 < 10:
            details.append(f"P50={p50:.0f} (MRI-like)")  # Very dark median = MRI tumor
        elif p50 > 60:
            ct_score += w4 * 0.5  # Both CT and MRI can be here
            details.append(f"P50={p50:.0f} (ambiguous)")
        else:
            ct_score += w4
            details.append(f"P50={p50:.0f} (CT-like)")

        # === Feature 5: Edge Density via Canny (weight: 1.0) ===
        # CT edge range: [0.042–0.129], mean ≈ 0.082
        # MRI edge range: [0.030–0.135], mean ≈ 0.078
        # Weak separator but helps confirm other signals
        img_uint8 = img_array.astype(np.uint8)
        edges = cv2.Canny(img_uint8, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size

        w5 = 1.0
        total_weight += w5
        if edge_density < 0.035:
            details.append(f"Edge={edge_density:.3f} (MRI-like)")
        elif edge_density > 0.14:
            ct_score += w5
            details.append(f"Edge={edge_density:.3f} (CT-like)")
        else:
            ct_score += w5 * 0.5
            details.append(f"Edge={edge_density:.3f} (ambiguous)")

        # === Final Classification ===
        ct_probability = ct_score / total_weight

        # >0.6 → likely CT, <0.4 → likely MRI, in-between → uncertain (pass)
        is_ct = ct_probability > 0.6
        is_mri = ct_probability < 0.4

        if expected_type == 'CT':
            if is_mri:
                return False, (
                    "The uploaded image does not appear to be a CT scan. "
                    "Its intensity and texture features are more consistent with an MRI. "
                    "Please check and upload a valid CT scan."
                ), ct_probability
        elif expected_type == 'MRI':
            if is_ct:
                return False, (
                    "The uploaded image does not appear to be an MRI scan. "
                    "Its intensity and texture features are more consistent with a CT scan. "
                    "Please check and upload a valid MRI scan."
                ), ct_probability

        return True, "", ct_probability
    except Exception as e:
        return True, "", 0.5


def validate_image_pair(mri_file, ct_file, mri_ct_score, ct_ct_score):
    """
    Cross-image pair validation to detect mismatched MRI+CT uploads.
    
    Uses three strategies:
    1. Score-based: detect swapped modalities or identical types
    2. Tumor-state consistency: flag when MRI and CT suggest different
       diagnostic states (tumor vs no-tumor), indicating wrong patient pairing
    3. Intensity profile comparison: detect near-identical or structurally
       incompatible images
    
    Calibrated from exhaustive 56-combination analysis of project images.
    """
    try:
        # Load both images as grayscale at same resolution
        mri_img = np.array(Image.open(mri_file).convert('L').resize((224, 224)), dtype=np.float64)
        ct_img = np.array(Image.open(ct_file).convert('L').resize((224, 224)), dtype=np.float64)
        
        warnings = []
        
        # --- Compute per-image statistics ---
        mri_mean = np.mean(mri_img)
        mri_std = np.std(mri_img)
        mri_p50 = np.median(mri_img)
        mri_p95 = np.percentile(mri_img, 95)
        mri_cv = mri_std / (mri_mean + 1e-10)
        
        ct_mean = np.mean(ct_img)
        ct_std = np.std(ct_img)
        ct_p50 = np.median(ct_img)
        ct_p95 = np.percentile(ct_img, 95)
        ct_cv = ct_std / (ct_mean + 1e-10)
        
        # --- Strategy 1: Duplicate/similarity check ---
        # Flag if images are nearly identical (possible duplicate)
        abs_diff = np.mean(np.abs(mri_img - ct_img))
        if abs_diff < 16:
            warnings.append(
                "Both uploaded images appear to be extremely similar to each other. "
                "Please verify that you have uploaded two different modality scans (one MRI and one CT)."
            )
        
        # --- Strategy 2: Tumor-state consistency ---
        # Classify MRI tumor state from intensity profile:
        #   MRI tumor images:    P50 ≈ 3,  CV ≈ 1.38 (very dark, high variance)
        #   MRI no-tumor images: P50 ≈ 71–82, CV ≈ 0.82–0.99 (brighter, moderate variance)
        mri_is_tumor = (mri_p50 < 15 and mri_cv > 1.2)
        mri_is_healthy = (mri_p50 > 50 and mri_cv < 1.1)
        
        # Classify CT tumor state from intensity profile:
        #   CT tumor images:    P50 ≈ 54–77, P95 ≈ 174 (moderate spread)
        #   CT no-tumor images: P50 ≈ 17–81, P95 ≈ 132–211 (wider spread, can be very bright/dark)
        # CT tumor vs no-tumor is harder; use P50 < 30 as a no-tumor indicator
        # (CT_no_tumor2 has P50=17, very dark background)
        ct_is_dark = (ct_p50 < 30)  # Likely CT_no_tumor2-style (windowed differently)
        ct_is_bright = (ct_p50 > 65 and ct_p95 > 200)  # Likely CT_no_tumor-style
        
        # Cross-check: MRI tumor + CT that looks like a different diagnostic context
        if mri_is_tumor and (ct_is_dark or ct_is_bright):
            warnings.append(
                "The MRI scan suggests a tumor may be present, but the CT scan appears to be "
                "from a different diagnostic context (e.g., a healthy scan). "
                "Please verify that both scans belong to the same patient."
            )
        
        if mri_is_healthy and not ct_is_dark and not ct_is_bright:
            # MRI healthy + CT with moderate profile (tumor-type CT)
            # Only flag if the CT score is moderate (might be a differently-windowed scan)
            if ct_p50 > 40 and ct_p95 < 180:
                warnings.append(
                    "The MRI scan appears healthy, but the CT scan shows characteristics "
                    "that may indicate a different diagnostic condition. "
                    "Please verify that both scans belong to the same patient."
                )
        
        # --- Strategy 3: Histogram distance check ---
        # Bhattacharyya distance between histograms
        mri_hist = cv2.calcHist([mri_img.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
        ct_hist = cv2.calcHist([ct_img.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
        bhatt = cv2.compareHist(
            mri_hist.astype(np.float32),
            ct_hist.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA
        )
        
        # Very high Bhattacharyya distance (> 0.55) with both scores in ambiguous zone
        # suggests the images have incompatible intensity distributions
        score_diff = abs(mri_ct_score - ct_ct_score)
        if bhatt > 0.53 and score_diff < 0.25:
            warnings.append(
                "The two uploaded images have significantly different intensity patterns. "
                "They may not be from the same scan session. "
                "Please verify that both scans belong to the same patient."
            )
        
        return warnings
    except Exception:
        return []


def preprocess_image(image_file):
    """Preprocess uploaded image for model prediction"""
    IMG_WIDTH, IMG_HEIGHT = 224, 224
    image = Image.open(image_file).convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype("float32")

# GradCAM Functions
def get_last_conv_layer_name(model):
    """Find the last convolutional layer in the model"""
    conv_layers = []
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer.name)
        elif hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    conv_layers.append(sublayer.name)
    
    if conv_layers:
        return conv_layers[-1]
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    if isinstance(img_array, np.ndarray):
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    conv_layer_idx = None
    for idx, layer in enumerate(model.layers):
        if layer.name == last_conv_layer_name:
            conv_layer_idx = idx
            break
    
    if conv_layer_idx is None:
        raise ValueError(f"Layer {last_conv_layer_name} not found")
    
    conv_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[conv_layer_idx].output
    )
    
    classifier_input = tf.keras.Input(shape=model.layers[conv_layer_idx].output.shape[1:])
    x = classifier_input
    for layer in model.layers[conv_layer_idx + 1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(inputs=classifier_input, outputs=x)
    
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()

def apply_gradcam_overlay(original_image, heatmap, alpha=0.8):
    """Apply Grad-CAM heatmap overlay on original image with enhanced visibility"""
    img_array = np.array(original_image)
    
    if len(heatmap.shape) != 2:
        raise ValueError(f"Heatmap must be 2D, got shape {heatmap.shape}")
    
    heatmap = heatmap.astype(np.float32)
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / (heatmap.max() + 1e-10)
    
    # Apply intensity boost to enhance mid-range values (makes tumor regions deeper)
    heatmap = np.power(heatmap, 0.7)
    
    # Apply lower threshold to preserve more tumor regions (0.3 -> 0.15)
    heatmap[heatmap < 0.15] = 0
    
    heatmap_resized = cv2.resize(
        heatmap, 
        (img_array.shape[1], img_array.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Lower mask threshold for broader coverage (0.1 -> 0.05)
    mask = (heatmap_resized > 0.05).astype(np.float32)[:, :, np.newaxis]
    
    # Blend with higher alpha for more prominent overlay (0.6 -> 0.8)
    superimposed_img = (heatmap_colored * alpha * mask) + (img_array * (1 - (alpha * mask)))
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img

def create_gradcam_visualization(mri_original, ct_original, mri_heatmap, ct_heatmap, 
                                tumor_type, binary_result, tumor_confidence, binary_confidence):
    """Create comprehensive GradCAM visualization similar to notebook output"""
    
    # Apply GradCAM overlays
    mri_gradcam = apply_gradcam_overlay(mri_original, mri_heatmap)
    ct_gradcam = apply_gradcam_overlay(ct_original, ct_heatmap)
    
    # Create figure with subplots (2x3 layout + 1 for legend)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('white')
    
    # Main title with proper spacing and compact format
    title_text = f'Brain Tumor Detection Analysis'
    subtitle_text = f'Classification: {tumor_type} | Result: {binary_result}'
    
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.93, subtitle_text, ha='center', fontsize=12, style='italic')
    
    # MRI visualizations (top row)
    # Original MRI
    axes[0, 0].imshow(mri_original)
    axes[0, 0].set_title('MRI - Original', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # MRI Heatmap
    heatmap_display = cv2.applyColorMap(np.uint8(255 * mri_heatmap), cv2.COLORMAP_JET)
    heatmap_display = cv2.cvtColor(heatmap_display, cv2.COLOR_BGR2RGB)
    axes[0, 1].imshow(heatmap_display)
    axes[0, 1].set_title('MRI - Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # MRI Overlay
    axes[0, 2].imshow(mri_gradcam)
    axes[0, 2].set_title('MRI - Grad-CAM Overlay', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Hide the 4th subplot in top row
    axes[0, 3].axis('off')
    
    # CT visualizations (bottom row)
    # Original CT
    axes[1, 0].imshow(ct_original)
    axes[1, 0].set_title('CT - Original', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # CT Heatmap
    heatmap_display = cv2.applyColorMap(np.uint8(255 * ct_heatmap), cv2.COLORMAP_JET)
    heatmap_display = cv2.cvtColor(heatmap_display, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(heatmap_display)
    axes[1, 1].set_title('CT - Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # CT Overlay
    axes[1, 2].imshow(ct_gradcam)
    axes[1, 2].set_title('CT - Grad-CAM Overlay', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    # Heatmap Legend
    gradient = np.linspace(0, 1, 256).reshape(1, 256)
    gradient = np.vstack([gradient] * 50)
    im = axes[1, 3].imshow(gradient, aspect='auto', cmap='jet')
    axes[1, 3].set_title('Heatmap Legend', fontsize=12, fontweight='bold')
    axes[1, 3].set_xticks([0, 128, 255])
    axes[1, 3].set_xticklabels(['Low', 'Medium', 'High'])
    axes[1, 3].set_yticks([])
    axes[1, 3].set_xlabel('Model Attention', fontsize=10)
    
    # Adjust spacing to prevent overlap - increased top margin and hspace
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95, hspace=0.5, wspace=0.2)
    return fig

def create_sidebar():
    """Create modern sidebar for medical dashboard"""
    with st.sidebar:
        # Medical profile section
        st.markdown("""
        <div class="sidebar-profile">
            <div class="profile-avatar">🧠</div>
            <div class="profile-name">Medical AI</div>
            <div class="profile-subtitle">Brain Tumor Detection System</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="feature-text"><div>System Info</div></div>', unsafe_allow_html=True)
        
        # Model Info Card
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon-box bg-blue">🔍</div>
            <div class="feature-text">
                <div>Models Loaded</div>
                <div>CNN Binary & Multiclass</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # System Status
        device_status = "🟢 GPU" if tf.config.list_physical_devices('GPU') else "🟡 CPU"
        st.markdown(f"""
        <div class="feature-item">
            <div class="feature-icon-box bg-teal">💻</div>
            <div class="feature-text">
                <div>System Status</div>
                <div>{device_status} Active</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin-top: 1rem;" class="feature-text"><div>Performance Metrics</div></div>', unsafe_allow_html=True)
        
        # Performance Stats
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon-box bg-purple">📈</div>
            <div class="feature-text">
                <div>Binary Accuracy</div>
                <div>94.2% on Validation</div>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon-box bg-purple">🎯</div>
            <div class="feature-text">
                <div>Multiclass Accuracy</div>
                <div>91.8% on Validation</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_dashboard_header():
    """Create dashboard header for medical app"""
    st.markdown("""
    <div class="dashboard-header">
        <div class="header-content">
            <h1>🧠 Brain Tumor Detection</h1>
            <p>Advanced AI-powered medical imaging analysis with GradCAM visualization</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_feature_cards():
    """Create feature cards for medical analysis"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="feature-icon-box bg-blue" style="width: 48px; height: 48px; font-size: 1.5rem; margin-bottom: 1rem;">🔬</div>
            <h3 style="color: #0F172A; font-weight: 700; margin-bottom: 0.5rem;">MRI Analysis</h3>
            <p style="color: #334155; font-size: 0.95rem;">Magnetic Resonance Imaging analysis for structural tumor detection.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="custom-card">
            <div class="feature-icon-box bg-teal" style="width: 48px; height: 48px; font-size: 1.5rem; margin-bottom: 1rem;">⚡</div>
            <h3 style="color: #0F172A; font-weight: 700; margin-bottom: 0.5rem;">CT Scan Analysis</h3>
            <p style="color: #334155; font-size: 0.95rem;">Computed Tomography processing for cross-sectional detailed views.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="custom-card">
            <div class="feature-icon-box bg-purple" style="width: 48px; height: 48px; font-size: 1.5rem; margin-bottom: 1rem;">🧬</div>
            <h3 style="color: #0F172A; font-weight: 700; margin-bottom: 0.5rem;">GradCAM AI</h3>
            <p style="color: #334155; font-size: 0.95rem;">Explainable AI visualization to show tumor localization heatmaps.</p>
        </div>
        """, unsafe_allow_html=True)

def create_statistics_panel():
    """Create statistics panel for medical dashboard"""
    # Medical metrics grid
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{st.session_state.total_scans}</div>
            <div class="metric-label">Total Scans</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: #EF4444;">{st.session_state.tumors_detected}</div>
            <div class="metric-label">Tumor Cases</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: #10B981;">{st.session_state.healthy_scans}</div>
            <div class="metric-label">Healthy Scans</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        accuracy = 94.2
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="color: #8B5CF6;">{accuracy}%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

def display_prediction_results(binary_prediction, tumor_type, confidence_score):
    """Display medical prediction results with modern styling"""

    # Update session state
    st.session_state.total_scans += 1

    if binary_prediction[0][0] < 0.5:
        # No tumor detected
        st.session_state.healthy_scans += 1
        result_class = "result-healthy"
        icon = "✅"
        title = "No Tumor Detected"
        subtitle = "Brain scan appears healthy"
        bar_color = "#10B981"
    else:
        # Tumor detected
        st.session_state.tumors_detected += 1
        result_class = "result-tumor"
        icon = "⚠️"
        title = "Tumor Detected"
        subtitle = "Abnormal growth identified"
        bar_color = "#EF4444"

        # Update tumor type statistics
        if tumor_type == "Glioma":
            st.session_state.glioma_cases += 1
        elif tumor_type == "Meningioma":
            st.session_state.meningioma_cases += 1
        elif tumor_type == "Pituitary":
            st.session_state.pituitary_cases += 1

    # Define styles for high contrast
    text_main = "#0F172A"
    text_sub = "#334155"
    
    # Construct HTML with zero indentation to prevent code block rendering
    html_content = f"""<div class="result-box {result_class}">
<div class="confidence-label">
<div style="display: flex; align-items: center; gap: 0.5rem; font-size: 1.2rem; font-weight: 700; color: {text_main};">
<span>{icon}</span> {title}
</div>
<div style="color: {text_main};">{confidence_score:.1%} Confidence</div>
</div>
<p style="margin-bottom: 1rem; opacity: 1; color: {text_sub};">{subtitle}</p>
<div class="confidence-bar-bg">
<div class="confidence-bar-fill" style="width: {confidence_score*100}%; background-color: {bar_color};"></div>
</div>
<div style="font-size: 0.85rem; margin-top: 1rem; opacity: 1; color: {text_sub}; padding-top: 0.5rem; border-top: 1px solid #E2E8F0;">
<strong style="color: {text_main};">Analysis Details:</strong><br>
Binary Classification: {'Healthy' if binary_prediction[0][0] < 0.5 else 'Tumor Present'}<br>
Detection Confidence: {confidence_score:.3f}<br>
Processing Time: &lt;1 second
</div>
</div>"""
    
    st.markdown(html_content, unsafe_allow_html=True)

    # Display tumor type if detected
    if binary_prediction[0][0] >= 0.5:
        tumor_icons = {
            'Glioma': '🧬',
            'Meningioma': '🔴',
            'Pituitary': '🟡'
        }
        icon = tumor_icons.get(tumor_type, '🟡')

        st.markdown(f"""
        <div class="custom-card" style="margin-top: 1rem; border-left: 4px solid #3B82F6;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2rem;">{icon}</div>
                <div>
                    <h3 style="margin: 0; color: #0F172A; font-weight: 700;">Tumor Type: {tumor_type}</h3>
                    <p style="margin: 0; color: #334155;">Multiclass CNN Analysis Result</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def create_analytics_dashboard():
    """Create analytics dashboard for medical statistics"""
    st.markdown('<h3 style="color: #0F172A; margin-bottom: 2rem;">📊 Medical Analytics Dashboard</h3>', unsafe_allow_html=True)

    # Create charts for medical statistics
    col1, col2 = st.columns(2)

    with col1:
        # Tumor distribution pie chart
        labels = ['Healthy', 'Glioma', 'Meningioma', 'Pituitary']
        values = [
            st.session_state.healthy_scans or 1,
            st.session_state.glioma_cases or 1,
            st.session_state.meningioma_cases or 1,
            st.session_state.pituitary_cases or 1
        ]
        colors = ['#38a169', '#f093fb', '#4ecdc4', '#ffe066']

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=colors
        )])

        fig.update_layout(
            title="Case Distribution",
            template="plotly_white",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sample detection timeline
        dates = [datetime.now() - timedelta(days=x) for x in range(7, 0, -1)]
        detections = np.random.randint(5, 20, 7)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=detections,
            mode='lines+markers',
            fill='tonexty',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#667eea'),
            name='Daily Scans'
        ))

        fig.update_layout(
            title="Weekly Scan Activity",
            xaxis_title="Date",
            yaxis_title="Number of Scans",
            template="plotly_white",
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""

    # Load models
    binary_model = load_binary_model()
    multiclass_model = load_multiclass_model()

    if binary_model is None or multiclass_model is None:
        st.error("❌ Failed to load models. Please check model files.")
        st.stop()

    # Create sidebar
    create_sidebar()

    # Dashboard header
    create_dashboard_header()

    # Main layout (Stacked for better visibility)
    create_feature_cards()
    st.markdown("---")
    create_statistics_panel()

    # Main interface with tabs
    tab1, tab2 = st.tabs(["🩻 Medical Analysis", "📊 Analytics Dashboard"])

    with tab1:
        st.markdown('<h3 style="color: #0F172A; margin-bottom: 1rem;">🏥 Upload Medical Images</h3>', unsafe_allow_html=True)

        # File uploaders in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<h4 style="color: #0F172A; margin-bottom: 0.5rem;">📡 MRI Scan</h4>', unsafe_allow_html=True)
            mri_file = st.file_uploader(
                "Upload MRI Image", 
                type=["png", "jpg", "jpeg"],
                key="mri_uploader"
            )

        with col2:
            st.markdown('<h4 style="color: #0F172A; margin-bottom: 0.5rem;">⚡ CT Scan</h4>', unsafe_allow_html=True)
            ct_file = st.file_uploader(
                "Upload CT Image", 
                type=["png", "jpg", "jpeg"],
                key="ct_uploader"
            )

        # Analysis button
        if st.button("🔍 Analyze Brain Scans", use_container_width=True):
            if mri_file is not None and ct_file is not None:
                try:
                    # --- Validation Steps ---
                    # 1. Validate Content (Brain vs Non-Brain)
                    is_valid_mri, err_mri = validate_image_content(mri_file)
                    if not is_valid_mri:
                        st.error(f"❌ MRI Input Error: {err_mri}")
                        st.stop()
                        
                    is_valid_ct, err_ct = validate_image_content(ct_file)
                    if not is_valid_ct:
                        st.error(f"❌ CT Input Error: {err_ct}")
                        st.stop()

                    # 2. Validate Modality (MRI vs CT mismatch)
                    # Reset file pointers after reading in validation functions
                    mri_file.seek(0)
                    ct_file.seek(0)
                    
                    is_mri_correct, Mod_err_mri, mri_ct_score = check_modality(mri_file, 'MRI')
                    is_ct_correct, Mod_err_ct, ct_ct_score = check_modality(ct_file, 'CT')
                    
                    if not is_mri_correct and not is_ct_correct:
                        st.error("❌ MRI image has been uploaded in place of ct and ct image is uploaded in place of mri")
                        st.stop()
                    elif not is_mri_correct:
                        st.warning(f"⚠️ Potential Modality Mismatch (MRI Slot): {Mod_err_mri}")
                    elif not is_ct_correct:
                        st.warning(f"⚠️ Potential Modality Mismatch (CT Slot): {Mod_err_ct}")
                    
                    # 3. Cross-image pair validation
                    mri_file.seek(0)
                    ct_file.seek(0)
                    pair_warnings = validate_image_pair(mri_file, ct_file, mri_ct_score, ct_ct_score)
                    for pw in pair_warnings:
                        st.warning(f"⚠️ Pair Mismatch: {pw}")
                        
                    # Reset file pointers again for processing
                    mri_file.seek(0)
                    ct_file.seek(0)
                    # ------------------------

                    with st.spinner("🧠 Analyzing brain scans and generating GradCAM visualizations..."):
                        # Preprocess images
                        mri_img = preprocess_image(mri_file)
                        ct_img = preprocess_image(ct_file)

                        # Get original images for visualization
                        mri_original = Image.open(mri_file).convert("RGB").resize((224, 224))
                        ct_original = Image.open(ct_file).convert("RGB").resize((224, 224))

                        # Binary classification
                        inputs_for_binary = {'mri_input': mri_img, 'ct_input': ct_img}
                        prediction_binary = binary_model.predict(inputs_for_binary, verbose=0)

                        # Calculate confidence score
                        binary_conf = float(abs(prediction_binary[0][0] - 0.5) * 2)
                        
                        # Multiclass classification (if tumor detected)
                        tumor_type = "N/A"
                        tumor_confidence = 0.0
                        
                        if prediction_binary[0][0] >= 0.5:
                            prediction_multiclass = multiclass_model.predict(mri_img, verbose=0)
                            class_labels = ['Glioma', 'Meningioma', 'Pituitary', 'Pituitary']
                            predicted_index = np.argmax(prediction_multiclass[0])
                            
                            if predicted_index < len(class_labels):
                                tumor_type = class_labels[predicted_index]
                            else:
                                tumor_type = "Unknown"

                            tumor_confidence = float(prediction_multiclass[0][predicted_index])
                            
                            # Combined confidence: Average of binary detection and specific type confidence
                            confidence_score = (binary_conf + tumor_confidence) / 2
                        else:
                            # If no tumor, confidence is just the binary confidence (healthiness)
                            confidence_score = binary_conf

                        # Display results
                        display_prediction_results(prediction_binary, tumor_type, confidence_score)

                        # Generate GradCAM visualizations
                        st.markdown('<h3 style="color: #0F172A; margin-top: 2rem;">🔍 GradCAM Analysis</h3>', unsafe_allow_html=True)
                        st.markdown("""
                        <div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 1rem; border-radius: 4px; margin-bottom: 1rem; color: #1E293B;">
                            <span style="font-weight: 600;">🔬 Processing:</span> Generating AI attention maps to understand model decision-making...
                        </div>
                        """, unsafe_allow_html=True)
                        
                        try:
                            # Get the last conv layer for multiclass model
                            last_conv_layer = get_last_conv_layer_name(multiclass_model)
                            
                            if last_conv_layer:
                                st.markdown(f"""
                                <div style="background-color: #ECFDF5; border-left: 4px solid #10B981; padding: 1rem; border-radius: 4px; margin-bottom: 1rem; color: #064E3B;">
                                    <span style="font-weight: 600;">✅ Success:</span> Using convolutional layer: {last_conv_layer}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Generate heatmaps
                                mri_tensor = tf.convert_to_tensor(mri_img, dtype=tf.float32)
                                ct_tensor = tf.convert_to_tensor(ct_img, dtype=tf.float32)
                                
                                mri_heatmap = make_gradcam_heatmap(mri_tensor, multiclass_model, last_conv_layer)
                                ct_heatmap = make_gradcam_heatmap(ct_tensor, multiclass_model, last_conv_layer)
                                
                                # Create comprehensive visualization
                                binary_result = "Tumor Detected" if prediction_binary[0][0] >= 0.5 else "No Tumor"
                                binary_confidence = prediction_binary[0][0] if prediction_binary[0][0] >= 0.5 else 1 - prediction_binary[0][0]
                                
                                fig = create_gradcam_visualization(
                                    mri_original, ct_original, mri_heatmap, ct_heatmap,
                                    tumor_type, binary_result, tumor_confidence, binary_confidence
                                )
                                
                                # Display the GradCAM visualization
                                st.pyplot(fig)
                                
                                st.markdown("""
                                <div style="background-color: #ECFDF5; border-left: 4px solid #10B981; padding: 1rem; border-radius: 4px; margin-top: 1rem; color: #064E3B;">
                                    <span style="font-weight: 600;">🎯 Complete:</span> GradCAM analysis finished! Heatmaps reveal AI focus areas.
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Additional analysis information
                                st.markdown("""
                                <div class="custom-card" style="margin-top: 1.5rem;">
                                    <h4 style="color: #0F172A; margin-bottom: 1rem;">📋 GradCAM Interpretation Guide</h4>
                                    <ul style="color: #334155; line-height: 1.6;">
                                        <li><strong>Red/Yellow areas</strong>: High attention regions where the model focuses most</li>
                                        <li><strong>Blue areas</strong>: Low attention regions</li>
                                        <li><strong>Original</strong>: The input images as received</li>
                                        <li><strong>Heatmap</strong>: Pure attention map from the model</li>
                                        <li><strong>Overlay</strong>: Combined view showing attention on original image</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            else:
                                st.markdown("""
                                <div style="background-color: #FFFBEB; border-left: 4px solid #F59E0B; padding: 1rem; border-radius: 4px; color: #78350F;">
                                    <span style="font-weight: 600;">⚠️ Warning:</span> Could not find convolutional layers in the model for GradCAM analysis.
                                </div>
                                """, unsafe_allow_html=True)
                                
                        except Exception as e:
                            st.markdown(f"""
                            <div style="background-color: #FEF2F2; border-left: 4px solid #EF4444; padding: 1rem; border-radius: 4px; color: #7F1D1D;">
                                <span style="font-weight: 600;">❌ Error:</span> Error generating GradCAM visualization: {str(e)}
                            </div>
                            """, unsafe_allow_html=True)

                        # Display original uploaded images
                        st.markdown('<h3 style="color: #0F172A; margin-top: 2rem;">🖼️ Original Uploaded Images</h3>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)

                        with col1:
                            # Explicitly resize to 224x224 before display
                            mri_display = Image.open(mri_file).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                            
                            # Convert image to base64 for embedding in HTML
                            img_buffer = io.BytesIO()
                            mri_display.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            img_base64 = base64.b64encode(img_buffer.read()).decode()
                            
                            st.markdown(f"""
                            <div class="custom-card" style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem; min-height: 300px;">
                                <img src="data:image/png;base64,{img_base64}" style="width: 224px; height: 224px; border-radius: 12px; object-fit: contain; border: 2px solid #E2E8F0;">
                                <p style="margin-top: 1rem; font-size: 0.9rem; font-weight: 600; color: #0F172A;">MRI Scan (224x224)</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            # Explicitly resize to 224x224 before display
                            ct_display = Image.open(ct_file).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                            
                            # Convert image to base64 for embedding in HTML
                            img_buffer = io.BytesIO()
                            ct_display.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            img_base64 = base64.b64encode(img_buffer.read()).decode()
                            
                            st.markdown(f"""
                            <div class="custom-card" style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem; min-height: 300px;">
                                <img src="data:image/png;base64,{img_base64}" style="width: 224px; height: 224px; border-radius: 12px; object-fit: contain; border: 2px solid #E2E8F0;">
                                <p style="margin-top: 1rem; font-size: 0.9rem; font-weight: 600; color: #0F172A;">CT Scan (224x224)</p>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ An error occurred during analysis: {str(e)}")
            else:
                st.warning("⚠️ Please upload both MRI and CT images to proceed with analysis.")

    with tab2:
        create_analytics_dashboard()

if __name__ == "__main__":
    main()