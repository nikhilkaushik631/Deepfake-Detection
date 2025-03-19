import streamlit as st
import cv2
import torch
import timm
import torch.nn as nn
import mediapipe as mp
from torchvision import transforms
from PIL import Image
import math
import tempfile
import os
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Define the model architecture
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, latent_dim=512, num_heads=2, num_layers=1, hidden_dim=256):
        super(TransformerEncoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.5,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        x = self.norm(x)
        return self.transformer_encoder(x)

class DeepfakeModel(nn.Module):
    def __init__(self, num_classes, latent_dim=512, num_heads=2, num_layers=1, hidden_dim=256, max_seq_len=20):
        super(DeepfakeModel, self).__init__()

        base_model = timm.create_model('efficientnet_b3', pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])

        feat_dim = 1536
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(feat_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        self.pos_encoder = PositionalEncoding(latent_dim, dropout=0.2, max_len=max_seq_len)

        self.transformer = TransformerEncoder(
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, return_features=False):
        batch_size, seq_length, c, h, w = x.shape

        # Subsample frames to reduce computation
        sample_rate = 2
        if seq_length > 10:
            x = x[:, ::sample_rate, :, :, :]
            seq_length = x.shape[1]

        x = x.view(batch_size * seq_length, c, h, w)
        x = self.feature_extractor(x)
        x = self.projection(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.pos_encoder(x)
        features = self.transformer(x)
        x = features.transpose(1, 2)  # [B, D, T]
        x = self.temporal_pool(x).squeeze(-1)  # [B, D]
        output = self.classifier(x)

        if return_features:
            return output, features

        return output

# Function to process video
@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DeepfakeModel(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model, device

def process_video(video_path, model, device, target_size=(336, 336)):
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.7
    )

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Process video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // 30)  # Get 30 frames

    face_tensors = []
    frame_indices = []
    frames_processed = 0
    
    progress_bar = st.progress(0)
    
    for frame_idx in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update progress
        progress_bar.progress(min(1.0, frame_idx / total_frames))

        # Detect and process face
        detection = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not detection.detections:
            continue

        # Get face bounding box
        bbox = detection.detections[0].location_data.relative_bounding_box
        h, w = frame.shape[:2]
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        # Crop and transform face
        face_crop = frame[y:y+height, x:x+width]
        if face_crop.size > 0:
            face_tensor = transform(Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)))
            face_tensors.append(face_tensor)
            frame_indices.append(frame_idx)
            frames_processed += 1

    cap.release()
    progress_bar.progress(1.0)

    # Make prediction
    if len(face_tensors) < 5:
        return "Unknown (insufficient faces)", 0.0

    # Prepare sequence
    if len(face_tensors) > 30:
        face_tensors = face_tensors[:30]
    while len(face_tensors) < 30:
        face_tensors.append(face_tensors[-1])

    with torch.no_grad():
        sequence = torch.stack(face_tensors).unsqueeze(0).to(device)
        output = model(sequence)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    return "FAKE" if pred_class == 1 else "REAL", confidence

# Streamlit app
# Create a global variable to store the model
model = None
device = None

def main():
    global model, device
    
    st.set_page_config(page_title="Deepfake Detection", page_icon="🔍", layout="wide")
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom right, #1a1a1a, #4a4a4a);
        }
        .css-1d391kg {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
        }
        .stVideo {
        width: 450px !important;
        margin: 0 auto;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: white;'>Deepfake Detection System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: justify; color: white;'>
    This application uses a deep learning model to detect deepfake videos. 
    Upload a video file and the system will analyze it to determine if it's real or fake. 
    The model is based on the EfficientNet architecture and uses a transformer to process the video frames.
    The model was trained on the Celebrity Deepfake(V2) dataset and achieved an F1 score of 95% on the test set.
    </div>
    """, unsafe_allow_html=True)
    
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.device = None
        st.session_state.model_loaded = False
        st.session_state.model_load_success = False  

    # File uploader for model
    model_path = st.sidebar.file_uploader("Upload Model File (.pth)", type=["pth"])
    st.markdown("""
    <style>
        .css-1d391kg {
            background-color: grey;
        }
    </style>
    """, unsafe_allow_html=True)

    # Option to use default weights after the upload section
    if model_path is None:
        if st.sidebar.button("Use Default Model"):
            default_model_path = "C:\\Users\\nikhi\\Downloads\\b3_95.5f1.pth"  
            
            # Load the model
            try:
                st.session_state.model, st.session_state.device = load_model(default_model_path)
                st.session_state.model_loaded = True
                st.session_state.model_load_success = True 
            except Exception as e:
                st.sidebar.error(f"Error loading default model: {str(e)}")
    else:
        # Save the uploaded model file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_model:
            tmp_model.write(model_path.getvalue())
            model_path_local = tmp_model.name
        
        # Load the model
        try:
            st.session_state.model, st.session_state.device = load_model(model_path_local)
            st.session_state.model_loaded = True
            st.session_state.model_load_success = True  # Set UI flag
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")

    if st.session_state.model_load_success:
        st.sidebar.success("Model loaded successfully!")
    
    
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    
    if video_file is not None:
        st.video(video_file, format="video/mp4", start_time=0)
        
       # Process button
        if st.button("Process Video"):
            # Check if model is loaded
            if not st.session_state.model_loaded:
                st.error("No model loaded. Please upload a model file or use the default model.")
                return
                
            with st.spinner("Processing... This may take a while depending on the video length."):
                try:
                    # Save the uploaded video file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                        tmp_video.write(video_file.getvalue())
                        video_path_local = tmp_video.name
                        
                    classification, confidence = process_video(
                        video_path_local, 
                        st.session_state.model, 
                        st.session_state.device
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Classification", classification)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Style the result
                    if classification == "REAL":
                        st.success(f"This video appears to be REAL with {confidence:.2%} confidence.")
                    else:
                        st.error(f"This video appears to be FAKE with {confidence:.2%} confidence.")
                    
                    # Clean up temporary file
                    try:
                        os.unlink(video_path_local)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
        
        # Clean up temporary files
        try:
            os.unlink(video_path_local)
            os.unlink(model_path_local)
        except:
            pass
    else:
        st.info("Please upload a video file")
    


if __name__ == "__main__":
    main()
    
