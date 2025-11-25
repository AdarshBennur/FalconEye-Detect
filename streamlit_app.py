"""
FalconEye-Detect Streamlit Web Application
Interactive web application for bird vs drone classification and detection.
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Import our inference utilities
import sys
from pathlib import Path as PathObj

# Add scripts directory to path dynamically
scripts_dir = PathObj(__file__).parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from inference_utils import ModelInference, StreamlitInferenceUtils

# Page configuration
st.set_page_config(
    page_title="FalconEye-Detect",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #4169E1);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .prediction-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4169E1;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #2E8B57, #4169E1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

class FalconEyeApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        # Initialize instance variables for model tracking
        self.inference = None
        self.streamlit_utils = None
        self.classification_count = 0
        self.detection_count = 0
        self.model_info = None
        
        # Initialize the application
        self.initialize_app()
    
    def load_models(self):
        """Load all available models and store counts"""
        try:
            self.inference = ModelInference()
            self.inference.load_all_available_models()
            
            # Store model counts as instance variables
            self.classification_count = len(self.inference.classification_models)
            self.detection_count = len(self.inference.detection_models)
            
            # Store model information
            self.model_info = self.inference.get_model_info()
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def validate_models(self):
        """Validate that at least one model is loaded"""
        total_models = self.classification_count + self.detection_count
        
        if total_models > 0:
            return True
        else:
            st.session_state.error_message = "No trained models found. Please train models first."
            return False
    
    def initialize_app(self):
        """Initialize the application and load models"""
        
        # Ensure weights are available (for Streamlit Cloud deployment)
        if 'weights_fetched' not in st.session_state:
            try:
                import subprocess
                import sys
                from pathlib import Path
                
                # Use sync_weights.py for better checksum verification
                sync_script = Path(__file__).parent / "scripts" / "sync_weights.py"
                if sync_script.exists():
                    # Show progress to user
                    with st.spinner("Synchronizing model weights..."):
                        # Run weight synchronization
                        result = subprocess.run(
                            [sys.executable, str(sync_script)],
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 min timeout
                        )
                        
                        if result.returncode != 0:
                            st.warning(f"Weight sync note: {result.stdout}")
                        else:
                            # Success message
                            if "Downloaded:" in result.stdout:
                                st.success("✅ Model weights synchronized")
                
                st.session_state.weights_fetched = True
            except Exception as e:
                st.warning(f"Could not run weight synchronization: {e}")
                st.session_state.weights_fetched = True  # Continue anyway
        
        if 'models_loaded' not in st.session_state:
            with st.spinner("Loading models..."):
                # Load models using new method
                if self.load_models():
                    # Validate models
                    if self.validate_models():
                        self.streamlit_utils = StreamlitInferenceUtils(self.inference)
                        
                        # Store in session state for persistence
                        st.session_state.models_loaded = True
                        st.session_state.inference = self.inference
                        st.session_state.streamlit_utils = self.streamlit_utils
                        st.session_state.model_info = self.model_info
                        st.session_state.classification_count = self.classification_count
                        st.session_state.detection_count = self.detection_count
                        
                        # Store loaded model names for debugging
                        st.session_state.loaded_classification_models = list(self.inference.classification_models.keys())
                        st.session_state.loaded_detection_models = list(self.inference.detection_models.keys())
                    else:
                        st.session_state.models_loaded = False
                else:
                    st.session_state.models_loaded = False
                    st.session_state.error_message = "Failed to load models."
        else:
            # Restore from session state
            self.inference = st.session_state.inference
            self.streamlit_utils = st.session_state.streamlit_utils
            self.model_info = st.session_state.model_info
            self.classification_count = st.session_state.classification_count
            self.detection_count = st.session_state.detection_count
    
    def render_header(self):
        """Render application header"""
        
        st.markdown("""
        <div class="main-header">
            <h1>🦅 FalconEye-Detect 🚁</h1>
            <p>AI-Powered Aerial Object Classification & Detection System</p>
            <p>Distinguish between Birds and Drones using Deep Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with controls and information"""
        
        st.sidebar.title("🔧 Controls")
        
        if not st.session_state.models_loaded:
            st.sidebar.error("⚠️ No models available")
            st.sidebar.info("Please train models using the training scripts first.")
            return None
        
        model_info = st.session_state.model_info
        
        # Model selection
        st.sidebar.subheader("📊 Model Selection")
        
        # Classification model selection
        if model_info['classification_models']:
            classification_model = st.sidebar.selectbox(
                "Classification Model",
                options=['Auto'] + model_info['classification_models'],
                help="Choose a classification model for bird vs drone prediction"
            )
            classification_model = None if classification_model == 'Auto' else classification_model
        else:
            classification_model = None
            st.sidebar.warning("No classification models available")
        
        # Detection model selection
        if model_info['detection_models']:
            detection_model = st.sidebar.selectbox(
                "Detection Model", 
                options=['Auto'] + model_info['detection_models'],
                help="Choose a detection model for object localization"
            )
            detection_model = None if detection_model == 'Auto' else detection_model
        else:
            detection_model = None
            st.sidebar.warning("No detection models available")
        
        # Task selection - dynamically determine available options
        st.sidebar.subheader("🎯 Task Selection")
        
        # Build task options based on available models
        task_options = []
        if self.classification_count > 0:
            task_options.append('Classification')
        if self.detection_count > 0:
            task_options.append('Detection')
        if self.classification_count > 0 and self.detection_count > 0:
            task_options.append('Both')
        
        # Set default task
        default_task = task_options[0] if task_options else 'Classification'
        
        # Show task selection
        task = st.sidebar.radio(
            "Select Task",
            options=task_options,
            index=0,
            help="Choose what type of analysis to perform"
        )
        
        # Detection settings - only show if detection models are available
        if task in ['Detection', 'Both'] and self.detection_count > 0:
            st.sidebar.subheader("⚙️ Detection Settings")
            confidence_threshold = st.sidebar.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Minimum confidence for object detection"
            )
        else:
            confidence_threshold = 0.5
        
        # Model information
        st.sidebar.subheader("📋 Model Information")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("Classification Models", len(model_info['classification_models']))
        
        with col2:
            st.metric("Detection Models", len(model_info['detection_models']))
        
        st.sidebar.info(f"**Classes:** {', '.join(model_info['class_names'])}")
        
        # Debug: Show loaded models
        with st.sidebar.expander("🔍 Loaded Models (Debug)", expanded=False):
            if 'loaded_classification_models' in st.session_state:
                st.write("**Classification:**")
                for model in st.session_state.loaded_classification_models:
                    st.text(f"✓ {model}")
            
            if 'loaded_detection_models' in st.session_state:
                st.write("**Detection:**")
                for model in st.session_state.loaded_detection_models:
                    st.text(f"✓ {model}")
        
        return {
            'classification_model': classification_model,
            'detection_model': detection_model,
            'task': task.lower(),
            'confidence_threshold': confidence_threshold
        }
    
    def render_image_upload(self):
        """Render image upload section"""
        
        st.subheader("📸 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPG, JPEG, or PNG image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.markdown("### Image Info")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**Size:** {image.size}")
                st.write(f"**Mode:** {image.mode}")
                st.write(f"**Format:** {image.format}")
        
        return uploaded_file
    
    def render_sample_images(self):
        """Render sample images section"""
        
        st.subheader("🖼️ Try Sample Images")
        
        # Sample images paths - get project root dynamically
        project_root = PathObj(__file__).parent
        sample_base_path = project_root / "data"
        classification_samples = sample_base_path / "classification_dataset" / "test"
        detection_samples = sample_base_path / "object_detection_dataset" / "test" / "images"
        
        sample_options = []
        sample_paths = {}
        
        # Add classification samples
        if classification_samples.exists():
            bird_samples = list((classification_samples / "bird").glob("*.jpg"))[:3]
            drone_samples = list((classification_samples / "drone").glob("*.jpg"))[:3]
            
            for i, img_path in enumerate(bird_samples):
                key = f"Sample Bird {i+1}"
                sample_options.append(key)
                sample_paths[key] = img_path
            
            for i, img_path in enumerate(drone_samples):
                key = f"Sample Drone {i+1}"
                sample_options.append(key)
                sample_paths[key] = img_path
        
        # Add detection samples
        if detection_samples.exists():
            det_samples = list(detection_samples.glob("*.jpg"))[:3]
            for i, img_path in enumerate(det_samples):
                key = f"Detection Sample {i+1}"
                sample_options.append(key)
                sample_paths[key] = img_path
        
        if sample_options:
            selected_sample = st.selectbox(
                "Select a sample image",
                options=["None"] + sample_options
            )
            
            if selected_sample != "None":
                sample_path = sample_paths[selected_sample]
                image = Image.open(sample_path)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(image, caption=f"{selected_sample}", use_column_width=True)
                
                with col2:
                    st.markdown("### Sample Info")
                    st.write(f"**Type:** {selected_sample}")
                    st.write(f"**Size:** {image.size}")
                    st.write(f"**Path:** {sample_path.name}")
                
                return image, str(sample_path)
        else:
            st.info("No sample images found in the dataset.")
        
        return None, None
    
    def process_image(self, image_source, settings):
        """Process image with selected models and settings"""
        
        # Check if models are loaded
        if not st.session_state.models_loaded:
            st.error("❌ Models not loaded! Please ensure models are trained and available.")
            return None
        
        # Validate image input
        if image_source is None:
            st.warning("⚠️ No image provided. Please upload or select an image.")
            return None
        
        # Validate image type
        if not isinstance(image_source, (Image.Image, np.ndarray, str)):
            st.error("❌ Invalid image type. Expected PIL Image, numpy array, or file path.")
            return None
        
        # Check model availability for selected task
        task = settings.get('task', 'classification')
        if task in ['classification', 'both'] and self.classification_count == 0:
            st.error("❌ No classification models available. Please train a classification model first.")
            return None
        
        if task in ['detection', 'both'] and self.detection_count == 0:
            st.error("❌ No detection models available. Please train a detection model first.")
            return None
        
        with st.spinner("Processing image..."):
            try:
                # Convert PIL image to numpy array if needed
                if isinstance(image_source, Image.Image):
                    # Validate PIL Image
                    if image_source.size[0] == 0 or image_source.size[1] == 0:
                        st.error("❌ Invalid image dimensions. Image has zero width or height.")
                        return None
                    image_array = np.array(image_source)
                elif isinstance(image_source, str):
                    # Load from file path
                    if not os.path.exists(image_source):
                        st.error(f"❌ Image file not found: {image_source}")
                        return None
                    image = Image.open(image_source)
                    image_array = np.array(image)
                else:
                    image_array = image_source
                
                # Validate numpy array dimensions
                if image_array.size == 0:
                    st.error("❌ Invalid image. Image array is empty.")
                    return None
                
                # Make predictions
                results = self.inference.predict_and_visualize(
                    image_array,
                    task=task,
                    classification_model=settings.get('classification_model'),
                    detection_model=settings.get('detection_model'),
                    conf_threshold=settings.get('confidence_threshold', 0.5)
                )
                
                return results
                
            except FileNotFoundError as e:
                st.error(f"❌ File not found: {str(e)}")
                return None
            except ValueError as e:
                st.error(f"❌ Invalid image format: {str(e)}")
                return None
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.exception(e)  # Show full traceback for debugging
                return None
    
    def display_results(self, results, settings):
        """Display prediction results"""
        
        if not results:
            st.warning("No results to display")
            return
        
        st.subheader("🎯 Prediction Results")
        
        # Classification results
        if 'classification' in results and results['classification']:
            st.markdown("### 🔍 Classification Results")
            
            classification_result = results['classification']
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Main prediction
                predicted_class = classification_result['class_name']
                confidence = classification_result['confidence']
                
                if predicted_class == 'Bird':
                    st.success(f"🦅 **Predicted: {predicted_class}**")
                else:
                    st.info(f"🚁 **Predicted: {predicted_class}**")
                
                st.metric("Confidence", f"{confidence:.1%}")
                st.metric("Model Used", classification_result['model_name'])
            
            with col2:
                # Confidence chart
                probs = classification_result['probabilities']
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(probs.keys()),
                        y=list(probs.values()),
                        marker_color=['red', 'green'],
                        text=[f"{v:.3f}" for v in probs.values()],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title="Class Probabilities",
                    yaxis_title="Probability",
                    showlegend=False,
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Detection results
        if 'detection' in results and results['detection']:
            st.markdown("### 🎯 Detection Results")
            
            detection_result = results['detection']
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if 'annotated_image' in results:
                    st.image(
                        results['annotated_image'],
                        caption="Detection Results with Bounding Boxes",
                        use_column_width=True
                    )
            
            with col2:
                st.metric("Objects Detected", detection_result['num_detections'])
                st.metric("Model Used", detection_result['model_name'])
                
                if detection_result['detections']:
                    st.markdown("#### Detected Objects:")
                    
                    for i, detection in enumerate(detection_result['detections'], 1):
                        with st.expander(f"Object {i}: {detection['class_name']}"):
                            st.write(f"**Confidence:** {detection['confidence']:.3f}")
                            bbox = detection['bbox']
                            st.write(f"**Bounding Box:** ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")
                            st.write(f"**Class:** {detection['class_name']}")
        
        # Both classification and detection summary
        if settings['task'] == 'both' and 'classification' in results and 'detection' in results:
            st.markdown("### 📊 Combined Analysis Summary")
            
            summary_cols = st.columns(4)
            
            with summary_cols[0]:
                st.metric(
                    "Classification", 
                    results['classification']['class_name'] if results['classification'] else 'N/A'
                )
            
            with summary_cols[1]:
                st.metric(
                    "Confidence", 
                    f"{results['classification']['confidence']:.1%}" if results['classification'] else 'N/A'
                )
            
            with summary_cols[2]:
                st.metric(
                    "Objects Found", 
                    results['detection']['num_detections'] if results['detection'] else 0
                )
            
            with summary_cols[3]:
                agreement = "✅ Consistent" if (
                    results['classification'] and results['detection'] and 
                    results['detection']['num_detections'] > 0 and
                    any(det['class_name'] == results['classification']['class_name'] 
                        for det in results['detection']['detections'])
                ) else "⚠️ Check needed"
                
                st.metric("Analysis", agreement)
    
    def render_batch_processing(self):
        """Render batch processing section"""
        
        st.subheader("📦 Batch Processing")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images for batch processing",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Select multiple images to process them all at once"
        )
        
        if uploaded_files:
            st.write(f"**Selected {len(uploaded_files)} images**")
            
            if st.button("Process All Images"):
                # Validate that classification models are available
                if self.classification_count == 0:
                    st.error("❌ No classification models available for batch processing.")
                    return
                
                with st.spinner(f"Processing {len(uploaded_files)} images..."):
                    results = []
                    errors = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        status_text.text(f"Processing {uploaded_file.name}...")
                        
                        # Process image with error handling
                        try:
                            image = Image.open(uploaded_file)
                            
                            # Validate image
                            if image.size[0] == 0 or image.size[1] == 0:
                                errors.append({
                                    'filename': uploaded_file.name,
                                    'error': 'Invalid dimensions'
                                })
                                continue
                            
                            result = self.inference.predict_classification(np.array(image))
                            
                            if result:
                                results.append({
                                    'filename': uploaded_file.name,
                                    'predicted_class': result['class_name'],
                                    'confidence': result['confidence']
                                })
                            else:
                                errors.append({
                                    'filename': uploaded_file.name,
                                    'error': 'Prediction failed'
                                })
                        except Exception as e:
                            errors.append({
                                'filename': uploaded_file.name,
                                'error': str(e)
                            })
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    if results:
                        st.success(f"✅ Successfully processed {len(results)} out of {len(uploaded_files)} images")
                        
                        df = pd.DataFrame(results)
                        
                        st.subheader("Batch Processing Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Images", len(results))
                        
                        with col2:
                            bird_count = sum(1 for r in results if r['predicted_class'] == 'Bird')
                            st.metric("Birds Detected", bird_count)
                        
                        with col3:
                            drone_count = sum(1 for r in results if r['predicted_class'] == 'Drone')
                            st.metric("Drones Detected", drone_count)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name="falcon_eye_batch_results.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("⚠️ No images were successfully processed.")
                    
                    # Show errors if any
                    if errors:
                        with st.expander(f"⚠️ {len(errors)} image(s) failed to process"):
                            error_df = pd.DataFrame(errors)
                            st.dataframe(error_df, use_container_width=True)

    
    def render_about(self):
        """Render about section"""
        
        st.subheader("ℹ️ About FalconEye-Detect")
        
        st.markdown("""
        **FalconEye-Detect** is an AI-powered system for distinguishing between birds and drones in aerial imagery.
        
        ### 🎯 **Key Features:**
        - **Classification:** Binary classification (Bird vs Drone)
        - **Object Detection:** Localize objects with bounding boxes
        - **Multiple Models:** Custom CNN, Transfer Learning (ResNet50, MobileNetV2, EfficientNetB0), YOLOv8
        - **Real-time Processing:** Fast inference on uploaded images
        - **Batch Processing:** Handle multiple images at once
        
        ### 🏗️ **Architecture:**
        - **Custom CNN:** Built-from-scratch convolutional neural network
        - **Transfer Learning:** Pre-trained models fine-tuned for bird/drone classification
        - **YOLOv8:** State-of-the-art object detection for localization
        
        ### 📊 **Applications:**
        - Airport bird strike prevention
        - Wildlife protection monitoring  
        - Security and defense surveillance
        - Airspace safety management
        
        ### 🔧 **Technical Stack:**
        - **Framework:** PyTorch, Ultralytics YOLOv8
        - **Frontend:** Streamlit
        - **Image Processing:** OpenCV, PIL
        - **Visualization:** Plotly, Matplotlib
        """)
    
    def run(self):
        """Main application runner"""
        
        # Render header
        self.render_header()
        
        # Check if models are loaded
        if not st.session_state.models_loaded:
            st.error("❌ No trained models found!")
            st.info("Please train the models first using the training scripts in the `scripts/` folder.")
            st.code("""
            # Run these commands to train the models:
            cd /path/to/your/FalconEye-Detect/project
            python scripts/train_custom_cnn.py
            python scripts/train_transfer_learning.py
            python scripts/train_yolov8.py
            """)
            return
        
        # Render sidebar
        settings = self.render_sidebar()
        
        if not settings:
            return
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Single Image", "📦 Batch Processing", "🔍 Model Comparison", "ℹ️ About"])
        
        with tab1:
            # Image upload section
            uploaded_file = self.render_image_upload()
            
            # Sample images section
            if uploaded_file is None:
                sample_image, sample_path = self.render_sample_images()
                
                if sample_image:
                    if st.button("🚀 Analyze Sample Image"):
                        results = self.process_image(sample_image, settings)
                        if results:
                            self.display_results(results, settings)
            else:
                # Process uploaded image
                if st.button("🚀 Analyze Uploaded Image"):
                    image = Image.open(uploaded_file)
                    results = self.process_image(image, settings)
                    if results:
                        self.display_results(results, settings)
        
        with tab2:
            self.render_batch_processing()
        
        with tab3:
            st.subheader("📊 Model Comparison")
            
            # Load model summaries if available
            project_root = PathObj(__file__).parent
            results_path = project_root / "results"
            comparison_file = results_path / "model_comparison_report.json"
            
            if comparison_file.exists():
                try:
                    with open(comparison_file, 'r') as f:
                        comparison_data = json.load(f)
                    
                    if 'classification_comparison' in comparison_data:
                        st.markdown("### Classification Models Performance")
                        df = pd.DataFrame(comparison_data['classification_comparison'])
                        st.dataframe(df, use_container_width=True)
                        
                        # Performance chart
                        fig = px.bar(
                            df, 
                            x='Model', 
                            y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                            title="Classification Models Performance Comparison",
                            barmode='group'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if 'detection_comparison' in comparison_data:
                        st.markdown("### Detection Models Performance")
                        df_det = pd.DataFrame(comparison_data['detection_comparison'])
                        st.dataframe(df_det, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error loading comparison data: {str(e)}")
            else:
                st.info("No model comparison data available. Run the evaluation script first.")
                st.code("python scripts/model_evaluation.py")
        
        with tab4:
            self.render_about()

def main():
    """Main function to run the Streamlit app"""
    
    app = FalconEyeApp()
    app.run()

if __name__ == "__main__":
    main()
