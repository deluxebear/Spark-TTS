# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import soundfile as sf
import logging
import gradio as gr
import platform

from datetime import datetime
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI

# Colab-specific configurations
IN_COLAB = 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False

def setup_colab_environment():
    """Setup environment for Google Colab"""
    if IN_COLAB:
        print("🔧 Setting up Google Colab environment...")
        
        # Install required packages
        print("📦 Installing required packages...")
        os.system("pip install gradio soundfile torch torchaudio")
        
        # Setup directories
        os.makedirs("/content/pretrained_models", exist_ok=True)
        os.makedirs("/content/example/results", exist_ok=True)
        
        # Set up ngrok for public access (optional)
        try:
            os.system("pip install pyngrok")
            from pyngrok import ngrok
            print("🌐 Ngrok installed for public access")
        except:
            print("⚠️ Ngrok installation failed, using Gradio share instead")
        
        print("✅ Colab environment setup completed!")

def get_device():
    """Get appropriate device for Colab"""
    if IN_COLAB:
        # In Colab, prefer CUDA if available, otherwise use CPU
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("⚠️ GPU not available, using CPU")
    else:
        # Local environment logic
        if platform.system() == "Darwin":
            device = torch.device("mps:0")
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    
    return device

def download_model_if_needed(model_dir="pretrained_models/Spark-TTS-0.5B"):
    """Download model if not exists (for Colab)"""
    if IN_COLAB and not os.path.exists(model_dir):
        print("📥 Downloading Spark-TTS model...")
        # Add your model download logic here
        # For example:
        # os.system(f"wget -O {model_dir}.tar.gz 'YOUR_MODEL_URL'")
        # os.system(f"tar -xzf {model_dir}.tar.gz")
        print("⚠️ Please upload your model to the pretrained_models directory")
        return False
    return True

def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B"):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")
    
    # Check if model exists
    if not download_model_if_needed(model_dir):
        raise FileNotFoundError(f"Model not found at {model_dir}")
    
    device = get_device()
    logging.info(f"Using device: {device}")
    
    model = SparkTTS(model_dir, device)
    return model

def run_tts(
    text,
    model,
    prompt_text=None,
    prompt_speech=None,
    gender=None,
    pitch=None,
    speed=None,
    save_dir="/content/example/results" if IN_COLAB else "example/results",
):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Saving audio to: {save_dir}")

    if prompt_text is not None:
        prompt_text = None if len(prompt_text) <= 1 else prompt_text

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")

    # Perform inference and save the output audio
    with torch.no_grad():
        wav = model.inference(
            text,
            prompt_speech,
            prompt_text,
            gender,
            pitch,
            speed,
        )

        sf.write(save_path, wav, samplerate=16000)

    logging.info(f"Audio saved at: {save_path}")
    return save_path

def build_ui(model_dir="pretrained_models/Spark-TTS-0.5B"):
    """Build Gradio interface optimized for Colab"""
    
    # Initialize model
    try:
        model = initialize_model(model_dir)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    # Define callback function for voice cloning
    def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record):
        """Gradio callback to clone voice using text and optional prompt speech."""
        if not text.strip():
            return None, "Please enter text to synthesize"
        
        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        prompt_text_clean = None if len(prompt_text) < 2 else prompt_text

        try:
            audio_output_path = run_tts(
                text,
                model,
                prompt_text=prompt_text_clean,
                prompt_speech=prompt_speech
            )
            return audio_output_path, "✅ Audio generated successfully!"
        except Exception as e:
            return None, f"❌ Error: {str(e)}"

    # Define callback function for creating new voices
    def voice_creation(text, gender, pitch, speed):
        """Gradio callback to create a synthetic voice with adjustable parameters."""
        if not text.strip():
            return None, "Please enter text to synthesize"
        
        try:
            pitch_val = LEVELS_MAP_UI[int(pitch)]
            speed_val = LEVELS_MAP_UI[int(speed)]
            audio_output_path = run_tts(
                text,
                model,
                gender=gender,
                pitch=pitch_val,
                speed=speed_val
            )
            return audio_output_path, "✅ Audio generated successfully!"
        except Exception as e:
            return None, f"❌ Error: {str(e)}"

    # Custom CSS for better mobile experience
    css = """
    .gradio-container {
        max-width: 100% !important;
        padding: 10px !important;
    }
    .tab-nav {
        margin-bottom: 10px;
    }
    """

    with gr.Blocks(css=css, title="Spark-TTS") as demo:
        # Header with info for Colab users
        gr.HTML('''
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🎤 Spark-TTS by SparkAudio</h1>
            <p>High-quality Text-to-Speech synthesis with voice cloning capabilities</p>
        </div>
        ''')
        
        if IN_COLAB:
            gr.Markdown("""
            ### 📋 Colab Usage Instructions:
            1. Make sure your model files are uploaded to `/content/pretrained_models/Spark-TTS-0.5B/`
            2. Use the interface below to generate speech
            3. Generated audio files will be saved to `/content/example/results/`
            """)

        with gr.Tabs():
            # Voice Clone Tab
            with gr.TabItem("🎯 Voice Clone"):
                gr.Markdown("### Upload reference audio or record your voice")

                with gr.Row():
                    with gr.Column():
                        prompt_wav_upload = gr.Audio(
                            sources=["upload"],
                            type="filepath",
                            label="📁 Upload Reference Audio (16kHz+)",
                        )
                    with gr.Column():
                        prompt_wav_record = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="🎙️ Record Reference Audio",
                        )

                with gr.Row():
                    text_input = gr.Textbox(
                        label="📝 Text to Synthesize", 
                        lines=3, 
                        placeholder="Enter the text you want to convert to speech...",
                        value="Hello, this is a test of voice cloning technology."
                    )
                    prompt_text_input = gr.Textbox(
                        label="📄 Reference Text (Optional)",
                        lines=3,
                        placeholder="Enter the text of the reference audio (recommended for same language)...",
                    )

                generate_button_clone = gr.Button("🚀 Generate Speech", variant="primary")
                
                with gr.Row():
                    audio_output_clone = gr.Audio(label="🔊 Generated Audio")
                    status_clone = gr.Textbox(label="Status", interactive=False)

                generate_button_clone.click(
                    voice_clone,
                    inputs=[text_input, prompt_text_input, prompt_wav_upload, prompt_wav_record],
                    outputs=[audio_output_clone, status_clone],
                )

            # Voice Creation Tab
            with gr.TabItem("🎨 Voice Creation"):
                gr.Markdown("### Create synthetic voice with custom parameters")

                with gr.Row():
                    with gr.Column():
                        gender = gr.Radio(
                            choices=["male", "female"], 
                            value="male", 
                            label="👤 Gender"
                        )
                        pitch = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, 
                            label="🎵 Pitch Level"
                        )
                        speed = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, 
                            label="⚡ Speed Level"
                        )
                    
                    with gr.Column():
                        text_input_creation = gr.Textbox(
                            label="📝 Input Text",
                            lines=4,
                            placeholder="Enter text here...",
                            value="You can generate a customized voice by adjusting parameters such as pitch and speed.",
                        )
                        create_button = gr.Button("🎯 Create Voice", variant="primary")

                with gr.Row():
                    audio_output_creation = gr.Audio(label="🔊 Generated Audio")
                    status_creation = gr.Textbox(label="Status", interactive=False)

                create_button.click(
                    voice_creation,
                    inputs=[text_input_creation, gender, pitch, speed],
                    outputs=[audio_output_creation, status_creation],
                )

        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center;">
            <p>Powered by <strong>Spark-TTS</strong> | Optimized for Google Colab</p>
        </div>
        """)

    return demo

def launch_colab():
    """Launch function specifically for Colab"""
    # Setup environment
    if IN_COLAB:
        setup_colab_environment()
    
    # Build UI
    demo = build_ui()
    if demo is None:
        print("❌ Failed to build UI due to model loading issues")
        return
    
    # Launch with appropriate settings for Colab
    if IN_COLAB:
        # In Colab, use share=True for public access
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            quiet=False
        )
    else:
        # Local launch
        demo.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860
        )

# Main execution
if __name__ == "__main__":
    launch_colab()
