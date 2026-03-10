"""
CoverComposer - SIMPLEST VERSION
Just the essentials - no complexity
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
from datetime import datetime
import json
import torch
import torchaudio

# Create Flask app
app = Flask(__name__)
CORS(app)

# Create folder
os.makedirs('music_files', exist_ok=True)

# Store music info
music_list = []

# ============ LOAD AI MODEL ============

def load_model():
    """Load MusicGen model"""
    try:
        print("\n🎵 Loading AI model... (wait 1-2 minutes)")
        from audiocraft.models import MusicGen
        model = MusicGen.get_pretrained("facebook/musicgen-small", device="cpu")
        print("✅ AI Model Loaded!\n")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Make sure you installed: pip install audiocraft torch torchaudio")
        return None

MODEL = load_model()

# ============ ROUTES ============

@app.route('/')
def home():
    """Show website"""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate music from prompt"""
    try:
        data = request.json
        prompt = data.get('prompt')
        duration = int(data.get('duration', 10))
        
        print(f"\n📝 Request received: {prompt}")
        
        if not prompt:
            return jsonify({'success': False, 'error': 'Prompt required'}), 400
        
        if MODEL is None:
            return jsonify({'success': False, 'error': 'AI model not loaded. Try refreshing page.'}), 500
        
        # Create filename
        music_id = f"music_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        file_path = f"music_files/{music_id}.wav"
        
        print(f"🎵 Generating: {prompt}")
        
        # Generate music
        MODEL.set_generation_params(duration=duration)
        with torch.no_grad():
            waveform = MODEL.generate([prompt], progress=False)
        
        print(f"📊 Waveform shape: {waveform.shape}")
        
        # Save audio
        torchaudio.save(file_path, waveform.cpu(), 32000)
        
        print(f"✅ Saved: {file_path}\n")
        
        # Store info
        info = {
            'id': music_id,
            'prompt': prompt,
            'duration': duration,
            'created': datetime.now().isoformat(),
            'file': file_path
        }
        music_list.append(info)
        
        response = {
            'success': True,
            'id': music_id,
            'url': f'/api/download/{music_id}'
        }
        
        print(f"📤 Sending response: {response}")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/download/<music_id>')
def download(music_id):
    """Download music file"""
    try:
        file_path = f"music_files/{music_id}.wav"
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            file_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'{music_id}.wav'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/list')
def list_music():
    """Get all generated music"""
    try:
        return jsonify({'success': True, 'music': music_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Check if server is running"""
    return jsonify({
        'success': True,
        'status': 'running',
        'model': MODEL is not None
    })

# ============ RUN ============

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎵 CoverComposer Started")
    print("="*50)
    print("👉 Open: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000)