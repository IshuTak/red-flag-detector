
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import sys
import logging
from datetime import datetime
import traceback


project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.flag_detector import FlagDetector

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize 
model = None

def init_model():
    """Initialize the FlagDetector model"""
    global model
    try:
        model_path = 'models/bert_model/saved_model/best_model.pth'
        model = FlagDetector(model_path=model_path)
        logger.info("Model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        logger.error(traceback.format_exc())
        return False

@app.route('/')
def home():
    """Render home page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analyze_message', methods=['POST'])
def analyze_message():
    """Analyze a single message"""
    try:
        # Get and validate input
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        
        if not model:
            if not init_model():
                return jsonify({'error': 'Model initialization failed'}), 500
        
        
        result = model.predict(message)
        
        
        logger.info(f"Message analyzed - Prediction: {result['prediction']}, "
                   f"Severity: {result.get('severity', 'N/A')}")
        
        
        response = {
            **result,
            'timestamp': datetime.now().isoformat(),
            'analysis_version': '2.0',
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error analyzing message: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Analysis failed',
            'details': str(e),
            'status': 'error'
        }), 500

@app.route('/analyze_conversation', methods=['POST'])
def analyze_conversation():
    """Analyze multiple messages"""
    try:
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        messages = data.get('messages', [])
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
        
        
        if not model:
            if not init_model():
                return jsonify({'error': 'Model initialization failed'}), 500
        
        
        result = model.analyze_messages(messages)
        
        
        logger.info(
            f"Conversation analyzed - "
            f"Total messages: {result['overall_assessment']['total_messages']}, "
            f"Red flags: {result['overall_assessment']['red_flags']}, "
            f"Severity: {result['overall_assessment']['severity']}"
        )
        
        
        response = {
            **result,
            'timestamp': datetime.now().isoformat(),
            'analysis_version': '2.0',
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error analyzing conversation: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Analysis failed',
            'details': str(e),
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Initialize 
    init_model()
    
    
    app.config.update(
        DEBUG=True,
        TEMPLATES_AUTO_RELOAD=True
    )
    
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
