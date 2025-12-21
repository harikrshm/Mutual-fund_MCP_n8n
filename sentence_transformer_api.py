"""
Sentence Transformer API Server for n8n Integration
Runs locally on http://localhost:8000
No API keys required - uses local Sentence Transformers model
"""

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os
import sys

app = Flask(__name__)

# Load the model once at startup
print("=" * 60)
print("Loading Sentence Transformer model: all-MiniLM-L6-v2")
print("=" * 60)
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded successfully!")
    print(f"✓ Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print("=" * 60)
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": "all-MiniLM-L6-v2",
        "dimension": model.get_sentence_embedding_dimension()
    })

@app.route('/embed', methods=['POST'])
def embed():
    """
    Generate embeddings for text
    
    Expected JSON body:
    {
        "text": "Your text here"
    }
    
    Returns:
    {
        "embedding": [0.1, 0.2, ...],
        "dimension": 384,
        "model": "all-MiniLM-L6-v2"
    }
    """
    try:
        # Get JSON body
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "Empty request body"
            }), 400
        
        # Check for 'text' field (also accept 'query' for compatibility)
        text = data.get('text') or data.get('query')
        
        if not text:
            return jsonify({
                "error": "Missing 'text' or 'query' field in request body"
            }), 400
        
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                "error": "Text must be a non-empty string"
            }), 400
        
        # Generate embedding
        print(f"Generating embedding for: {text[:50]}...")
        embedding = model.encode(text, convert_to_numpy=True).tolist()
        
        return jsonify({
            "embedding": embedding,
            "dimension": len(embedding),
            "model": "all-MiniLM-L6-v2",
            "text": text  # Echo back the text for verification
        }), 200
        
    except Exception as e:
        print(f"Error in /embed: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/embed/batch', methods=['POST'])
def embed_batch():
    """
    Generate embeddings for multiple texts
    
    Expected JSON body:
    {
        "texts": ["text1", "text2", ...]
    }
    
    Returns:
    {
        "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
        "count": 2,
        "dimension": 384,
        "model": "all-MiniLM-L6-v2"
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400
        
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "error": "Missing 'texts' field in request body"
            }), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                "error": "texts must be a non-empty array"
            }), 400
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        
        return jsonify({
            "embeddings": embeddings,
            "count": len(embeddings),
            "dimension": len(embeddings[0]) if embeddings else 0,
            "model": "all-MiniLM-L6-v2"
        }), 200
        
    except Exception as e:
        print(f"Error in /embed/batch: {e}")
        return jsonify({
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/health", "/embed", "/embed/batch"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("=" * 60)
    print(f"Starting Sentence Transformer API server")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Health check: http://{host}:{port}/health")
    print(f"Embed endpoint: http://{host}:{port}/embed")
    print("=" * 60)
    
    app.run(host=host, port=port, debug=False)

