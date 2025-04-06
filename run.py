from waitress import serve
from app import app  # Make sure 'app.py' has the Flask app named 'app'

if __name__ == '__main__':
    print("🚀 Starting FlickPick backend on http://localhost:5000")
    serve(app, host='0.0.0.0', port=5000)
