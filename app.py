from flask import Flask
from backend.api.routes import api_bp

app = Flask(__name__)
app.register_blueprint(api_bp, url_prefix='/api')
app.run(host="0.0.0.0", port=8000)

if __name__ == '__main__':
    app.run(debug=True)
