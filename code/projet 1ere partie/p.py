from flask import Flask
from .p import app

app = Flask(__name__)

@app.route('/')
def index():
    return "MOULOUYA!"

if __name__ == "__main__":
    app.run()