from flask import Flask
from App.utils import load_model, Model, download_model

app = Flask(__name__)

modelPath = download_model()

model = Model(load_model(modelPath))

from App import views



