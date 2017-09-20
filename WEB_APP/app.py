from flask import Flask, render_template
import predict_dog
from flask.json import jsonify
app = Flask(__name__)

@app.route('/')
def index():
    result = predict_dog.get_one_prediction_for_web("output_graph.pb",
                                                    "output_labels.txt",
                                                    "static/IMG_4342.JPG")
    return render_template('dog.html', predictions=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
