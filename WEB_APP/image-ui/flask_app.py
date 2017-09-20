import os
import json
from collections import defaultdict
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from src.predict_dog import get_one_prediction_for_web
from src.predict_dog import get_recommendation_for_web
from src.util import get_toys_info
from src.util import get_random_dogToys
#import control
import src.image as image

app = Flask(__name__)


def allowed_file(filename):
    """Checks whether the file type is allowed.

    Returns
    -------
    bool: whether the type is right or not
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


@app.route("/")
def home_page():
    """Displays the HTML of the home page of the app.

    Returns
    -------
    str: the html content returned to be displayed.
    """
    return render_template('welcome.html')

@app.route("/submit")
def up_page():
    """Displays the HTML of the home page of the app.

    Returns
    -------
    str: the html content returned to be displayed.
    """
    return render_template('bea_upload_prompt.html')

@app.route("/checkbox", methods=["GET", "POST", "REQUEST"])
def check_page():
    """Displays the HTML of the home page of the app.

    Returns
    -------
    str: the html content returned to be displayed.
    """
    samples = get_random_dogToys(n=5)
    return render_template('bea_checkbox.html', samples=samples)


@app.route("/uploader", methods=["POST"])
def uploader_post():
    """Receives the POST request to upload an image,
    processes it in any way (VisualRecognition)
    and returns the labels and faces detected in the image.

    Returns
    -------
    str: the html content returned to be displayed.
    """
    if request.method == 'POST':
        f = request.files['file']
        sfname = 'uploads/'+str(secure_filename(f.filename))

        if os.path.exists(sfname):
            os.remove(sfname)

        f.save(sfname)
        res_pred, recomendations = get_recommendation_for_web("resource/output_graph.pb",
                                                        "resource/output_labels.txt",
                                                    sfname)
        result = {}
        result['img_url'] = sfname
        pred_str = []
        pred_icon = []
        img_path = "../static/icons/"
        for pred in res_pred['three_predictions'][0]:
            pred_str.append(' : '.join([pred[0], "{:.2f}".format(pred[1])]))
            pred_icon.append(img_path+'_'.join(pred[0].split())+'.jpg')
        result['labels'] = zip(pred_str, pred_icon) #['dog:0.5', 'cat:0.25', 'jeff:0.05']
        # uncomment below to activate image reduction
        result['toys'] = get_toys_info(recomendations)
        image.reduceImage(sfname, 256)
        return render_template('partials/bea_result.html', result=result)


@app.route('/uploads/<path:path>')
def serve_images(path):
    """Serves an image displayed within the HTML document.

    Returns
    -------
    str: the content to display in the <img> tag
    """
    return send_from_directory('uploads/', path)

@app.route("/checkbox_submit", methods=["POST"])
def checkbox_submit():
    if request.method == 'POST':
        f = request.json
    print f['data']
    #return render_template("under construction")
    return render_template('partials/checkbox.html')

def flask_app_launch(**kwargs):
    """Launches the flask application server.

    Keyword Arguments
    -----------------
    **host: the host to attach the web server (default 0.0.0.0).
    **port: the port to attach the web server (default 8080).
    **debug: whether to switch to debug mode or not.
    """
    host_ = kwargs.get('host', '0.0.0.0')
    port_ = kwargs.get('port', 8080)
    debug_ = kwargs.get('debug', True)

    app.config['UPLOAD_FOLDER'] = './uploads/'
    app.run(host=host_, port=port_, debug=debug_, threaded=True)


if __name__ == '__main__':
    print("ERROR: to launch this app, run server.py instead")
    #flask_app_launch()
