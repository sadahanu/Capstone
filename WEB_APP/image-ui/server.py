#import cf_deployment_tracker
import json
import flask_app
import os

# Emit Bluemix deployment event
#cf_deployment_tracker.track()


if __name__ == '__main__':
    # On Bluemix, get the port number from the environment variable PORT
    # When running this app on the local machine, default the port to 8080
    port = int(os.getenv('PORT', 8080))

    print("Launching Flask app on port {}".format(port))
    # launches the flask app server
    flask_app.flask_app_launch(port=port)
