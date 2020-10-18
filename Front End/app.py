from flask import Flask, request, render_template, redirect  # Blueprint
# from flask_cors import cross_origin
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import numpy as np
import base64
# import cv2
# import os

# app1 = Blueprint("app1", __name__, static_folder="static", template_folder="templates", url_prefix="/num_rec_model")
app = Flask(__name__)

model = load_model(r"D:\Software Development\Projects\Object Recognition (CNN on CIFAR-10)\CNN_on_CIFAR-10.h5")

@app.route("/")
# @cross_origin()
def home():
    return render_template("canvas.html")


@app.route("/predict",  methods = ["GET", "POST"])
# @cross_origin()
def image_view():
    if request.method == "POST":
        # Fetching the Canvas URL through POST Method
        data_url = request.files["link"]

        # Opening the image
        img = Image.open(data_url)

        # resize image and ignore original aspect ratio
        img_resized = img.resize((32,32))

        # convert image to numpy array
        img_array = np.asarray(img_resized)

        # Reshaping image numpy array as taken by the model
        img_reshape = img_array.reshape(1,32,32,3)


        # print(data_url)

        # ############################## Important Commands which can come in handy later (DO NOT DELETE) ################

        # # # To Save RGB covnverted Image
        # # rgb_im.save(r'C:\Users\iprak\Desktop\grayscale_converted.jpeg')
        # # # To convert image in Numpy Array
        # # image_final = np.array(image)
        # # # To print the shape of converted numpy array
        # # print(image_final.shape)
        # # # To show the image through opencv --- First parameter takes window as input
        # # # In our case its not there, thus its left blank
        # # cv2.imshow("", image_final)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()

        # ############################## ################################################ ###############################

        # Predicting the number
        output = model.predict_classes(img_reshape)
        print(output)

        # Replacing the output array index with the labels
        # Created dictionary of labels
        outcomes = {0: "Airplane", 
                    1: "Automobile",
                    2: "Bird",
                    3: "Cat",
                    4: "Deer",
                    5: "Dog",
                    6: "Frog",
                    7: "Horse",
                    8: "Ship",
                    9: "Truck"}
                    
        # mapped outcomes dictionary with the output
        result = outcomes.get(output[0], "Unexpected Outcome")

        # Rendering the prediction on to the webpage
        return render_template("canvas.html", prediction_text = "The image is of {}".format(result))

# run_cmd_file = r"cd\Software Development\Projects\Object Recognition (CNN on CIFAR-10)\Front End\python app.py"
# os.system(run_cmd_file)

if __name__ == "__main__":
    app.run(debug = True)
