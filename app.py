from flask import Flask, render_template, redirect, request, jsonify, url_for
from io import BytesIO
from PIL import Image
import base64

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

from skimage import measure
from skimage import morphology
from skimage.transform import resize
from sklearn.cluster import KMeans

from tensorflow.keras import layers
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

app = Flask(__name__)
loaded = keras.models.load_model("model")

def split_lung_parenchyma(target,size,thr):
    file_bytes = np.asarray(bytearray(target.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    try:
        img_thr= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,size,thr).astype(np.uint8)
    except:
        img_thr= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,999,thr).astype(np.uint8)
    img_thr=255-img_thr
    img_test=measure.label(img_thr, connectivity = 1)
    props = measure.regionprops(img_test)
    img_test.max()
    areas=[prop.area for prop in props]
    ind_max_area=np.argmax(areas)+1
    del_array = np.zeros(img_test.max()+1)
    del_array[ind_max_area]=1
    del_mask=del_array[img_test]
    img_new = img_thr*del_mask
    mask_fill=fill_water(img_new)
    img_new[mask_fill==1]=255
    img_out=img*~img_new.astype(bool)
    return img_out

def fill_water(img):
    copyimg = img.copy()
    copyimg.astype(np.float32)
    height, width = img.shape
    img_exp=np.zeros((height+20,width+20))
    height_exp, width_exp = img_exp.shape
    img_exp[10:-10, 10:-10]=copyimg
    mask1 = np.zeros([height+22, width+22],np.uint8)   
    mask2 = mask1.copy()
    mask3 = mask1.copy()
    mask4 = mask1.copy()
    cv2.floodFill(np.float32(img_exp), mask1, (0, 0), 1) 
    cv2.floodFill(np.float32(img_exp), mask2, (height_exp-1, width_exp-1), 1) 
    cv2.floodFill(np.float32(img_exp), mask3, (height_exp-1, 0), 1) 
    cv2.floodFill(np.float32(img_exp), mask4, (0, width_exp-1), 1)
    mask = mask1 | mask2 | mask3 | mask4
    output = mask[1:-1, 1:-1][10:-10, 10:-10]
    return output


@app.route("/")
def index():
    if request.args.get("error"):
        return render_template("index.html",error=request.args.get("error"))
    return render_template("index.html")

@app.route("/proses", methods=["POST","GET"])
def proses():
    if request.method=="POST":
        if request.files["file"]:
            try:
                file = request.files["file"]
                imagefile = BytesIO(file.read())
                imagefile.seek(0)
                gambar = split_lung_parenchyma(imagefile,15599,-96)
                
                a = Image.fromarray(gambar)
                buffered = BytesIO()
                a.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue())

                backtorgb = cv2.cvtColor(gambar,cv2.COLOR_GRAY2RGB)
                backtorgb = cv2.resize(backtorgb,(100,100))
                img_array = tf.expand_dims(backtorgb, 0)
                img_array = normalization_layer(img_array)

                hasil = loaded.predict(img_array)[0]
                h = np.argmax(hasil)

                return render_template("hasil.html",gambar=img_str.decode("utf-8"), hasil=str(h))
            except:
                return redirect(url_for("index",error="badimage"))
        else:
            return jsonify({"status":"error","msg":"bad request"})
    else:
        return jsonify({"status":"error","msg":"bad request"})

if __name__=='__main__':
    app.run(debug=True)