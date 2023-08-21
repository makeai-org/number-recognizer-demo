from flask import Flask as fk
from flask import url_for as url_for
from flask import render_template as rt
from flask import request as rq
import tensorflow as tf
import keras as ker
import numpy as np
import PIL
from PIL import Image
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist as mn
from keras.models import load_model
gpus = tf.config.list_physical_devices('GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=30000)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

def FindNum(num):
  greatest = num[0][0]
  greatest_num = 0
  for num_i in range(len(num[0])):
      print(num[0][num_i])
      comp = num[0][num_i]
      if comp > greatest:
          greatest_num = num_i
          print(str(comp) + ">" + str(greatest))
          greatest = comp
  return greatest_num

model = load_model('number-recognizer-model.h5')
print(model.summary())
(X_train, y_train),(X_test,y_test) = mn.load_data()
assert X_train.shape[0] == 60000
assert y_train.shape[0] == 60000
assert X_test.shape[0] == 10000
assert y_test.shape[0] == 10000
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
model.evaluate(X_test,y_test,batch_size=128,verbose=2)

app = fk(__name__)

list_img=[]
for i in range(9):
    ext = 'mnist_img/X_train[' + str(i) + '].png'
    list_img.append(ext)

@app.route('/', methods=['POST','GET'])
def test():
    version = tf.__version__
    label='To Demo'
    if rq.method == 'POST':
        req = rq.form['number']
        print(req)
        try:
            if -60000>int(req) or 60000<int(req):
                return rt('index.html', title='Number recognizer',link='index', linklabel=label, list_img=list_img, img="Out of Index",pred="Error: Out of index, no prediction",value="Not good error", version=version)
            elif int(req) > -1:
                img_name = 'mnist_img/X_train[' + str(req) + '].png'
                num = int(req)
            else:
                img_name = 'mnist_img/X_train[' + str(60000+int(req)) + '].png'
                num = 60000+int(req)
        except:
            return rt('index.html', title='Number recognizer',link='index', linklabel=label, list_img=list_img, img="Out of Index",pred="Error: Out of index, no prediction",value="Not good error", version=version)
        #try:
        print(req)
        print(img_name)
        value = X_train[int(req)]
        value = tf.reshape(value,shape=(1,28,28,1))
        pred = FindNum(model.predict(value))
        print(pred)
        i_img_name = 'static/' + img_name
        try:
            print(img_name)
            print("tried")
            img_test = Image.open(i_img_name)
        except:
            print('new')
            fig,axs= plt.subplots()
            img_val = tf.reshape(value, shape=(28,28,1))
            axs.imshow(img_val,cmap='gray')
            axs.axis('off')
            name = 'X_train[' + str(num) + ']'
            fig.savefig(r"static/mnist_img/{0}.png".format(name))
            img_test = Image.open(i_img_name)
        return rt('index.html', title='Demo', link='index', linklabel=label, list_img=list_img, img=img_name,pred=pred,value=value, version=version)
    else:   
        return rt('index.html', link='index', linklabel=label, list_img=list_img,version=version, title='Demo')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=False)
