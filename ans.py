from tensorflow.keras.preprocessing import image
import  numpy as np
import keras

img = image.load_img(r"C:\Users\Lenovo\Desktop\Linkdin_Projects\PVC_CNN\PVC\train\pvc\newplot_1.jpg", target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.
model = keras.models.load_model("vgg_model.h5")
predictions = model.predict(img)
name = ["no","yes"]
print(name[np.argmax(predictions)])
