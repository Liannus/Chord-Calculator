from keras.models import model_from_json
import cv2
import os
import numpy as np

class ChordModel(object):

    CHORD_LIST = ["a", "am",
                     "b", "bm",
                     "c", "d",
                     "e", "em", "f", "fm", "g"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_chord(self, img):
        self.preds = self.loaded_model.predict(img)
        return ChordModel.CHORD_LIST[np.argmax(self.preds)]

#Loads the model
model = ChordModel("model.json", "model_weights.h5")

#Loads the images to test
basePath = os.getcwd()
imagesToIdentify = [basePath+"/media/images/train/a/a.png", basePath+"/media/images/train/am/am.png", basePath+"/media/images/train/b/b.png", basePath+"/media/images/train/bm/bm.png", basePath+"/media/images/train/c/c.png", basePath+"/media/images/train/d/d.png", basePath+"/media/images/train/e/e.png", basePath+"/media/images/train/em/em.png", basePath+"/media/images/train/f/f.png", basePath+"/media/images/train/fm/fm.png", basePath+"/media/images/train/g/g.png"]
correctChords = ["a", "am", "b", "bm", "c", "d", "e", "em", "f", "fm", "g"]

print ("Identifying chords")
for x in range(len(imagesToIdentify)):
    image = cv2.imread(imagesToIdentify[x])
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image3 = cv2.resize(image2, (200,300))
    pred = model.predict_chord(image3[np.newaxis, :, :, np.newaxis])
    print("Prediction is " + pred + ", correct answer is "+correctChords[x])

