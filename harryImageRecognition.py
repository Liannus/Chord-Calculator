from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("media/images/models/model_ex-001_acc-0.100000.h5")
prediction.setJsonPath("media/images/json/model_class.json")
prediction.loadModel(num_objects=11)

predictions, probabilities = prediction.predictImage("media/images/test/am/am.png", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)