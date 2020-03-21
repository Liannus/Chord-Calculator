from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("media/images")
model_trainer.trainModel(num_objects=11, num_experiments=3, enhance_data=True, batch_size=1, show_network_summary=True)