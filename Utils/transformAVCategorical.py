import numpy
from keras.models import load_model


"""
Categories: 0: Neutral, 1: Happy, 2: Sad, 3:
Surprise, 4: Fear, 5: Disgust, 6: Anger

"""


arousal = 1
valence = -1

modelDirectory = "arousalToCategorical.h5"
model = load_model(modelDirectory)
reading = numpy.array([arousal,valence])

reading = numpy.expand_dims(reading, 0)
reading = numpy.expand_dims(reading, 0)
category = numpy.argmax(model.predict([reading])[0][0])

print ("Category: "  + str(category))
