#Converts Keras-output h5 networks to nn-files expected by Unity ML-agents.
#2 steps: h5 to pb: https://medium.com/@aneeshpanoli/how-to-use-a-pre-trained-tensorflow-keras-models-with-unity-ml-agents-bee9933ce3c1
#And then pb to nn: https://github.com/Unity-Technologies/ml-agents/blob/master/UnitySDK/Assets/ML-Agents/Plugins/Barracuda.Core/Barracuda.md

import tensorflow as tf

# set learning phase to 0 since the model is already trained
tf.keras.backend.set_learning_phase(0)
#load the model
pre_model = tf.keras.models.load_model('path/to/your/model.h5')
#convert h5 to protobuffer
builder = pb_builder.SavedModelBuilder('path/to/the/pb/folder')
builder.save()
