import tensorflow as tf
import cv2
from keras.models import model_from_json
from keras.optimizers import RMSprop
import numpy as np
cap = cv2.VideoCapture(0)

while(True):
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    x = int(height//2 - 64)
    y = int(width//2 - 64)
    frame = cap.read()[1][x : x + 128,  y : y + 128]

    # Display the resulting frame
    cv2.imshow('frame', frame)
    #PRESS Q TO CAPTURE WANTED IMAGE
    if cv2.waitKey(1) & 0xFF == 13:
        #print(gray.shape)
        cv2.imwrite('input.png', frame)

        break

cap.release()
cv2.destroyAllWindows()

file = open('cnn_model_500im_epoch30_batch40_layer3_RGB_reshape.json', 'r')
model_json = file.read()
file.close()

loaded_model = model_from_json(model_json)
loaded_model.load_weights('RBGmodel.h5')

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# use categorical crossentropy as loss function
loaded_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
          'X', 'Y']

x = cv2.imread('input.png')
q = cv2.imread('input.png')

x = np.expand_dims(x, axis=0)


#print(x.shape)
y = loaded_model.predict_classes(x)

window = "Output"

font = cv2.FONT_HERSHEY_PLAIN

org = (50, 50)

fontScale = 3

color = (255, 0 ,0)

thickness = 2

q = cv2.putText(q, labels[y[0]], org, font, fontScale, color, thickness, cv2.LINE_AA)

cv2.imshow(window, q)
cv2.waitKey(0)