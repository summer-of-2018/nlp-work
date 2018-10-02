import bilsm_crf_model
import process_data
import numpy as np
import tensorflow as tf

model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
model.load_weights('model/crf_b.h5')

y_pred = model.predict(train_x)
y_pred2 = []
for row in y_pred:
    for w in row:
        y_pred2.append(np.argmax(w))
y_pred2 = np.array(y_pred2)
y_label = train_y.reshape(len(y_pred2))
y_label[y_label<0]=11
y_pred2[y_pred2<0]=11
con = tf.confusion_matrix(y_label, y_pred2)
sess = tf.Session()
with sess.as_default():
        print(sess.run(con))
