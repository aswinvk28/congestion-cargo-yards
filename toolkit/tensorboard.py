import tensorflow.compat.v1 as tf
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

logs_path = '/home/project/logs'

PATH_TO_CKPT = '/home/project/toolkit/ssd_mobilenet_v1_coco_2017_11_17' + '/frozen_inference_graph.pb'

# fake data
image_tensor = np.ones((1,300,300,3))
noise = np.random.normal(0, 0.1, size=image_tensor.shape)
y = np.power(image_tensor, 2) + noise

tf_x = tf.placeholder(tf.float32, image_tensor.shape, name='image_tensor')
tf_y = tf.placeholder(tf.float32, y.shape, name='y')

layer = tf.layers.dense(tf_x, 100, tf.nn.relu, name='hidden_layer')
output = tf.layers.dense(layer, 3, tf.nn.relu, name='output_layer')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

loss = tf.losses.mean_squared_error(tf_y, output, scope='loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter(logs_path, sess.graph)     # write to file
merge_op = tf.summary.merge_all()                       # operation to merge all summary

for step in range(100):
    # train and net output
    _, result = sess.run([train_op, merge_op], {tf_x: image_tensor, tf_y: y})
    writer.add_summary(result, step)

# Lastly, in your terminal or CMD, type this :
# $ tensorboard --logdir path/to/log
# open you google chrome, type the link shown on your terminal or CMD. (something like this: http://localhost:6006)