import tempfile
import tensorflow as tf

fileName = "../datasets/numerai_datasets/numerai_training_data.csv"
test_fileName = "../datasets/numerai_datasets/numerai_training_data.csv"

record_defaults = []
feature_names = []

# 50 Features and 1 target
for iteration in range (0, 51):
  record_defaults.append([0.0])
  feature_names.append("feature_" + str(iteration))

def format_data(file):
  tensors = tf.decode_csv(fileName, record_defaults=record_defaults, field_delim=",")
  features = tensors[:-1]
  labels = tensors[-1]

  return features, labels

def train_input_fn():
  return input_fn(fileName)

def eval_input_fn():
  return input_fn(test_fileName)

model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=feature_names, model_dir=model_dir)

with tf.Session() as sess:
  m.fit(train_input_fn, steps=200)
  results = m.evaluate(eval_input_fn, steps=1)
  for key in sorted(results):
    print "%s: %s" % (key, results[key])
  # score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
  # print("Accuracy: %f" % score)
  # print(classifier.eval(feed_dict={x:features, y: labels}))
  #loss = tf.contrib.losses.log(predictions, labels)





'''
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(fileName)

record_defaults = []
# 50 Features and 1 target
for iteration in range (0, 51):
  record_defaults.append([0.0])

tensors = tf.decode_csv(value, record_defaults=record_defaults, field_delim=",")
features = tensors[:-1]
label = tensors[-1]

with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    example, label = sess.run([features, label])

  coord.request_stop()
  coord.join(threads)
'''