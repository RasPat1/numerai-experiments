import tensorflow as tf

fileName = "../datasets/numerai_datasets/numerai_training_data.csv"

reader = tf.TextLineReader(skip_header_lines=1)
record_defaults = []

# 50 Features and 1 target
for iteration in range (0, 51):
  record_defaults.append([0.0])

tensors = tf.decode_csv(fileName, record_defaults=record_defaults, field_delim=",")


