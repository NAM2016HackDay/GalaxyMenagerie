import re
import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

class NodeLookup(object):
  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = 'imagenet_2012_challenge_label_map_proto.pbtxt'
    if not uid_lookup_path:
      uid_lookup_path = 'imagenet_synset_to_human_label_map.txt'
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    if not gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    proto_as_ascii_lines = gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    node_id_to_uid = {}
    proto_as_ascii = gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    node_id_to_name = {}
    for key, val in node_id_to_uid.iteritems():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  with gfile.FastGFile('classify_image_graph_def.pb', 'r') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(image):
  if not gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = gfile.FastGFile(image).read()

  create_graph()

  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    node_lookup = NodeLookup()

    top_k = predictions.argsort()[:][::-1]
    data = []
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      if score > 0.01:
          data.append([human_string,score])
    return data

def main(_):

  col = []
  id_old = '0'
  with open('out/out.txt','r') as f:
    for line in f:
      id_new = line.split('\t')[0]
      if id_new != id_old:
        col.append(id_new)
        id_old = id_new

  gal_old = 'a'
  for filename in glob.glob('images/*.jpg'):
    string = filename.replace('images/','').replace('.jpg','')
    gal_id = string.split('_')
    if gal_id[0] not in col: 
      with tf.Graph().as_default():
        with open('out/out.txt','a') as f:
          if gal_id[0] == gal_old:
            print gal_id[1]
            results = run_inference_on_image(filename)
            for i in results:
              data.append(i)
              if gal_id[1] == '4':
                data.sort(key=lambda x: x[1])
                data = data[::-1]
                data_old = []
                for i in data:
                  if i[0] not in data_old:
                    data_old.append(i[0])
                    f.write(gal_id[0]+'\t'+i[0]+'\t'+str(i[1])+'\n')
          else:
            print gal_id[0], gal_id[1]
            gal_old = gal_id[0]
            data = []
            results = run_inference_on_image(filename)
            for i in results:
              data.append(i)

if __name__ == '__main__':

  tf.app.run()
