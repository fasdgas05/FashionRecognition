import tensorflow as tf
from PIL import Image
import io
from object_detection.utils import dataset_util
import sys



def create_tf_example(example, LABEL_DICT):
  # TODO(user): Populate the following variables from your example.
  f_image = Image.open(example['image_name'])
  width, height = f_image.size # Image height # Image width
  filename = example['image_name'].encode() # Filename of the image. Empty if image is not from file
  encoded_image_data = io.BytesIO() # Encoded image bytes
  f_image = f_image.convert('RGB')
  f_image.save(encoded_image_data, format='jpeg')
  encoded_image_data = encoded_image_data.getvalue()
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [example['x_1'] / width] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [example['x_2'] / width] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [example['y_1'] / height] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [example['y_2'] / height] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [LABEL_DICT[example['category_type']].encode()] # List of string class name of bounding box (1 per box)
  classes = [example['category_type']] # List of integer class id of bounding box (1 per box)

  assert (xmins[0] >= 0.) and (xmaxs[0] < 1.01) and (ymins[0] >= 0.) and (ymaxs[0] < 1.01), (example, width, height, width, height, xmins, xmaxs, ymins, ymaxs)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

flags = tf.compat.v1.app.flags
outfile_name = 'train_dataset.record'
if len(sys.argv) == 2:
  if sys.argv[1] == 'train':
      outfile_name = 'train_dataset.record'
  elif sys.argv[1] == 'val':
    outfile_name = 'val_dataset.record'
  elif sys.argv[1] == 'test':
    outfile_name = 'test_dataset.record'
flags.DEFINE_string('output_path', 'label/' + outfile_name, '')
FLAGS = flags.FLAGS

def main(_):
  writer = tf.compat.v1.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO(user): Write code to read in your dataset to examples variable
  if len(sys.argv) == 1:
    data = open('merged_train.txt', 'r').read().splitlines()
  if len(sys.argv) == 2:
    if sys.argv[1] == 'train':
      data = open('merged_train.txt', 'r').read().splitlines()
    elif sys.argv[1] == 'val':
      data = open('merged_val.txt', 'r').read().splitlines()
    elif sys.argv[1] == 'test':
      data = open('merged_test.txt', 'r').read().splitlines()


  cats = open('D:/notebook/cabstone/data/Anno_fine/list_category_cloth.txt').readlines()
  categories = list()
  for i in cats[2:]:
  	categories.append(i.split()[0])
  	pass

  label_dict = dict()
  cnt = 1
  for i in categories:
  	label_dict[cnt] = i
  	cnt += 1
  examples = list()

  for line in data:
  	cur = dict()
  	ss = line.split()
  	ss[0] = ss[0].replace('img','img_highres',1)
  	cur['image_name'] = 'D:/notebook/cabstone/data/' + ss[0]
  	cur['x_1'] = int(ss[1])
  	cur['y_1'] = int(ss[2])
  	cur['x_2'] = int(ss[3])
  	cur['y_2'] = int(ss[4])
  	cur['category_type'] = int(ss[5])

  	examples.append(cur)
  	pass


  for example in examples:
    print(example['image_name'], '->', outfile_name)
    try:
	    tf_example = create_tf_example(example, label_dict)
	    writer.write(tf_example.SerializeToString())
	except Exception as e:
		print(type(e), e)
		continue

  writer.close()


if __name__ == '__main__':
  tf.compat.v1.app.run()