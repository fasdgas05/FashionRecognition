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
  f_image.save(encoded_image_data, format='jpeg')
  encoded_image_data = encoded_image_data.getvalue()
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [example['x_1'] / width] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [example['x_2'] / width] # List of normalized right x coordinates in bounding box (1 per box)
  ymins = [example['y_1'] / height] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [example['y_2'] / height] # List of normalized bottom y coordinates in bounding box (1 per box)
  classes_text = [LABEL_DICT[example['category_type']].encode()] # List of string class name of bounding box (1 per box)
  classes = [example['category_type']] # List of integer class id of bounding box (1 per box)

  assert (xmins[0] >= 0.) and (xmaxs[0] < 1.01) and (ymins[0] >= 0.) and (ymaxs[0] < 1.01), (example, width, height, xmins, xmaxs, ymins, ymaxs)

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


def main(_):

	split_mask = dict()

	f = open('D:/notebook/cabstone/data/Eval/list_eval_partition.txt','r')
	types = f.read().splitlines()[2:]
	f.close()

	f = open('D:/notebook/cabstone/data/Anno_coarse/list_category_img.txt','r')
	cate = f.read().splitlines()[2:]
	f.close()

	f = open('D:/notebook/cabstone/data/Anno_coarse/list_bbox.txt')
	bb = f.read().splitlines()[2:]
	f.close()

	data = {'train' : list(), 'val' : list(), 'test' : list()}

	for i in range(len(types)):
		ts = types[i].split()
		cs = cate[i].split()
		bs = bb[i].split()

		if ts[1] in data.keys():
			data[ts[1]].append({
				'image_name' :  'D:/notebook/cabstone/data/' + ts[0],
				'x_1' : int(bs[1]),
				'y_1' : int(bs[2]),
				'x_2' : int(bs[3]),
				'y_2' : int(bs[4]),
				'category_type' : int(cs[1])
				})
			pass
		else:
			print('Error' + ts[0])
			pass
		pass


	cats = open('D:/notebook/cabstone/data/Anno_coarse/list_category_cloth.txt').readlines()
	categories = list()
	for i in cats[2:]:
		categories.append(i.split()[0])
		pass

	label_dict = dict()
	cnt = 1
	for i in categories:
		label_dict[cnt] = i
		cnt += 1
		pass

	flags = tf.compat.v1.app.flags

	for name in data:
		outfile_name = f'{name}_dataset.record'
		flags = tf.compat.v1.app.flags
		flags.DEFINE_string('output_path', 'model/label/' + outfile_name, '')
		FLAGS = flags.FLAGS
		writer = tf.compat.v1.python_io.TFRecordWriter(FLAGS.output_path)
		for example in data[name]:
			print(example['image_name'], '->', outfile_name)
			try:
				tf_example = create_tf_example(example, label_dict)
				writer.write(tf_example.SerializeToString())
			except Exception as e:
				print(e)
				continue
			pass
		writer.close()


if __name__ == '__main__':
	tf.compat.v1.app.run()	