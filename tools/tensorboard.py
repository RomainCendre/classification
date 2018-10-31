import os
import logging
import sys

from os.path import join
from keras.callbacks import TensorBoard
from tensorboard import default
from tensorboard import program
import tensorflow


class TensorBoardTool:

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        # Remove http messages
        log = logging.getLogger('werkzeug').setLevel(logging.ERROR)
        # Start tensorboard server
        tb = program.TensorBoard(default.PLUGIN_LOADERS, default.get_assets_zip_provider())
        tb.configure(argv=['--logdir', self.dir_path])
        url = tb.launch()
        sys.stdout.write('TensorBoard at %s \n' % url)

    def write_batch(self):
        file = open(join(self.dir_path, 'tb_launch.bat'), "w")
        file.write('tensorboard --logdir={}'.format("./"))
        file.close()


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tensorflow.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tensorflow.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


class DataProjector:

    def test(datas, labels, path):
        tf_data = tf.Variable(datas)
        with tf.Session() as sess:
            saver = tf.train.Saver([tf_data])
            sess.run(tf_data.initializer)
            saver.save(sess, os.path.join(path, 'tf_data.ckpt'))
            config = projector.ProjectorConfig()
            # One can add multiple embeddings.
            embedding = config.embeddings.add()
            embedding.tensor_name = tf_data.name
            # Link this tensor to its metadata(Labels) file
            savetxt(os.path.join(path, 'metadata.tsv'), labels, delimiter='\t', fmt='%s')
            embedding.metadata_path = os.path.join(path, 'metadata.tsv')
            # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(tf.summary.FileWriter(path), config)