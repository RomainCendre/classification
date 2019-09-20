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
        tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
        tb.configure(argv=[None, '--logdir', self.dir_path])
        url = tb.launch()
        sys.stdout.write('TensorBoard at %s \n' % url)

    def write_batch(self):
        file = open(join(self.dir_path, 'tb_launch.bat'), "w")
        file.write('tensorboard --logdir={}'.format("./"))
        file.close()


class TensorBoardWriter(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        super(TensorBoardWriter, self).__init__(join(log_dir, 'training'), **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tensorflow.summary.FileWriter(self.val_log_dir)
        super(TensorBoardWriter, self).set_model(model)

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
        super(TensorBoardWriter, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TensorBoardWriter, self).on_train_end(logs)
        self.val_writer.close()