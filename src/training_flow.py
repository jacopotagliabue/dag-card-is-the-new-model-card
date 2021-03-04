from metaflow import FlowSpec, step, IncludeFile, S3, Parameter, current
import os
from io import StringIO
from random import choice
from dotenv import load_dotenv
# load envs from file
load_dotenv(verbose=True)
# make sure wand api key is there
assert os.getenv('WANDB_API_KEY') is not None


class RegressionModel(FlowSpec):
    """
    RegressionModel is a DAG that produces a regression model over product prices. Given as input a set of features
    and a list of prices per product, the output is a Keras model able to predict the price of unseen items.
    """
    DATA_FILE = IncludeFile(
        'dataset',
        help='Text File With Regression Numbers',
        is_text=True,
        default='dataset.txt')

    LEARNING_RATES = Parameter(
        name='learning_rates',
        help='Learning rates to test, comma separeted',
        default='0.1,0.2'
    )

    @step
    def start(self):
        """
        Read data in, and parallelize model building with two params.
        """
        raw_data = StringIO(self.DATA_FILE).readlines()
        self.dataset = [[float(_) for _ in d.strip().split('\t')] for d in raw_data]
        split_index = int(len(self.dataset) * 0.8)
        self.train_dataset = self.dataset[:split_index]
        self.test_dataset = self.dataset[split_index:]
        print("Training data: {}, test data: {}".format(len(self.train_dataset), len(self.test_dataset)))
        self.learning_rates = [float(_) for _ in self.LEARNING_RATES.split(',')]
        self.next(self.train_model, foreach='learning_rates')

    @step
    def train_model(self):
        """
        Train a regression model and use s3 client from metaflow to store the model tar file.
        """
        # this is the current learning rate in the fan-out
        self.learning_rate = self.input
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras import layers
        import tarfile
        import wandb
        from wandb.keras import WandbCallback
        # this name comes in handy later, as a naming convention for building the card
        wandb_run_name = '{}:{}-{}'.format(current.flow_name, current.run_id, self.learning_rate)
        wandb.init(project=current.flow_name, name=wandb_run_name)
        # build the model
        x_train = np.array([[_[0]] for _ in self.train_dataset])
        y_train = np.array([_[1] for _ in self.train_dataset])
        x_test = np.array([[_[0]] for _ in self.test_dataset])
        y_test = np.array([_[1] for _ in self.test_dataset])
        x_model = tf.keras.Sequential([layers.Dense(input_shape=[1,], units=1)])
        self.model_summary = x_model.summary()
        x_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_absolute_error', metrics=[tf.keras.metrics.MeanSquaredError()])
        history = x_model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[WandbCallback()])
        self.hist = history.history
        self.results = x_model.evaluate(x_test, y_test)
        print("Test set results: {}".format(self.results))
        model_name = "regression-model-{}/1".format(self.learning_rate)
        local_tar_name = 'model-{}.tar.gz'.format(self.learning_rate)
        x_model.save(filepath=model_name)
        # zip keras folder to a single tar file
        with tarfile.open(local_tar_name, mode="w:gz") as _tar:
            _tar.add(model_name, recursive=True)
        with open(local_tar_name, "rb") as in_file:
            data = in_file.read()
            with S3(run=self) as s3:
                url = s3.put(local_tar_name, data)
                # save this path for downstream reference!
                self.s3_path = url
        # finally join with the other runs
        self.next(self.join_runs)

    @step
    def join_runs(self, inputs):
        """
        Join the parallel runs, merge results into a dictionary, and finally pick the best model according
        to some custom logic.
        """
        # merge results (loss) from runs with different parameters
        self.results_from_runs = {
            inp.learning_rate:
                {
                    'metrics': inp.results,
                    'tar': inp.s3_path,
                    'summary': inp.model_summary
                }
            for inp in inputs}
        print("Current results: {}".format(self.results_from_runs))
        # pick one according to some logic: here just pick a random one
        self.best_learning_rate = choice(list(self.results_from_runs.keys()))
        self.best_s3_model_path = self.results_from_runs[self.best_learning_rate]['tar']
        self.best_model_metrics = self.results_from_runs[self.best_learning_rate]['metrics']
        self.best_model_summary = self.results_from_runs[self.best_learning_rate]['summary']
        self.next(self.end)

    @step
    def end(self):
        """
        DAG is done!
        """
        print('Dag ended!')


if __name__ == '__main__':
    RegressionModel()