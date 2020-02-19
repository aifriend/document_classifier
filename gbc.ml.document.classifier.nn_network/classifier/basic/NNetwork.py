import os
import shutil
import sys

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.neural_network._multilayer_perceptron import BaseMultilayerPerceptron
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import column_or_1d
from tensorflow_core.python.keras import Sequential, callbacks
from tensorflow_core.python.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow_core.python.keras.saving.save import save_model, load_model

from classifier.IClassify import IClassify
from common.model.ClassFile import ClassFile
from commonsLib import loggerElk


class NNetwork(IClassify):
    class CNNClassifier(ClassifierMixin, BaseMultilayerPerceptron):

        def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                     solver='adam', alpha=0.0001,
                     batch_size='auto', learning_rate="constant",
                     learning_rate_init=0.001, power_t=0.5, max_iter=200,
                     shuffle=True, random_state=None, tol=1e-4,
                     verbose=True, warm_start=False, momentum=0.9,
                     nesterovs_momentum=True, early_stopping=False,
                     validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                     epsilon=1e-8, n_iter_no_change=10, max_fun=15000, conf=None):
            super().__init__(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation, solver=solver, alpha=alpha,
                batch_size=batch_size, learning_rate=learning_rate,
                learning_rate_init=learning_rate_init, power_t=power_t,
                max_iter=max_iter, loss='log_loss', shuffle=shuffle,
                random_state=random_state, tol=tol, verbose=verbose,
                warm_start=warm_start, momentum=momentum,
                nesterovs_momentum=nesterovs_momentum,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                n_iter_no_change=n_iter_no_change, max_fun=max_fun)
            # Load model
            self.conf = conf
            self.cnn_keras = None
            self.logger = loggerElk(__name__, True)

        def init(self):
            # Building the model
            self.cnn_keras = Sequential()

            # Creating the method for model
            # Step 1- Convolution
            self.cnn_keras.add(Convolution2D(64, (5, 5),
                                             input_shape=(self.conf.nn_image_size, self.conf.nn_image_size, 1),
                                             activation='relu'))
            # adding another layer
            self.cnn_keras.add(Convolution2D(64, (4, 4), activation='relu'))
            # Pooling it
            self.cnn_keras.add(MaxPooling2D(pool_size=(2, 2)))
            # Adding another layer
            self.cnn_keras.add(Convolution2D(32, (3, 3), activation='relu'))
            # Pooling
            self.cnn_keras.add(MaxPooling2D(pool_size=(2, 2)))
            # Step 2- Flattening
            self.cnn_keras.add(Flatten())
            # Step 3- Full connection
            self.cnn_keras.add(Dense(units=128, activation='relu'))
            # For the output step
            self.cnn_keras.add(Dense(units=self.conf.nn_class_size, activation='softmax'))
            self.cnn_keras.add(Dropout(0.02))
            # Add reularizers
            # cnn_keras.add(Dense(128,
            #               input_dim = 128,
            #               kernel_regularizer = regularizers.l1(0.001),
            #               activity_regularizer = regularizers.l1(0.001),
            #               activation = 'relu'))

            self.cnn_keras.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # dropout = cnn_keras.add(Dropout(0.2))

        def save(self):
            try:
                if self.cnn_keras:
                    self.logger.Information('IClassify::saving model...')
                    dir_path = os.path.join(self.conf.working_path, self.conf.nn_model_path)
                    if os.path.exists(dir_path):
                        shutil.rmtree(dir_path)
                    os.makedirs(dir_path)
                    save_model(self.cnn_keras, filepath=dir_path, overwrite=True)
            except Exception as e:
                self.logger.Debug('NNetwork::SAVE::{}::{}'.format(e, self.conf.nn_model_path), sys.exc_info())

        def load(self):
            try:
                self.logger.Information('IClassify::loading model...')
                file_list = ClassFile.list_files(os.path.join(self.conf.working_path, self.conf.nn_model_path))
                for file in file_list:
                    file_root, file_name = os.path.split(file)
                    if file_name == self.conf.nn_model_name:
                        self.cnn_keras = load_model(
                            filepath=os.path.join(self.conf.working_path, self.conf.nn_model_path))
                        break
            except Exception as e:
                self.cnn_keras = None
                self.logger.Debug('NNetwork::LOAD::{}::{}'.format(e, self.conf.nn_model_path), sys.exc_info())

        def fit(self, training_set, validation_set):
            """
            Fit the model to data matrix X and target(s) y.

            """
            check_pointer = callbacks.ModelCheckpoint(
                filepath=self.conf.working_path,
                monitor='val_accuracy',
                verbose=self.conf.nn_verbose,
                save_best_only=True,
                save_weights_only=False,
                mode='auto')
            history = self.cnn_keras.fit(x=training_set,
                                         verbose=self.conf.nn_verbose,
                                         steps_per_epoch=(training_set.n // self.conf.nn_batch_size),
                                         epochs=self.conf.nn_epochs,
                                         validation_data=validation_set,
                                         validation_steps=(validation_set.n // self.conf.nn_batch_size),
                                         callbacks=[check_pointer])

            return history

        @property
        def partial_fit(self):
            """Update the model with a single iteration over the given data.

            classes : array, shape (n_classes), default None
                Classes across all calls to partial_fit.
                Can be obtained via `np.unique(y_all)`, where y_all is the
                target vector of the entire dataset.
                This argument is required for the first call to partial_fit
                and can be omitted in the subsequent calls.
                Note that y doesn't need to contain all labels in `classes`.

            Returns
            -------
            self : returns a trained MLP model.
            """
            # if self.solver not in _STOCHASTIC_SOLVERS:
            #     raise AttributeError("partial_fit is only available for stochastic"
            #                          " optimizer. %s is not stochastic"
            #                          % self.solver)
            # return self._partial_fit
            return

        @staticmethod
        def get_classes(y):
            label_binarizer = LabelBinarizer()
            label_binarizer.fit(y)
            return label_binarizer.classes_

        def _partial_fit(self, X, y, classes=None):
            # if _check_partial_fit_first_call(self, classes):
            #     self._label_binarizer = LabelBinarizer()
            #     if type_of_target(y).startswith('multilabel'):
            #         self._label_binarizer.fit(y)
            #     else:
            #         self._label_binarizer.fit(classes)
            #
            # super()._partial_fit(X, y)
            #
            # return self
            pass

        def _validate_input(self, X, y, incremental):
            X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                             multi_output=True)
            if y.ndim == 2 and y.shape[1] == 1:
                y = column_or_1d(y, warn=True)

            if not incremental:
                self._label_binarizer = LabelBinarizer()
                self._label_binarizer.fit(y)
                self.classes_ = self._label_binarizer.classes_
            elif self.warm_start:
                classes = unique_labels(y)
                if set(classes) != set(self.classes_):
                    raise ValueError("warm_start can only be used where `y` has "
                                     "the same classes as in the previous "
                                     "call to fit. Previously got %s, `y` has %s" %
                                     (self.classes_, classes))
            else:
                classes = unique_labels(y)
                if len(np.setdiff1d(classes, self.classes_, assume_unique=True)):
                    raise ValueError("`y` has classes not in `self.classes_`."
                                     " `self.classes_` has %s. 'y' has %s." %
                                     (self.classes_, classes))

            y = self._label_binarizer.transform(y)
            return X, y

        def predict(self, X):
            """Predict using the multi-layer perceptron classifier

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                The input data.

            Returns
            -------
            y : array-like, shape (n_samples,) or (n_samples, n_classes)
                The predicted classes.
            """
            # check_is_fitted(self)
            # y_pred = self._predict(X)
            #
            # if self.n_outputs_ == 1:
            #     y_pred = y_pred.ravel()
            #
            # return self._label_binarizer.inverse_transform(y_pred)

            y_pred = self.cnn_keras.predict(X)
            return y_pred

        def predict_log_proba(self, X):
            """Return the log of probability estimates.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input data.

            Returns
            -------
            log_y_prob : array-like, shape (n_samples, n_classes)
                The predicted log-probability of the sample for each class
                in the model, where classes are ordered as they are in
                `self.classes_`. Equivalent to log(predict_proba(X))
            """
            # y_prob = self.predict_proba(X)
            # return np.log(y_prob, out=y_prob)
            pass

        def predict_proba(self, X):
            """Probability estimates.

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                The input data.

            Returns
            -------
            y_prob : array-like, shape (n_samples, n_classes)
                The predicted probability of the sample for each class in the
                model, where classes are ordered as they are in `self.classes_`.
            """
            # check_is_fitted(self)
            # y_pred = self.cnn_keras.predict_proba(X)
            #
            # if self.n_outputs_ == 1:
            #     y_pred = y_pred.ravel()
            #
            # if y_pred.ndim == 1:
            #     return np.vstack([1 - y_pred, y_pred]).T
            # else:
            #     return y_pred
            return self.predict(X)

    def __init__(self, conf):
        super().__init__(conf)
        self.clf = self.CNNClassifier(
            solver=self.conf.nn_solver,
            alpha=self.conf.nn_alpha,
            hidden_layer_sizes=self.conf.nn_hidden_layer_sizes,
            random_state=self.conf.nn_random_state,
            verbose=self.conf.nn_verbose,
            conf=conf
        )

    def train(self, train, validation, test):
        self.clf.init()
        history = self.clf.fit(train, validation)
        self.save_model()
        response = self._get_prediction(test)
        return response, history

    def _get_prediction(self, test):
        response = list()
        try:
            class_list = {v: k for k, v in test.class_indices.items()}
            probabilities = self.clf.cnn_keras.predict_generator(test, verbose=1,
                                                                 steps=(test.n / self.conf.nn_batch_size))
            probabilities = probabilities.astype(np.float16)
            predicted_id = np.argmax(probabilities, axis=1)
            predicted = [class_list[k] for k in predicted_id]

            response.append(predicted)
            response.append(class_list)
            response.append(probabilities.tolist())
        except Exception as e:
            self.logger.Debug('NNetwork::get_prediction::{}'.format(e), sys.exc_info())
            return None

        return response

    def predict(self, test):
        self.load_model()
        response = self._get_prediction(test)
        return response

    def load_model(self, **kwargs):
        self.clf.load()

    def save_model(self, **kwargs):
        self.clf.save()

    def has_model(self):
        return self.clf.cnn_keras and self.clf.cnn_keras is not None
