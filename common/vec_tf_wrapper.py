#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : vec_tf_wrapper.py
# @Author: harry
# @Date  : 3/15/21 3:23 AM
# @Desc  : https://github.com/NeoExtended/stable-baselines/blob/master/stable_baselines/common/vec_env/vec_tf_wrapper.py

import json
import os
import zipfile
from abc import ABC, abstractmethod
from collections import OrderedDict

import tensorflow as tf
from stable_baselines.common import save_util
from stable_baselines.common.vec_env import VecEnvWrapper


class BaseTFWrapper(VecEnvWrapper, ABC):
    """
    Base class for wrappers which incorporate ML models like curiosity. Helps with saving and loading.
    Methods are copied from stable_baselines.common.BaseRLModel and stable_baselines.common.ActorCriticRLModel.
    :param env: (gym.Env) The environment to wrap.
    :param _init_setup_model: (bool) Boolean indicating weather the setup_model() method should be called on init.
    """

    def __init__(self, env, _init_setup_model):
        super().__init__(env)

        self._param_load_ops = None

    @abstractmethod
    def get_parameter_list(self):
        """
        Get tensorflow Variables of model's parameters
        This includes all variables necessary for continuing training (saving / loading).
        :return: (list) List of tensorflow Variables
        """
        pass

    def get_parameters(self):
        """
        Get current model parameters as dictionary of variable name -> ndarray.
        :return: (OrderedDict) Dictionary of variable name -> ndarray of model's parameters.
        """
        parameters = self.get_parameter_list()
        parameter_values = self.sess.run(parameters)
        return_dictionary = OrderedDict((param.name, value) for param, value in zip(parameters, parameter_values))
        return return_dictionary

    def load_parameters(self, load_path_or_dict, exact_match=True):
        """
        Load model parameters from a file or a dictionary
        Dictionary keys should be tensorflow variable names, which can be obtained
        with ``get_parameters`` function. If ``exact_match`` is True, dictionary
        should contain keys for all model's parameters, otherwise RunTimeError
        is raised. If False, only variables included in the dictionary will be updated.
        This does not load agent's hyper-parameters.
        .. warning::
            This function does not update trainer/optimizer variables (e.g. momentum).
            As such training after using this function may lead to less-than-optimal results.
        :param load_path_or_dict: (str or file-like or dict) Save parameter location
            or dict of parameters as variable.name -> ndarrays to be loaded.
        :param exact_match: (bool) If True, expects load dictionary to contain keys for
            all variables in the model. If False, loads parameters only for variables
            mentioned in the dictionary. Defaults to True.
        """
        # Make sure we have assign ops
        if self._param_load_ops is None:
            self._setup_load_operations()

        if isinstance(load_path_or_dict, dict):
            # Assume `load_path_or_dict` is dict of variable.name -> ndarrays we want to load
            params = load_path_or_dict
        else:
            # Assume a filepath or file-like. Use existing deserializer to load the parameters.
            # We only need the parameters part of the file, so only load that part.
            _, params = BaseTFWrapper._load_from_file(load_path_or_dict, load_data=False)
            params = dict(params)

        feed_dict = {}
        param_update_ops = []
        # Keep track of not-updated variables
        not_updated_variables = set(self._param_load_ops.keys())
        for param_name, param_value in params.items():
            placeholder, assign_op = self._param_load_ops[param_name]
            feed_dict[placeholder] = param_value
            # Create list of tf.assign operations for sess.run
            param_update_ops.append(assign_op)
            # Keep track which variables are updated
            not_updated_variables.remove(param_name)

        # Check that we updated all parameters if exact_match=True
        if exact_match and len(not_updated_variables) > 0:
            raise RuntimeError("Load dictionary did not contain all variables. " +
                               "Missing variables: {}".format(", ".join(not_updated_variables)))

        self.sess.run(param_update_ops, feed_dict=feed_dict)

    def _setup_load_operations(self):
        """
        Create tensorflow operations for loading model parameters
        """
        # Assume tensorflow graphs are static -> check
        # that we only call this function once
        if self._param_load_ops is not None:
            raise RuntimeError("Parameter load operations have already been created")
        # For each loadable parameter, create appropiate
        # placeholder and an assign op, and store them to
        # self.load_param_ops as dict of variable.name -> (placeholder, assign)
        loadable_parameters = self.get_parameter_list()
        # Use OrderedDict to store order for backwards compatibility with
        # list-based params
        self._param_load_ops = OrderedDict()
        with self.graph.as_default():
            for param in loadable_parameters:
                placeholder = tf.placeholder(dtype=param.dtype, shape=param.shape)
                # param.name is unique (tensorflow variables have unique names)
                self._param_load_ops[param.name] = (placeholder, param.assign(placeholder))

    @staticmethod
    def _save_to_file_zip(save_path, data=None, params=None):
        """Save model to a .zip archive
        :param save_path: (str or file-like) Where to store the model
        :param data: (OrderedDict) Class parameters being stored
        :param params: (OrderedDict) Model parameters being stored
        """
        # data/params can be None, so do not
        # try to serialize them blindly
        if data is not None:
            serialized_data = save_util.data_to_json(data)
        if params is not None:
            serialized_params = save_util.params_to_bytes(params)
            # We also have to store list of the parameters to store the ordering for OrderedDict.
            # We can trust these to be strings as they are taken from the Tensorflow graph.
            serialized_param_list = json.dumps(list(params.keys()), indent=4)

        # Check postfix if save_path is a string
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".zip"

        # Create a zip-archive and write our objects there. This works when save_path
        # is either str or a file-like
        with zipfile.ZipFile(save_path, "w") as file_:
            # Do not try to save "None" elements
            if data is not None:
                file_.writestr("data", serialized_data)
            if params is not None:
                file_.writestr("parameters", serialized_params)
                file_.writestr("parameter_list", serialized_param_list)

    @staticmethod
    def _load_from_file(load_path, load_data=True, custom_objects=None):
        """Load model data from a .zip archive
        :param load_path: (str or file-like) Where to load model from
        :param load_data: (bool) Whether we should load and return data (class parameters). Mainly used by `load_parameters` to
            only load model parameters (weights).
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :return: (dict, OrderedDict) Class parameters and model parameters
        """
        # Check if file exists if load_path is a string
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".zip"):
                    load_path += ".zip"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

        # Open the zip archive and load data.
        with zipfile.ZipFile(load_path, "r") as file_:
            namelist = file_.namelist()
            # If data or parameters is not in the zip archive, assume they were stored as None (_save_to_file allows this).
            data = None
            params = None
            if "data" in namelist and load_data:
                # Load class parameters and convert to string
                # (Required for json library in Python 3.5)
                json_data = file_.read("data").decode()
                data = save_util.json_to_data(json_data, custom_objects=custom_objects)

            if "parameters" in namelist:
                # Load parameter list and and parameters
                parameter_list_json = file_.read("parameter_list").decode()
                parameter_list = json.loads(parameter_list_json)
                serialized_params = file_.read("parameters")
                params = save_util.bytes_to_params(serialized_params, parameter_list)

        return data, params

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def save(self, save_path):
        pass

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        """
        Load the model from file
        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)

        model = cls(env=env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        #        model.set_env(env)
        model.setup_model()

        model.load_parameters(params)

        return model
