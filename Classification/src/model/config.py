# configuration, hyperparameters
# json file 

import os
import json
from types import SimpleNamespace

class Struct(object):
    """ Wrapping class for configuration. """
    def __init__(self, dict_):
        self.__dict__.update(dict_)
        
    def __repr__(self):
        return f"{self.__dict__}"

class ModelConfig(object):
    def __init__(self, version=None, **kwargs): # give mutable object
        self.version = version
        self.hyperparams = kwargs.pop("hyperparams", {})
        
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())
    
    # to_dict and to_json_string from "hugging_face library/src/transformers/modelcard.py".
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    
    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)
        
    @classmethod
    def from_json_file(cls, json_pth):     
        with open(json_pth) as f:
            config = json.load(f, object_hook = lambda d: SimpleNamespace(**d))
        return config
    
    # TODO: save options to dictionary or json file.