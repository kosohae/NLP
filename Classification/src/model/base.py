class BaseConfig(nn.Module):
    """ Base class for TextCNN model config """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt # opt json file로 관리할 계획
    
    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            raise AssertionError
        os.makedirs(save_directory, exist_ok=True)
    
    # 모델의 공통 기능 : 모델 parameter 저장.
    # 모델 구성 불러오기

class ModelConfig(object):
    def __init__(self, file_dir=None, **kwargs):
        if file_dir is None: #file_dir : json file => read and return
            cls.from_json_file(file_dir)
        else:
            self.embedding_dir = kwargs.pop('embedding_dir', False)
            self.hidden_dim =  kwargs.pop('hidden_dim', 0)
            self.max_seq_len = kwargs.pop('max_seq_len' , 128)
            self.num_filters = kwargs.pop('num_filters', 0)
            self.filter_sizes = kwargs.pop('filter_sizes',False)
            self.embedding_dim = kwargs.pop('embedding_dim',0)
            self.dropout_rate = kwargs.pop('dropout_rate',0.0)
            
            self.batch_size = kwargs.pop('batch_size', 0)
            self.learning_rate = kwargs.pop('learning_rate',0)
            self.grad_clip = kwargs.pop('grad_clip',0.0)
            self.max_epoch = kwargs.pop('max_epoch',0)
            
            self.gpu = kwargs.pop('gpu', False)
               
    @classmethod
    def from_json_file(cls, json_file):
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict) # obj.key ...
    
    @classmethod
    def _dict_from_json_file(cls, json_file):
        """ load defined config json file"""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return json.load(text)
    
    def __eq__(self, other):
        """ check original config and others"""
        return self.__dict__ == other.__dict__