from configs.configVietOCR import params, dataset_params
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

class TrainVietOCR():
    def __init__(self, dataset_path, device = 'cuda:0', model = 'vgg_transformer'):
        global params, dataset_params

        dataset_params['data_root'] = dataset_path
        self.model = model
        self.config = Cfg.load_config_from_name(self.model)
        self.config['trainer'].update(params)
        self.config['dataset'].update(dataset_params)
        self.config['device'] = device
        self.config['dataloader']['num_workers'] = 0
        self.trainer = Trainer(self.config, pretrained = True)    

    def __call__(self, config_yml_path):
        self.trainer.train()
        self.trainer.config.save(config_yml_path)
        

    def check_precision(self):
        print(self.trainer.precision())