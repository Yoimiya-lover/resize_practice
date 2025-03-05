import os
import torch
import image_net_config
from image_net_evaluator import ImageNetEvaluator
from image_net_trainer import ImageNetTrainer
from image_net_data_loader import ImageNetDataloade 
#train&val数据集路径
DATASET_DIR = './dataset/'
#定义ImageNetDataPipeline类,包括数据加载、finetune训练和评估方法
class ImageNetDataPipeline:

    @staticmethod 
    '标记为静态方法'
    def get_val_dataloader() -> torch.utils.data.DataLoader:
        """
        Instantiates a validation dataloader for Imageenet dataset and returns it
        """
        data_loader = ImageNetDataLoader(DATASET_DIR,
            image_size=image_net_config.dataset['image_size'],
            batch_size=image_net_config.evaluation['batch_size'],
            is_training=False,
            num_workers=image_net_config.evaluation['num_workers']).data_loader

        return data_loader    
        @staticmethod
        def evaluate(model: torch.nn.Module, use_cuda: bool) ->float:
            evaluator = ImageNetEvaluator(DATASET_DIR,)