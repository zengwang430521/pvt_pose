import warnings
warnings.simplefilter("ignore", UserWarning)
from utils import TrainOptions
from train.transformer_trainer import TransformerTrainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = TransformerTrainer(options)
    trainer.train()
