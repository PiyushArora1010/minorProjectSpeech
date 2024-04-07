import argparse
from trainer import trainer

parser = argparse.ArgumentParser()

parser.add_argument('--model_type', type=str, default='xgb')
parser.add_argument('--data', type=str, default='urbansound')
parser.add_argument('--feature_type', type=str, default='stft')
# "mfcc", "log_mel", "stft"

if __name__ == '__main__':
    args = parser.parse_args()
    trainer = trainer(args)
    trainer.run()
