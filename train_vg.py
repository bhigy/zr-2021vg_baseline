import datetime
import logging
import random
import torch

import platalea.basic as M
import platalea.dataset as D
from platalea.experiments.config import get_argument_parser


args = get_argument_parser()
# Parsing arguments
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)

# Logging the arguments
logging.info('Arguments: {}'.format(args))


logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
        args.audio_features_fn, split='train', batch_size=32, shuffle=True,
        downsampling_factor=args.downsampling_factor),
    val=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
        args.audio_features_fn, split='val', batch_size=32, shuffle=False))

config = dict(
    SpeechEncoder=dict(
        conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2,
                  padding=0, bias=False),
        rnn=dict(input_size=64, hidden_size=args.hidden_size_factor, num_layers=4,
                 bidirectional=True, dropout=0),
        att=dict(in_size=2 * args.hidden_size_factor, hidden_size=128)),
    ImageEncoder=dict(
        linear=dict(in_size=2048, out_size=2*args.hidden_size_factor),
        norm=True),
    margin_size=0.2)

logging.info('Building model')
net = M.SpeechImage(config)
run_config = dict(max_lr=args.cyclic_lr_max, min_lr=args.cyclic_lr_min, epochs=args.epochs,
                  l2_regularization=args.l2_regularization,)

logging.info('Training')
old_time = datetime.datetime.now()
logging.info(f'Start of training - {old_time}')
M.experiment(net, data, run_config, wandb_mode='disabled')
new_time = datetime.datetime.now()
logging.info(f'End of training - {new_time}')
diff_time = new_time - old_time
logging.info(f'Total duration: {diff_time}')
