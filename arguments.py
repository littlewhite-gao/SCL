import argparse


def str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--data_dir", required=True,
                        help="Location of data files (model weights, etc).")
    parser.add_argument("--electra_model", required=True,
                        help="The path of the model being fine-tuned.")
    parser.add_argument("--epochs", required=True, type=float,
                        help="epochs")
    parser.add_argument("--output_dir", required=True,
                        help="output dir")
    parser.add_argument("--hparams", default="{}", required=True,
                        help="JSON dict of model hyperparameters.")

    #  pooling
    parser.add_argument("--pooling", default=None, type=str,
                        choices=['average', 'last2_cls_average', 'last2_pool_cls', 'last2_seq_average', None],
                        help="one of ['average', 'last2_cls_average', 'last2_seq_average', None]")

    # write data you want
    parser.add_argument("--write", default=True, type=str2bool,
                        help="write wrong predict data or not")
    parser.add_argument("--log_every", default=100, type=int,
                        help="log every n steps")

    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay rate")
    parser.add_argument("--warmup", default=0.1, type=float,
                        help="warmup proportion")
    parser.add_argument("--layerwise", default=0.8, type=float,
                        help="layer_wise_decay")

    parser.add_argument("--noise_scale", default=1e-5, type=float,
                        help="stddev of noise")
    parser.add_argument("--tau", default=1.0, type=float,
                        help="temperature")
    parser.add_argument("--alpha", default=1.0, type=float,
                        help="the coefficient of contrastive loss")
    parser.add_argument("--scl_drop", default=0.1, type=float,
                        help="rate of embedding dropout")
    parser.add_argument("--use_cl", default=False, type=str2bool,
                        help="use contrastive learning or not")
    parser.add_argument("--c_type", default=None, type=str,
                        choices=[None, '2dropout', 'drop_embedding', 'feature_cutoff', 'token_cutoff',
                                 'shuffle', 'noise'], help="alpha")
    parser.add_argument("--cut_rate", default=0.1, type=float,
                        help="rate of feature_cutoff, token_cutoff and shuffle_length")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout prob")
    parser.add_argument("--seed", default=0, type=int,
                        help="random seed")
    parser.add_argument("--set_seed", default=False, type=str2bool,
                        help="set seed or not")
    args = parser.parse_args()
    return args