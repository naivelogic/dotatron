import argparse
import sys, os

def train_default_args(epilog=None):    
    parser = argparse.ArgumentParser(epilog=epilog,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def detect_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='normal', required=False)
    parser.add_argument("--output_dir", type=str, default="" , required=False)
    parser.add_argument("--conf", type=float, default=0.5, required=False)
    parser.add_argument("--gpu_id", type=bool, default=0, required=False,)
    return parser

def test_tron_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='normal', required=False)
    parser.add_argument("--output_dir", type=str, default="" , required=False)
    parser.add_argument("--conf", type=float, default=0.5, required=False)
    parser.add_argument("--use_gpu", type=bool, default=False, required=False)
    parser.add_argument("--ckpt_path", type=str, default="", required=False)
    parser.add_argument("--num_images", type=int, default=5, required=False)
    return parser