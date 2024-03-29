import argparse, os, random, sys, torch
from .utils.logger import logger
from .utils.utils import SRC_PATH
from datetime import datetime
import numpy as np
from .inferencing import inferencing
from .mining import mining

__version__ = "0.1.33"

def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    available_languages = ["Python", "Java", "C++", "C", "C#", "JavaScript", "TypeScript", "Ruby", "PHP", "Go", "Swift"]
    models = ["deepjit", "cc2vec", "simcom", "lapredict", "tlel", "jitline", "la", "lr"]
    dataset = ["gerrit", "go", "platform", "jdt", "qt", "openstack"]
    modes = ["local", "remote"]

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("-debug", action="store_true", help="Turn on system debug print")
    common_parser.add_argument("-log_to_file", action="store_true", help="Logging to file instead of stdout")
    common_parser.add_argument("-dg_save_folder", type=str, required=True, help="")
    common_parser.add_argument("-mode", type=str, default="local", help="Mode of extractor", choices=modes)
    common_parser.add_argument("-repo_name", type=str, required=True, help="Repo name")
    common_parser.add_argument("-repo_owner", type=str, default="", help="Repo owner name")
    common_parser.add_argument("-repo_path", type=str, default="", help="Path to git repository")
    common_parser.add_argument("-repo_language", type=str, default="", choices=available_languages, help="Main language of repo")

    mining_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    mining_parser.set_defaults(func=mining)
    mining_parser.add_argument("-pyszz_path", type=str, default="", help="Path to pyszz")

    inferencing_parser = argparse.ArgumentParser(parents=[common_parser], add_help=False)
    inferencing_parser.set_defaults(func=inferencing)
    inferencing_parser.add_argument("-models", nargs="+", type=str, default=[], choices=models, help="List of deep learning models")
    inferencing_parser.add_argument("-dataset",type=str,default="openstack",choices=dataset,help="Dataset's name",)
    inferencing_parser.add_argument("-cross", action="store_true", help="Cross project")
    inferencing_parser.add_argument("-uncommit", action="store_true", help="Include uncommit in list when using -top",)
    inferencing_parser.add_argument("-device", type=str, default="cpu", help="Eg: cpu, cuda, cuda:1")
    inferencing_parser.add_argument("-sort", action="store_true", help="Sort output of model by predict score")
    inferencing_parser.add_argument("-vsc", action="store_true", help="Output for vsc")
    inferencing_parser.add_argument("-threshold", type=float, default=0.5, help="Threshold for warning")
    inferencing_parser.add_argument("-no_warning", action="store_true", help="Supress output warning")
    inferencing_parser.add_argument("-commit_hash", nargs="+", type=str, default=[], help="List of commit hashes")
    inferencing_parser.add_argument("-top", type=int, default=0, help="Number of top commits")

    parser = argparse.ArgumentParser(prog="DefectGuard", description="A tool for mining, training, evaluating for Just-in-Time Defect Prediction")
    parser.add_argument("-version", action="version", version="%(prog)s " + __version__)
    subparsers = parser.add_subparsers()
    subparsers.add_parser('mining', parents=[mining_parser], help='Mining Function')
    subparsers.add_parser('inferencing', parents=[inferencing_parser], help='Inferencing Function')

    options = parser.parse_args(args)

    if not options.debug:
        logger.disable()

    if options.log_to_file:
        # Create a folder named 'logs' if it doesn't exist
        if not os.path.exists(f"{SRC_PATH}/logs"):
            os.makedirs(f"{SRC_PATH}/logs")

        # Define a file to log IceCream output
        log_file_path = os.path.join(f"{SRC_PATH}/logs", "logs.log")

        # Replace logging configuration with IceCream configuration
        logger.configureOutput(
            prefix=f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | ',
            outputFunction=lambda x: open(log_file_path, "a").write(x + "\n"),
        )

    if not hasattr(options, 'func'):
        parser.print_help()
        exit(1)
    options.func(options)
    
    return options