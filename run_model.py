import os
import argparse
import logging
import time
import shutil

from utils.image_utils import Params, handle_dataset_processing
from utils.config_utils import get_config
from utils.data_analyzer import get_grams_info, get_images_mean
from models.cnn_rnn_ctc import CNN_RNN_CTC
from models.rnn_ctc import RNN_CTC
from models.encoder_decoder import Encoder_Decoder
from models.ngrams import run_ngrams
from utils.model_analyzer import run_model_analyzer

logging.basicConfig(level=logging.INFO)

def create_and_run_model(config_name, eval_path=None, infer=False):
    if eval_path is not None and infer:
        raise Exception("Both infer_path and eval_path are set. But cannot infer and evaluate at the same time.")

    mappings = {'CNN_RNN_CTC': CNN_RNN_CTC, 'RNN_CTC': RNN_CTC, 'Encoder_Decoder': Encoder_Decoder}
 
    config = get_config(config_name)

    TRAIN = eval_path is None and not infer    # indicates whether we are in training mode or not

    if not infer:
        _setup_logging(config, config.restore_previous_model or not TRAIN)

    if TRAIN:
        if config.restore_previous_model:
            logging.info("-"*70 + "\n" + " "*20 + "Continuing with previous model\n" + "-"*70)

        else:
            logging.info("-"*70 + "\n" + " "*25 + "Starting a new model\n" + "-"*70)
            logging.info("Logging model parameters:") 
            try:
                attr = vars(config)
            except:
                attr = config._asdict()
            for k, v in attr.items():
                if not k.startswith('__'):
                    logging.info(str(k) + ": " + str(v))
    
    elif eval_path is not None:
        logging.info("\n" + "-"*50 + "\nEvaluating ({})\n".format(config.decoder_type.replace("_", " ").capitalize()) + "-"*50)

    if TRAIN:
        logging.info("\n" + "-"*50 + "\nInitializing Model and Building Graph\n" + "-"*50)
        tic = time.time();

    model = mappings[config.model](config, eval_path, infer);

    if TRAIN:
        toc = time.time()
        logging.info("Time to initialize model (including to load data): %f secs\n" % (toc-tic))
        
    if TRAIN or eval_path is not None:
        model()

    return model, config

def _setup_logging(config, use_previous_log):
    name = config.model
    save_dir = config.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not use_previous_log and os.path.isfile(os.path.join(save_dir, name + ".log")):
        os.remove(os.path.join(save_dir, name + ".log"))

    file_handler = logging.FileHandler(os.path.join(save_dir, name + ".log"))
    logging.getLogger().addHandler(file_handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Image preprocessing
    parser.add_argument("--image_preprocessing", help="do image preprocessing", action="store_true")
    parser.add_argument("--preprocess_path", help="path to images to process", type=str)
    parser.add_argument("--processed_path", help="where to save images", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--size", help="size of dataset to prepare", type=int)
    parser.add_argument("--extract_red_component", help="extract only the red component of the images", action="store_true", default=False)
    parser.add_argument("--horizontal_segmentation", help="do horizontal segmentation", action="store_true", default=False)
    parser.add_argument("--vertical_segmentation", help="do vertical segmentation", action="store_true", default=False)
    parser.add_argument("--processed_image_type", help="format to save the processed images in", type=str, default=".jpg")
    parser.add_argument("--correct_skew", help="correct skew in images", action="store_true", default=False)
    parser.add_argument("--remove_white_columns", help="trim all white columns", action="store_true", default=False)
    parser.add_argument("--remove_white_rows", help="trim all white rows", action="store_true", default=False)

    # Dataset analysis
    parser.add_argument("--data_analyzer", help="run the data analyzer", action="store_true")
    parser.add_argument("--images_path", help="path to images for data_analyzer", type=str)
    parser.add_argument("--gt_path", help="path to ground truths", type=str)

    # Train, evaluate, infer or analyze - all require the config file
    parser.add_argument("--config_name", help="the config file to use", type=str)
    parser.add_argument("--infer_path", help="path to images to run the model on", type=str)
    parser.add_argument("--eval_path", help="path to the evaluation dataset", type=str)
    parser.add_argument("--analyze_path", help="path to the dataset to analyze the model on", type=str)

    # Language model
    parser.add_argument("--lm", help="run the language model", action="store_true")
    parser.add_argument("--corpus", help="path to the corpus to train the language model on", type=str)
    parser.add_argument("--ngrams", help="number of grams to use for language model", type=int, default=3)
    parser.add_argument("--lm_save_dir", help="directory to save the language model in", type=str, default="trained_models/LM/")

    args = parser.parse_args()

    if args.image_preprocessing:
        config = Params(args.extract_red_component,
                        args.horizontal_segmentation,
                        args.vertical_segmentation,
                        True,
                        args.processed_image_type,
                        args.correct_skew,
                        args.remove_white_columns,
                        args.remove_white_rows)
        
        assert args.preprocess_path != args.processed_path # take precaution
        if os.path.isdir(args.processed_path):
            shutil.rmtree(args.processed_path)
        os.mkdir(args.processed_path)

        handle_dataset_processing(args.preprocess_path, args.processed_path, size=args.size, params=config, verbose=args.verbose)

    elif args.config_name is not None:
        model, config = create_and_run_model(args.config_name,
                                             eval_path=args.eval_path,
                                             infer=args.infer_path is not None or args.analyze_path is not None)
        
        if args.analyze_path is not None:
            run_model_analyzer(config, args.analyze_path, model)

        elif args.infer_path is not None:
            tic = time.time()
            out, urdu_out = model(args.infer_path)
            print(out)
            print(urdu_out)
            toc = time.time()
            print("Inferring time: ", toc-tic)

    elif args.lm:
        if not os.path.exists(args.lm_save_dir):
            os.makedirs(args.lm_save_dir)

        run_ngrams(args.corpus, args.lm_save_dir, args.ngrams)

    elif args.data_analyzer:
        if args.images_path is not None:
            get_images_mean(args.images_path)

        if args.gt_path is not None:
            get_grams_info(args.gt_path)
