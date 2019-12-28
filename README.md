This repository implements three deep learning architectures using TensorFlow.

1. RNN-CTC Architectures
2. CNN-RNN-CTC Architecture
3. Attention-Based Encoder-Decoder Architecture 

## Directory Structure

```
configs								contains .json config files for the models

data						
	*/data_folder_name				folder name needs to be set in the config file
    	images						images in .jpg, .png or .jpeg format
    	labels
    		gt_char.csv				labels to run a character-level model
			lt_char.csv				contains the ids to character map
    		gt_lig.csv				labels to run a ligature-level model
    		lt_lig.csv				contains the ids to ligature map

debugger
	custom_filters.py				used for implementing filters for tfdebugger

fonts
	jameel_noori_nastaleeq.ttf      Urdu font

gui									implelements a graphical user interface
	configs							contains .json config files used by the GUI
	abtform.py						layout of the "About" dialog box
	docsform.py						layout of the "Documentation" dialog box
	termsform.py					layout of the "Terms and Conditions" dialog box
	run_gui.py						starts the GUI and handles generated events
								    (menu selections and button presses)
	uhw_rc.py						resource file containing fonts and icons used
	uiform.py						layout of the main window; made with Qt Designer
	waitingspinnerwidget.py			custom Qt widget for "loading" spinner icons**   

models
	model_class.py					parent class of all models 
	encoder.py						uses RNNs to encode inputs
	rnn_ctc.py						implements the RNN-CTC model
	cnn.py							implements a convolutional network
	cnn_rnn_ctc.py					implements the CNN_RNN_CTC model
	decoder.py						implements an attention-based decoder
	encoder_decoder.py				implements the Encoder_Decoder model
	lm.py							implements prefix beam search
	ngrams.py**						implements an n-grams language model
	helpers.py						implements some helper functions

trained_models						contains parameters of previously trained models
	*/save_path						needs to set correctly in the config file
		model_name					contains the trained model's parameters
			checkpoint				used by TensorFlow's model restoring function
			model_name.data-*		used by TensorFlow's model restoring function
			model_name.index		used by TensorFlow's model restoring function
			model_name.meta			used by TensorFlow's model restoring function
			constants.pkl			contains some constants needed during inference
			histories.pkl			contains loss history, accuracy history etc
			split_indices.pkl		stores the train/val split during training		
		Tensorboard					contains files for Tensorboard visualizations
		analyzer_*.csv				outputs of model_analyzer (* is the decoder type)
		model_name.log				the log file for training & evaluation processes
		plots.png					training polts (loss vs epoch, grad norm vs epoch)
		attention_images****		shows the attention mechanism on an image
		infer_alignment.mp4****		combines attention images into a video
		infer_alignment.png****		shows the alignments for an image

utils
	accuracy_metrics.py				implements accuracy calculating functions
	config_utils.py					used to read .json config files
	data_utils.py					loads and processes data from the data folder
	image_utils.py					implements preprocessing functions for images
	model_analyzer.py				used for analyzing trained models
	data_analyzer.py				calculates attributes of the dataset
	helpers.py						some helper functions
	
dump								discarded code; retained because of emotional
									attachments 

launch_gui.py						launches the graphical user interface
requirements.txt					packages required to run code in this repository
run_model.py						provides a command line interface for the code
zip_files.sh						script to zip files in this repository
README.md							what you are reading right now
---------------------------------------------------------------------------------------
**   This has been taken from: https://github.com/z3ntu/QtWaitingSpinner
***  The entire file is taken (with slight modifications) from:
     https://github.com/giovannirescia/PLN-2015/tree/practico4/languagemodeling
**** Only in the case of Encoder-Decoder models
```

For all commands listed below, we assume (unless explicitly stated otherwise) that all images reside in a **single** folder i.e. the images are not divided into subdirectories.

## Install Required Packages

This code has been tested on python 3.6.3. The required packages can be installed through:

```
pip install -r requirements.txt
```

We recommend setting up a virtual environment for this code.

## Image Preprocessing

```shell
python run_model.py --image_preprocessing --preprocess_path path_to_images_to_preprocess --processed_path directory_to_store_processed_images_in
```

The following options for preprocessing may be specified:

```c
--verbose				  // shows the image being preprocessed
--extract_red_component	  // extract only the red component from the images
--horizontal_segmentation // horizontally segment the images
--vertical_segmentation   // vertically segment the images
--correct_skew			  // correct the skew (if present) in images
--remove_white_columns	  // remove any white columns in the image
--remove_white_rows		  // remove any white rows in the image
```

The preprocessing steps are performed in the same order above as listed above (i.e. if both horizontal segmentation and vertical segmentation are specified, then the former will be done before the latter).

Furthermore, the following options may also be provided:

```c
--size size			        // number of images to process; default is None
--processed_image_type type // format to save the processed images in; default is .jpg
```

## Dataset Analysis

To calculate the maximum, minimum and average heights and widths of images in a folder, do:

```shell
python run_model.py --data_analyzer --images_path path_to_images
```

To calculate the number of unigrams, bigrams and trigrams, do:

```shell
python run_model.py --data_analyzer --gt_path path_to_groundtruth
```

Here, path_to_groundtruth is the path to a **single** text file. Each line (lines are assumed to be separated with the new line character) is treated separately. For now, this function only supports Urdu text files.

## The Data Folder

For training, the path to the directory that contains the dataset must be provided. The data_utils.py file loads the data. It assumes the following structure of this directory:

```
data_folder
	images					   contains images in either .jpg, .jpeg or .png format
	labels
		gt_char.csv				labels to run a character-level model
		lt_char.csv				contains the ids to character map
    	gt_lig.csv				labels to run a ligature-level model
    	lt_lig.csv				contains the ids to ligature map
```

The first two files are used to train a character-level model and the last two files are used to train a ligature level model. Which scheme to use for training the models can be specified in the config file.

Each row in the gt_*.csv files contains the labels for an image. For example:

```
0001_0251_01.txt,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 11, 1, 12, 13, 5, 7, 14, 15, 15, 3, 16, 17, 18, 19, 15, 20, 14, 21, 22, 8, 9, 23, 14, 24, 25, 26, 5, 26, 27]
```

where 0001_0251_01.txt has a corresponding image 0001_0251_01.jpg (or any of the supported formats) in the images folder.

Each row in the lt_*.csv files is of the following format:

```
character,character_id
```

The path of the data_folder needs to set in the config file.

Such a structure of the data folder makes the code more flexible to different id assigning strategies.

**Note:** Due to some copyright reasons, we cannot include the Urdu handwriting dataset in this repository. The dataset, however, may be obtained by contacting the [Center for Language Engineering, Lahore](http://www.cle.org.pk/).

## Training

A model can be trained using the following command:

```sh
python run_model.py --config_name path_to_config_file
```

## Evaluation

A model may be evaluated on a (test) dataset using the following command:

```shell
python run_model.py --config_name path_to_config_file --eval_path path_to_eval_folder
```

The structure of the eval_folder should be the same as that described in the Data Folder section above.

## Inferring

For inferring use:

```shell
python run_model.py --config_name path_to_config_file --infer_path path_to_image
```

where path_to_image is the path to an image on which the model should make predictions on. The path_to_image may be the path to a single image or to a folder containing images.

## Analyzing

For analyzing models use:

```shell
python run_model.py --config_name path_to_config_file --analyze_path path_to_analyze_folder
```

The structure of the analyze_folder should be the same as that described in the Data Folder section above.

## Language Model

For training a language model on a corpus use:

```shell
python run_model.py --lm --corpus path_to_corpus --ngrams num_grams --lm_save_dir save_dir
```

where path_to_corpus is the path to a **single** text file that is to be used to train the language model, num_grams are the number of grams to be used and save_dir is the path to the folder to save the trained language model. 

## Config Files

For now, the code only supports .json files. Some entries must always be included in the config file regardless of the architecture of the model, while other entries are specific to certain architectures only.

### Type 1: Must Always Be Included

| Name                     | Type           | Description                                                  |
| ------------------------ | -------------- | ------------------------------------------------------------ |
| model                    | str            | Architecture to use. Options: RNN_CTC, CNN_RNN_CTC, Encoder_Decoder |
| save_dir                 | str            | Path to save/restore the model to/from                       |
| data_folder              | str            | Path to the data folder                                      |
| use_gpu                  | bool           | Use a GPU if available                                       |
| debug_mode               | bool           | Whether to use tfdebugger                                    |
| verbose                  | bool           | Whether to create TensorBoard visualizations                 |
| max_outputs              | int            | Number of outputs to save for TensorBoard                    |
| save_alignments          | bool           | Whether to create alignment images for Encoder-Decoder model. Always False for other architectures. Also, only True for Encoder-Decoder models when the decoder_type is greedy_search. |
| char_or_lig              | str            | Whether to use a character level or a ligature level model. Options: char or lig |
| use_dynamic_lengths      | bool           | Whether to handle image lengths dynamically (only works properly under certain conditions; see the code for details) |
| use_teacher_forcing      | bool           | Whether to use teacher forcing or not. Essentially acts as flag that distinguishes the Encoder_Decoder model (True in this case) from the CTC ones (False in this case). |
| use_sparse_labels        | bool           | Whether to feed labels to the TensorFlow graph as sparse tensors. Should be True for RNN_CTC and CNN_RNN_CTC models |
| save_best                | bool           | Whether to save the model that achieves the lowest validation loss during training |
| restore_previous_model   | bool           | Whether to restore the previous model (i.e. continue training) |
| restore_best_model       | bool           | Whether to restore the best model at the end of every epoch  |
| restore_best_model_range | float          | The minimum difference needed between the loss of the best model and the model at the end of an epoch in order for the former to be restored; only required if restore_best_model is set to True |
| early_stopping           | bool           | Whether to use early stopping                                |
| stop_after_num_epochs    | int            | The number of epochs to wait before early stopping; only required if early stopping is set to True |
| num_train                | int            | Size of the training data to load. None indicates all        |
| batch_size               | int            | The batch size                                               |
| buckets                  | int            | The number of buckets to divide the input images in (depending upon their widths) |
| num_epochs               | int            | The number of epochs to run the model for                    |
| print_every              | int            | The number of iterations after which the loss and accuracy should be printed on screen |
| image_size               | list of size 2 | The width and height respectively that the input images need to have. The width may be set to None in which case the height will be normalized and the width changed so as to maintain the aspect ratio. Note that width must be set to None if more than one buckets are to be used. |
| image_width_range        | int            | Only images having widths within image_width_range of the width specified in image_size are loaded. Setting this option to None loads all images Must be set to None if the width in image_size is set to None. |
| flip_image               | bool           | Whether to flip image horizontally                           |
| optimizer                | str            | The optimizer to use. Options: adam, sgd                     |
| lr                       | float          | The learning rate                                            |
| anneal_lr                | bool           | Whether to decay the learning rate                           |
| anneal_lr_every          | int            | The number of training steps after which to decay the lr. Only required if anneal_lr is set to True. |
| anneal_lr_rate           | float          | The rate at which to decay lr. Only required if anneal_lr is set to True. |
| dropout                  | float          | The dropout to use for the RNNs                              |
| max_grad_norm            | float          | The maximum allowable gradient. Gradients are scaled down if a gradient norm exceeds this value. Disabled with None. |
| l2_regularizer_scale     | float          | The amount of L2 regularization to use                       |

### Type 2: Specific to Convolutional Layers

| Name                    | Type                                   | Description                                                  |
| ----------------------- | -------------------------------------- | ------------------------------------------------------------ |
| cnn_num_layers          | int                                    | The number of convolutional layers to use                    |
| cnn_num_residual_layers | int                                    | The number of residual connections between layers            |
| cnn_activation          | int                                    | The activation function to use. Options: relu, leaky_relu    |
| cnn_num_filters         | list; each element type is int         | The number of filters to use at each layer                   |
| cnn_filter_strides      | list; each element is a list of size 2 | The first element in each inner list is the horizontal stride and the second element is the vertical stride (w.r.t to the image) |
| cnn_paddings            | list; each element type is str         | Whether to pad inputs at each layer. Options for each element: "VALID" (no padding), "SAME" (padding) |
| pool_sizes              | list; each element is a list of size 2 | The elements in the inner lists are the height and width of the pooling 'filter' at each layer |
| pool_strides            | list; each element is a list of size 2 | The first element in each inner list is the horizontal stride and the second element is the vertical stride (w.r.t to the image) |
| pool_paddings           | list; each element type is str         | Whether to pad inputs at each layer. Options for each element: "VALID" (no padding), "SAME" (padding) |
| do_batch_norm           | list; each element type is bool        | Whether to apply batch normalization at each layer           |

### Type 2: Specific to RNN-CTC and CNN-RNN-CTC Architectures

| Name                    | Type  | Description                                                  |
| ----------------------- | ----- | ------------------------------------------------------------ |
| rnn_num_layers          | int   | The number of RNN layers                                     |
| rnn_unit_type           | str   | The type of RNN to use. Options: lstm, gru, layer_norm_lstm  |
| rnn_type                | str   | Either unidirectional (uni) or bidirectional (bi)            |
| rnn_num_residual_layers | int   | The number of residual connections between RNN layers        |
| rnn_units               | int   | The number of units of the RNNs                              |
| decoder_type            | str   | Type of decoder to use. Options: greedy_search, beam_search, prefix_beam_search |
| beam_width              | int   | The beam width. Only needed if decoder_type is beam search or prefix_beam_search. |
| lm_path                 | str   | Path to the folder containing the language model. The file must be labeled ngrams_k.pkl where k is the number of grams being used. Only needed if decoder_type is prefix_beam_search. |
| ngrams                  | int   | The number of grams to use for the language model. Only needed if decoder_type is prefix_beam_search. |
| alpha                   | float | Weight of the language model. Only needed if decoder_type is prefix_beam_search |
| beta                    | float | How much to penalize CTC sequences with low ligature count. Only needed if decoder_type is prefix_beam_search. |
| discard_probability     | float | Minimum probability that an id must have at some given time step for it to be considered for decoding. Only needed if decoder_type is prefix_beam_search. |

### Type 3: Specific to Encoder-Decoder Architectures

| Name                           | Type  | Description                                                  |
| ------------------------------ | ----- | ------------------------------------------------------------ |
| extract_features               | bool  | Whether to use convolutional layers at the start for extracting features |
| encoder_num_layers             | int   | The number of RNN layers in the encoder                      |
| encoder_unit_type              | str   | The type of RNN to use for the encoder. Options: lstm, gru, layer_norm_lstm |
| encoder_type                   | str   | Use unidirectional (uni) or bidirectional (bi) RNN for encoder |
| encoder_num_residual_layers    | int   | The number of residual connections between RNN layers in the encoder |
| encoder_units                  | int   | The number of units of the RNNs in the encoder               |
| embed_size                     | int   | Size of the embedding matrix to use                          |
| pass_hidden_state              | bool  | Whether to initialize the decoder's state with the final encoder state |
| encoder_num_layers             | int   | The number of RNN layers in the decoder                      |
| encoder_unit_type              | str   | The type of RNN to use for the decoder. Options: lstm, gru, layer_norm_lstm |
| encoder_type                   | str   | Use unidirectional (uni) or bidirectional (bi) for the decoder |
| encoder_num_residual_layers    | int   | The number of residual connections between RNN layers in the decoder |
| encoder_units                  | int   | The number of units of the RNNs in the decoder               |
| do_scheduled_sampling          | bool  | Whether to do scheduled sampling                             |
| initial_not_sampling_prob      | bool  | Initial probability of using teacher forcing. Only needed if do_scheduled_sampling is set to True. |
| anneal_not_sampling_prob       | bool  | Decay the probability of using teacher forcing. Only needed if do_scheduled_sampling is set to True. |
| anneal_not_sampling_prob_every | int   | The number of training steps after which to decay the probability of using teacher forcing. Only required if anneal_not_sampling_probability is set to True. |
| anneal_not_sampling_prob_rate  | float | The rate at which to decay anneal_not_sampling_prob. Only required if anneal_not_sampling_probability is set to True. |
| decoder_type                   | str   | Type of decoder to use. Options: greedy_search, beam_search  |
| beam_width                     | int   | The beam width. Only needed if decoder_type is beam search.  |
| coverage_penalty_weight        | float | Weight to penalize the coverage of source sentence           |
| length_penalty_weight          | float | Weight to penalize length of output sequence                 |

The following rules must be observed when setting fields in config files:

1. All values must be written within inverted commas "".
2. For lists, no outer brackets should be used. Also, a space must be left between each element. For e.g. "1, 4, 3" is a valid entry.
3. For lists of lists, no outer brackets should be used. Inner lists should be enclosed within parenthesis. No space should be left between elements in the inner lists. However, a space must be added between the inner lists themselves. For e.g. "(2,2), (1,2), (1,2), (1,2), (1,2), (1,2), (1,1)" is a valid entry.

See the config files in the config folder for examples.

## Graphical User Interface

To launch the graphical user interface run:

```shell
python launch_gui.py
```

## Adapting to Other Languages

It is easy to adapt the code to support other languages. Only the class UrduTextReader and the functions convert_to_urdu and get_lookup_table in the data_utils.py file need to be changed.

## Changing the Structure of the Data Folder

The data_utils.py is the only file that depends on the structure of the data folder. New structures can thus be supported by modifying this file.