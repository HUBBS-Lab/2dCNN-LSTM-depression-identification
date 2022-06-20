# Pytorch implementation of 'Toward Knowledge-Driven Speech-Based Models of Depression: Leveraging Spectrotemporal Variations in Speech Vowels'
To run this repo, you will first need to download the processed feature files from this [Google Drive link](https://drive.google.com/drive/folders/1SWXtQ9zeSN8L2XOnY18SVKYVJ7tclIqa?usp=sharing). These files were not uploaded to GitHub due to the file size restriction. Unzip the file if needed and put them in the correct folder.

You only need to download the "speaker_LSTM_conv5" and "speaker_LSTM_pool4" if you just want to run the LSTM part.

Besides these files, you need to prepare four files and put them in the "data" folder:
1. the 'train_split_Depression_AVEC2017.csv' from the DAIC-WOZ dataset
2. the 'dev_split_Depression_AVEC2017.csv' from the DAIC-WOZ dataset
3. train_labels.pickle: this file should be a dictionary with the format of 
   ```javascript
    { "participant_id_1" : PHQ-8 binary label; "participant_id_2" : PHQ-8 binary label; ...}
   ```
   The participant_id are from the training set. Please note that the binary label for participant 409 should be 1 instead of 0 as shown in 'train_split_Depression_AVEC2017.csv'
4. dev_labels.pickle: same file structure with train_labels.pickle but with participants from the dev set

Train the models do not require a [FAVE aligner](https://github-wiki-see.page/m/JoFrhwld/FAVE/wiki/FAVE-align) setup. However, if you would like to start from scratch or run some of the codes in the "preprocessing" folder, then the FAVE aligner is needed
## Content in Google Drive 
#### data
- train_spec_vowel_v2.pickle: the log mel-spectrogram features stored in a dictionary with the format of 
  ```javascript
  { "participant_id_1" : 
      {"A":[spectrogram1, spectrogram2, ...];  
       "E":[spectrogram1, spectrogram2, ...];
       "I":[spectrogram1, spectrogram2, ...];
       "O":[spectrogram1, spectrogram2, ...];
       "U":[spectrogram1, spectrogram2, ...];
       "N":[spectrogram1, spectrogram2, ...]};
     "participant_id_2" : ...
     ...
   }
  ```
  The 'N' means 'not a vowel'. This file is used to train the vowel CNN
- dev_spec_vowel_v2.pickle: file with a similar structure of train_spec_vowel_v2 and is used to evaluate the vowel CNN
- train_spec_vowel_v3.pickle: the log mel-spectrogram features for each participant with the sequential information reserved
  ```javascript
  { "participant_id_1" : [spectrogram1, spectrogram2, ...];  
    "participant_id_2" : [spectrogram1, spectrogram2, ...];
    ...
   }
- dev_spec_vowel_v3.pickle: similar to train_spec_vowel_v3 but includes features for the dev set
#### speaker_LSTM_conv5
- speaker_embeddings_conv5.zip: the conv5 embedding from the vowel CNN 
#### speaker_LSTM_pool4
- speaker_embeddings_pool4.zip: the pool4 embedding from the vowel CNN 

## In this GitHub
#### preprocessing
- crop_audio.py: split the whole interview into utterances
- feature_extract_spectrogram.py: generate the data/dev_spec_vowel_v3.pickle and data/train_spec_vowel_v3.pickle
- vowel_feature_extraction.py: generate the data/dev_spec_vowel_v2.pickle and data/train_spec_vowel_v2.pickle, running this code requires the output of FAVE aligner for each utterance
#### vowel_classification_cnn
- model.py: the CNN strucutre we used in our experiment
- train.py and vowel_classification_cnn/validate.py: train  and validate the vowel CNN 
- vowel_cnn.pth: the stored model parameters for the trained cnn
#### speaker_LSTM_pool4 and speaker_LSTM_conv5
These two folders share similar file structures
- extract_feature_space.py: extract the CNN embeddings from spectrograms and the output is speaker_LSTM_conv5/speaker_embeddings_conv5 or speaker_LSTM_pool4/speaker_embeddings_pool4
- model_LSTM.py: the LSTM structure we used in our experiment
- model.py: the CNN structure, same as vowel_classification_cnn
- train.py and validate.py: the train and validation of our LSTM model
- saved_LSTM_pool4 and saved_LSTM_conv5: store our trained LSTM models

## Other settings
For a complete environmental setting we used, please check the 'env.txt'.

**This repository has been tested on the platform as in 'env.txt'. Configurations (e.g batch size, learning rate) may need to be changed on different platforms.**
