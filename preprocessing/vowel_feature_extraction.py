import librosa
# import librosa.display
import numpy as np
import pandas as pd
import os
import pickle
from scipy import stats
from collections import Counter
import csv


label_csv = pd.read_csv('../data/dev_split_Depression_AVEC2017.csv')
user_list = list(label_csv['Participant_ID'])


frame_size = int(16000*0.25)
overlap = 0.5

def extract_spec(s1, s2, y):
	y_tmp = y[s1: s2]
	melspec = librosa.feature.melspectrogram(y=y_tmp, sr=sr, n_fft=512, hop_length=128, n_mels=128, center=False)
	logmelspec = librosa.power_to_db(melspec)
	return logmelspec
	# return np.array([1, 2])

feature_dict = {}
for user in user_list:
	feature_dict[user] = {'A':[], 'E':[], 'I':[], 'O':[], 'U':[], 'N':[]}

	user_tmp = []
	idx = 0
	filepath = '../data/dev_audio_by_uttr/' + 'spk_'+ str(user) + '_uttr' + str(idx) + '.wav'
	filepath_vowel = '../data/dev_vowel_by_uttr/' + 'spk_'+ str(user) + '_uttr' + str(idx) + '.txt'
	while os.path.isfile(filepath):
		try:
			print(filepath_vowel)
			vowel_data = pd.read_csv(filepath_vowel, sep='\t')
			y, sr = librosa.load(filepath, sr=16000)
			# print(len(y))
			current_idx = 0
			# print(len(y))
			for row_idx, row in vowel_data.iterrows():
				vowel_tmp, start_tmp, end_tmp = row['vowel'][0], float(row['beg']), float(row['end'])
				start_idx, end_idx = int(start_tmp*16000), int(end_tmp*16000)
				if row_idx <= len(vowel_data) - 2:
					next_start_idx = int(float(vowel_data['beg'][row_idx+1]) * 16000)
					next_end_idx = int(float(vowel_data['end'][row_idx+1]) * 16000)
					next_vowel = vowel_data['vowel'][row_idx+1][0]
				else:
					next_start_idx = len(y)
					# next_end_idx = len(y)
				
				while (current_idx + frame_size) <= next_start_idx or current_idx < end_idx:
					seg_overlap = max(0, min(end_idx, current_idx + frame_size) - max(start_idx, current_idx))
					if seg_overlap >= int(frame_size*0.5) or seg_overlap >= (end_idx-start_idx)*0.8:
						# extract with vowel label
						logmelspec = extract_spec(current_idx, current_idx+frame_size, y)
						if logmelspec.shape[1] == 28:
							feature_dict[user][vowel_tmp].append(logmelspec)
						# print(current_idx, current_idx+frame_size, start_idx, end_idx, vowel_tmp)
					elif (current_idx + frame_size) <= next_start_idx:
						# extract without vowel lable
						logmelspec = extract_spec(current_idx, current_idx+frame_size, y)
						if logmelspec.shape[1] == 28:
							feature_dict[user]['N'].append(logmelspec)
						# print(current_idx, current_idx+frame_size, start_idx, end_idx, 'N')
					elif (current_idx + frame_size) > next_start_idx and (((current_idx + frame_size)-next_start_idx) >= int(frame_size*0.5) or ((current_idx + frame_size)-next_start_idx) >= (next_end_idx-next_start_idx)*0.8):
						logmelspec = extract_spec(current_idx, current_idx+frame_size, y)
						if logmelspec.shape[1] == 28:
							feature_dict[user][next_vowel].append(logmelspec)
						# print(current_idx, current_idx+frame_size, start_idx, end_idx, next_vowel)
					else:
						logmelspec = extract_spec(current_idx, current_idx+frame_size, y)
						if logmelspec.shape[1] == 28:
							feature_dict[user]['N'].append(logmelspec)
						# print(current_idx, current_idx+frame_size, start_idx, end_idx, 'N')
					current_idx = current_idx + int(frame_size*overlap)
		except:
			pass
		
		idx = idx+1
		filepath = '../data/dev_audio_by_uttr/' + 'spk_'+ str(user) + '_uttr' + str(idx) + '.wav'
		filepath_vowel = '../data/dev_vowel_by_uttr/' + 'spk_'+ str(user) + '_uttr' + str(idx) + '.txt'

        
with open('../data/dev_spec_vowel_v2.pickle', 'wb') as handle:
    pickle.dump(feature_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	
		




