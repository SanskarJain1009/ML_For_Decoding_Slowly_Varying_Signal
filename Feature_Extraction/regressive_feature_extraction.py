#Importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import stft
import math

#Declaring list to store Alpha Band Features
arr = []

#Declaring list to store Alpha Band Features in Decibel Sclae
arr_log = []

#Declaring flags 
open_flag = 0
close_flag = 0


for p in range(0, 21):
	if(p == 7):
		continue
	if(p<10):
		file_name = f"subject_0{p}"
		DF = pd.read_csv(f"{file_name}.csv")
	else:
		file_name = f"subject_{p}"
		DF = pd.read_csv(f"{file_name}.csv")
	j=0
	idx = 0
	flag = 0
	if(idx == 1):
		break
	DF_temp = DF
	for i in range(0, DF.shape[0]):
		upperLimit_DF = 0
		if(DF.iloc[i, 16]):
			DF_temp = DF.iloc[j:i, :16]
			r = DF_temp.shape[0]
			f = 512
			upperLimit_DF = math.floor(r/f) * f
			DF_temp = DF_temp.iloc[:upperLimit_DF, :]
			j = i  
			idx = idx + 1
			flag = 1
			open_flag = 1
		if(DF.iloc[i, 17]):
			DF_temp = DF.iloc[j:i, :16]
			r = DF_temp.shape[0]
			f = 512
			upperLimit_DF = math.floor(r/f) * f
			DF_temp = DF_temp.iloc[:upperLimit_DF, :]
			j = i 
			idx = idx + 1
			flag = 1
			close_flag = 1
	
		if(flag == 1):
			flag = 0
			for x in range(13, 16):
				mean_DF = (DF_temp.iloc[:, x]).mean()
				for y in range(0, upperLimit_DF):
					DF_temp.iat[y,x] = DF_temp.iat[y,x] - mean_DF
	
			for x in range (13, 16):
				f_DF, t_DF, Z_DF = stft(DF_temp.iloc[:, x], f, nperseg = 512)
				Z_DF = (np.square(np.abs(Z_DF)))
				Z_DF_log = 10 * np.log10(Z_DF)
				for index in range(0,t_DF.shape[0]):
					row_list = []
					row_list_log = []
					sum = 0
					for y in range(8, 14):
						row_list.append(Z_DF[y][index])
						row_list_log.append(Z_DF_log[y][index])
						sum = sum + Z_DF[y][index]
					row_list.append(sum/6)
					row_list_log.append(10 * np.log10(sum/6))
					if(open_flag == 1):
						row_list.append(0)
						row_list_log.append(0)

					if(close_flag == 1):
						row_list.append(1)
						row_list_log.append(1)

					arr.append(row_list)
					arr_log.append(row_list_log)
		open_flag = 0
		close_flag = 0

feature_data_frame = pd.DataFrame(arr, columns = ['8Hz', '9Hz', '10Hz', '11Hz', '12Hz', '13Hz', 'Mean', 'Label'])
feature_data_frame_log = pd.DataFrame(arr_log, columns = ['8Hz', '9Hz', '10Hz', '11Hz', '12Hz', '13Hz', 'Mean', 'Label'])
