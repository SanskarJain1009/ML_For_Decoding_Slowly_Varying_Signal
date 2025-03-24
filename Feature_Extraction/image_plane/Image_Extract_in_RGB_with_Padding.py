import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import stft
import math

open_flag = 0
close_flag = 0
image_index = 0
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
	DF_temp = DF
	for i in range(0, DF.shape[0]):
		upperLimit_DF = 0
		if(DF.iloc[i, 16]):
			DF_temp = DF.iloc[j:i, :16]
			#print(f"cond1 Before DF_temp_size = {DF_temp.shape}")
			r = DF_temp.shape[0]
			f = 512
			upperLimit_DF = math.floor(r/f) * f
			DF_temp = DF_temp.iloc[:upperLimit_DF, :]
			#print(f"cond1 After DF_temp_size = {DF_temp.shape}")
			j = i + 1 
			idx = idx + 1
			flag = 1
			open_flag = 1

		if(DF.iloc[i, 17]):
			#print(i)
			DF_temp = DF.iloc[j:i, :16]
			r = DF_temp.shape[0]
			f = 512
			upperLimit_DF = math.floor(r/f) * f
			DF_temp = DF_temp.iloc[:upperLimit_DF, :]
			j = i + 1 
			idx = idx + 1
			flag = 1
			close_flag = 1
	
		if(flag == 1):
			flag = 0

			for x in range(0, 16):
				mean_DF = (DF_temp.iloc[:, x]).mean()
				for y in range(0, upperLimit_DF):
					DF_temp.iat[y,x] = DF_temp.iat[y,x] - mean_DF

			for x in range (13, 16):
				f_DF, t_DF, Z_DF = stft(DF_temp.iloc[:, x], f, nperseg = 512)
				Z_DF = (np.square(np.abs(Z_DF)))
				plt.contourf(t_DF, f_DF, 10*(np.log10(Z_DF)))
				plt.ylim(8, 13)
				plt.yticks([8 , 13])
				if(x == 13):
					if(open_flag == 1):
						plt.savefig(f"train_o1/open/open.{image_index}.png")
						plt.close()
					if(close_flag == 1):
						plt.savefig(f"train_o1/close/close.{image_index}.png")
						plt.close()
				if(x == 14):
					if(open_flag == 1):
						plt.savefig(f"train_oz/open/open.{image_index}.png")
						plt.close()
					if(close_flag == 1):
						plt.savefig(f"train_oz/close/close.{image_index}.png")
						plt.close()
				if(x == 15):
					if(open_flag == 1):
						plt.savefig(f"train_o2/open/open.{image_index}.png")
						plt.close()
						open_flag = 0
					if(close_flag == 1):
						plt.savefig(f"train_o2/close/close.{image_index}.png")
						plt.close()
						close_flag = 0
						image_index = image_index + 1
			
			plt.close()	
	print(idx)
	
