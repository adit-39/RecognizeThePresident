import scipy.io.wavfile as wvf
import scipy.cluster.vq as sp
import numpy as np
import os
import re
import sys
from features import mfcc

def kmeans_Mfcc_mod_train():
	"""
	Obtain the MFCC coefficients from a wav file as a numpy matrix, then cluster the
	attributes into a single numeric value and print into a file given by opfile
	"""
	rate,sig = wvf.read("all_trng.wav")
	mfcc_feat = mfcc(sig,rate)
	global codebook
	codebook = sp.kmeans(mfcc_feat, 16)[0]
	wavfiles = ["new_obama_trng.wav","other_trng.wav"]
	for wavfile in wavfiles:
		final = []
		rate,sig = wvf.read(wavfile)
		mfcc_feat = mfcc(sig,rate)
		data = sp.vq(mfcc_feat,codebook)
		for i in data[0]:
			final.append(i)
		f = open(wavfile.split(".")[0]+"_vq.txt","w")
		for i in final:
			f.write(str(i)+"\n")
		f.close()
	return codebook

if __name__=="__main__":
	kmeans_Mfcc_mod_train()
