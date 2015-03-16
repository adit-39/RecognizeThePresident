#! /usr/bin/python
"""
Read the url from django input and get the audio from the video. Convert this 
into a wav file and then obtain MFCC coefficients and apply k-means clustering
 followed by vector quantization to build numerical representation of audio for
the HMMs
"""
import pafy
import scipy.io.wavfile as wvf
import scipy.cluster.vq as sp
import numpy as np
import os
import re
import sys
from features import mfcc

def get_audio_from_video(url,wavfile="temp"):
	"""
	Taking the YouTube URL as input, obtain the audio stream and convert it into
	wav format
	"""
	video = pafy.new(url)
	fname=""
	audiostreams = video.audiostreams
	for stream in audiostreams:
		if stream.extension == "m4a" and stream.bitrate == '128':
			fname = stream.download()
	if fname=="":
		for stream in audiostreams:
			if stream.extension == "m4a":
				fname = stream.download()
				break
	
	cmd = "ffmpeg -i "+re.escape(fname)+" "+re.escape(wavfile)+".wav"
	os.system(cmd)
# conversion of audio file format: ffmpeg -i audio.aac audio.wav

def kmeans_Mfcc(codebook,wavfile="temp.wav",opfile="temp_vq.txt"):
	"""
	Obtain the MFCC coefficients from a wav file as a numpy matrix, then cluster the
	attributes into a single numeric value and print into a file given by opfile
	"""
	rate,sig = wvf.read(wavfile)
	mfcc_feat = mfcc(sig,rate)
	#codebook = sp.kmeans(mfcc_feat, 8)[0]
	data = sp.vq(mfcc_feat,codebook)
	f = open(opfile,"w")
	for i in data[0]:
		f.write(str(i)+"\n")
	f.close()
	return opfile


if __name__=="__main__":
	get_audio_from_video(sys.argv[1],sys.argv[2])
	kmeans_Mfcc(sys.argv[2]+".wav",sys.argv[3])
