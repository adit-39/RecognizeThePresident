"""
FINDING PRESIDENT OBAMA:
Train model, once trained, we will save that model and use it every time
Instead of computing the q-index, we are finding the most occurrences in 20
(which denotes a second's worth of data) and declaring that state as the state
of the HMM (which is in Obama, Others and silence)

"""

from myhmm_scaled import MyHmmScaled
from preprocessing import *
from create_training import *

def read_file(name):
	"""
	Function to take in a training or test file and compose a list of 10-member
	sequences for use later
	"""
	with open(name) as f:
		model=f.readlines()

	s=[]
	sequences=[]
	i=0;

	for word in model:
		i+=1
		word=word[:-1]
		s.append(word)
		if i %10 == 0 or i == (len(model)-1):
			sequences.append(s)
			s=[]

	return sequences


def train_machine(data,init_model_file):
    """
    Function to take in training data as a list of 10-member lists along with an
    initial model file and output a HMM machine trained with that data.
    """
    M = MyHmmScaled(init_model_file)
    M.forward_backward_multi_scaled(data)

    return(M)

def test(output_seq,M1,M2,obs_file):
    """
    Function to take in the test observation sequence and output "silent",
    "single" or "multi" based on which of the three machines gives the highest
    probability in the evaluation problem for that output sequence
    """
    predicted = []
    oth_c = 0
    obama_c = 0
    next = 0
    t = 0.000
    d=dict()
    d["other"]=0
    d["obama"]=0
    print(obs_file+":")
    for obs in output_seq:
        p1 = M1.forward_scaled(obs)
        p2 = M2.forward_scaled(obs)


        if(p1 > p2):
            predicted.append("_____")
            oth_c+=1
            d["other"]+=1
        else:
            predicted.append("Obama")
            obama_c+=1
            d["obama"]+=1
        t+=0.05
        next+=1

        if(next % 20 == 0):
            p_other = d["other"]/20.0
            p_obama = d["obama"]/20.0
            if p_other > p_obama:
                print "{0} : Speech : ------".format(t)
            else:
                 print "{0} : Speech : Obama".format(t)
            d["other"]=0
            d["obama"]=0

    time = 0.000
    with open("op_"+obs_file,"w") as g:
        for val in predicted:
            g.write(str(time)+" :\t"+val+"\n")
            time+=0.005

    return predicted

if __name__=="__main__":
    codebook = kmeans_Mfcc_mod_train()
    d_other = read_file("other_trng_vq.txt")
    d_obama = read_file("new_obama_trng_vq.txt")
    M1 = train_machine(d_other,"model.txt")
    M2 = train_machine(d_obama,"model.txt")
#    print '*'*50
#    print M1.B
#    print '*'*50
#    print M2.B
    #obs_file_link = sys.argv[1]
    #get_audio_from_video(obs_file_link)
    obs_file = kmeans_Mfcc(codebook)
    all_obs = read_file(obs_file)
    pred_seq = test(all_obs,M1,M2,obs_file)
        #print pred_seq
