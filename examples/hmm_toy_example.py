"""
Toy example

"""
import numpy as np
from models.hmm import HMM_DIS

# H,L
prior = [0.5, 0.5]
# H,L -> H,L
transitions = [[0.5, 0.5],[0.4,0.6]]
#H,L->A,C,G,T
emmissions = [[0.2,0.3,0.3,0.2],[0.3,0.2,0.2,0.3]]
if __name__=="__main__":
    hmm = HMM_DIS(None)
    hmm.set_prior(np.array(prior))
    hmm.set_transitions(np.array(transitions))
    hmm.set_emmissions(np.array(emmissions))
    #    G G C A C T G A A
    X = [2,2,1,0,1,3,2,0,0]
    #    H H H L L L L L L
    Y = [0,0,0,1,1,1,1,1,1]

    Y_ =  hmm.viterbi(X)
    print(Y_)
    
