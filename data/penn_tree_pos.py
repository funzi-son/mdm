"""
POS dataset reader

Son N. Tran

"""
DATA_DIR = "/home/tra161/WORK/Data/sequence_labeling_data/pos/"

class PennTreePOS(object):

    def __init__(self,train_size):
        self._train_size = train_size
        

    def get_data(self):
        
