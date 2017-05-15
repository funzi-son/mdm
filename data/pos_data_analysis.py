DATA_DIR = "/home/tra161/WORK/Data/sequence_labeling_data/pos/"

import glob

def get_feature_size():
    files = glob.glob(DATA_DIR+"*.shin")
    targets = []
    min_index = 500000
    max_index = 0
    for f in files:
        with open(f) as f_reader:
            for line in f_reader:
                if not line:
                    continue
                strs = line.split()
                if strs[0] not in targets:
                    targets.append(strs[0])
                for s in strs:
                    if ":" in s:
                        try:
                            index = int(s[:s.index(':')])
                            if index<min_index:
                                min_index = index              
                            if index>max_index:
                                max_index = index
                        except ValueError:
                            continue

    print((min_index, max_index)) # 1,446054
    
def main():
    get_feature_size()
    
if __name__=="__main__":
    main()

    
