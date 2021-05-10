import os
import argparse
import pickle
folders = ["/data/yingshac/MMMHQA/imgFeatures/distractors"]
output_folders = ["/data/yingshac/MMMHQA/imgFeatures_upd/distractors"]
       
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)
args = parser.parse_args()

if __name__ == "__main__":
    for fo, output_fo in zip(folders, output_folders):
        print(output_fo)
        count = args.start
        for file in os.listdir(fo)[args.start:args.end]:
            
            count += 1
            if count%1000 == 0: 
                print("start = {}, finish up to {}".format(args.start, count))
            with open(os.path.join(fo, file), "rb") as f:
                feats = pickle.load(f)
            feats['pred_boxes'] = feats['pred_boxes'].tensor
            #print(feats)
            #print(file)
            with open(os.path.join(output_fo, file), "wb") as f:
                pickle.dump(feats, f)
