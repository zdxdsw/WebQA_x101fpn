CUDA_VISIBLE_DEVICES=1 python featureExtraction.py --batch_size 8 --bucket 'distractors' --start 300000 --end 320000 &
CUDA_VISIBLE_DEVICES=1 python featureExtraction.py --batch_size 8 --bucket 'distractors' --start 320000 --end 400000