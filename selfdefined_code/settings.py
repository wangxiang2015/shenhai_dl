import pandas as pd
from pathlib import Path


classes = sorted(['270492004', '164889003', '164890007', '426627000', '713427006', 
                  '713426002', '445118002', '39732003', '164909002', '251146004', 
                  '698252002', '10370003', '284470004', '427172004', '164947007', 
                  '111975006', '164917005', '47665007', '59118001', '427393009', 
                  '426177001', '426783006', '427084000', '63593006', '164934002', 
                  '59931005', '17338001'])

gpu_list = [0,1]

class_weights = None

data_df = pd.read_csv('used_filename.csv', index_col=0)
src_path = '/home/evsjtu/wangxiang/ecg_project/training_data'
# 截取前1000个数据
data_df = data_df[:1000]

ch_idx = 1 # 选择第 1 个导联
all_feats = pd.concat([pd.read_csv(f, index_col=0) for f in list(Path('feats/').glob(f'*/*all_feats_ch_{ch_idx}.zip'))])

padding = 'zero' # 'zero', 'qrs', or 'none'
window = 15*500

test_folds = [3]


# ~/wangxiang/ecg_project/ecgcode/transformerecgcode/test_multiplethread/mytest : 快速的
# ~/wangxiang/ecg_project/ecgcode/test_multiplethread/mytest ： 慢的