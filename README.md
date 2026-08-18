

## Usage
1. dual steram without pcl loss   

train command: python train_pcl.py -C configs/cfg_train_pcl.yaml -D 0

infer command: python infer_pcl.py -C configs/cfg_infer_pcl_val.yaml -D 0

3. dual steram + pcl loss
   
train command: python train_pcl_spec.py -C configs/cfg_train_pcl_spec_phase3.yaml -D 0

infer command: python infer_pcl_spec.py -C configs/cfg_infer_pcl_spec_val.yaml -D 0

