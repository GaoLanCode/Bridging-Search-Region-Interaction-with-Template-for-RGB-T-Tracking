# Bridging-Search-Region-Interaction-with-Template-for-RGB-T-Tracking
![image](https://github.com/GaoLanCode/Bridging-Search-Region-Interaction-with-Template-for-RGB-T-Tracking/assets/173158276/a6c84d5a-065d-4b1e-8c19-e77098eef531)  
This is the implementation of my work.    
This is a project assignment used in Professor Liang's computer vision course~  
## Demo Video


https://github.com/GaoLanCode/Bridging-Search-Region-Interaction-with-Template-for-RGB-T-Tracking/assets/173158276/cdb72cb8-fbbb-45e5-aa41-62feb61f3b5b


## Environment Installation
conda create -n tbsi python=3.8
echo "****************** Installing pytorch ******************"
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
pip install PyYAML  
pip install easydict  
pip install cython  
pip install opencv-python  
pip install pandas  
conda install -y tqdm  
pip install pycocotools  
pip install jpeg4py  
pip install tb-nightly  
pip install tikzplotlib  
pip install thop-0.0.31.post2005241907  
pip install colorama  
pip install lmdb  
pip install scipy  
pip install visdom  
pip install tensorboardX  
pip install setuptools==59.5.0  
pip install wandb  
pip install timm  
## Project Paths Setup
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in `./data`. It should look like:
```
${PROJECT_ROOT}
  -- data
      -- lasher
          |-- trainingset
          |-- testingset
          |-- trainingsetList.txt
          |-- testingsetList.txt
          ...
```

## Training
Download [ImageNet or SOT](https://pan.baidu.com/s/1U42J6b3g1htma0OvmXRQCw?pwd=at5b) pretrained weights and put them under `$PROJECT_ROOT$/pretrained_models`.

```
python tracking/train.py --script tbsi_track --config vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --save_dir ./output/vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --mode multiple --nproc_per_node 4
```

Replace `--config` with the desired model config under `experiments/tbsi_track`.

## Evaluation
Put the checkpoint into `$PROJECT_ROOT$/output/config_name/...` or modify the checkpoint path in testing code.

```
python tracking/test.py tbsi_track vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name lasher_test --threads 6 --num_gpus 1

python tracking/analysis_results.py --tracker_name tbsi_track --tracker_param vitb_256_tbsi_32x4_4e4_lasher_15ep_in1k --dataset_name lasher_test
```
