# Prototype-Propagation-Networks
This is the official code for IJCAI 2019 Paper: [Prototype Propagation Networks (PPN) for Weakly-supervised Few-shot Learning on Category Graph](https://arxiv.org/pdf/1905.04042.pdf)

We find weakly-labeled data as well as the propagation mechanism improve the performance of few-shot learning a lot.

If you find this project helpful, please consider to cite the following paper: 
```
@inproceedings{liu2019ppn,
title={Prototype Propagation Networks (PPN) for Weakly-supervised Few-shot Learning on Category Graph},
author={Liu, Lu and Zhou, Tianyi and Long, Guodong and Jiang, Jing and Yao, Lina and Zhang, Chengqi},
booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
year={2019}
}
```

## Dependencies
 - Python 3.6
 - Pytorch 1.0.0

## Datasets
 - Download datasets from [Google Drive](https://drive.google.com/drive/folders/1F792DyPe4XJJejwXyfkL3EWab0VK_5a2?usp=sharing)
 - Enter the dir of the downloaded datasets. Extract the datasets by ```tar -xvftiered-imagenet-pure.tar --directory ~/datasets/``` or ```tar -xvf tiered-imagenet-mix.tar --directory ~/datasets/```

## Training
- For 5 way 1 shot experiment on tiered-imagenet-pure:
  ```bash scripts/pp_buffer/train_anc.sh 0 5 1 pure all_level_avg_single 1```, where in order ```0``` is for which GPU to use, ```5``` is way, ```1``` is shot, ```pure``` is dataset (```mix``` otherwise), ```all_level_avg_single``` is training strategy used in our paper, ```1``` is the number of hops for propagation.

- For 5 way 5 shot experiment on tiered-imagenet-mix, where each parameter follows the same setup:
  ```bash scripts/pp_buffer/train_base_anc.sh 0 5 1 mix all_level_avg_single 1```

## Testing
- For 5 way 1 shot experiment on tiered-imagenet-pure:
  ```bash scripts/pp_buffer/test_anc_all_level.sh 0 5 1 pure all_level_avg_single 1 SEED```, where ```SEED``` is the random seed used in training (check the name of the training logs) and the other parameters follow the same setup.

- For 5 way 5 shot experiment on tiered-imagenet-pure, where each parameter follows the same setup:
  ```bash scripts/pp_buffer/test_base_anc_all_level.sh 0 5 1 mix all_level_avg_single 1 SEED```
