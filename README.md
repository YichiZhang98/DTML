## Dual-Task Mutual Learning for Semi-Supervised Medical Image Segmentation

### Introduction

This is the repository of '[Dual-Task Mutual Learning for Semi-Supervised Medical Image Segmentation](https://arxiv.org/pdf/2103.04708v2.pdf)' (PRCV 2021) 



### Usage

1. Clone the repo
```
git clone https://github.com/YichiZhang98/DTML
cd DTML
```
2. Put the data in data/2018LA_Seg_Training Set.

3. Train the model
```
cd code
python train_la_dtml.py
```

4. Test the model
```
python test_LA.py
```


### Acknowledgement
* This code is adapted from [UA-MT](https://github.com/yulequan/UA-MT), [SegWithDistMap](https://github.com/JunMa11/SegWithDistMap) and [DTC](https://github.com/HiLab-git/DTC). We thank all the authors for their contribution. 
* If our work is useful for your research, please consider citing:


    @inproceedings{zhang2021dual,
         title={Dual-Task Mutual Learning for Semi-Supervised Medical Image Segmentation},
         author={Zhang, Yichi and Zhang, Jicong},
         journal={arXiv preprint arXiv:2103.04708},
         year = {2021} }
