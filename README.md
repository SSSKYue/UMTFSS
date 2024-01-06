# Unsupervised-FSS
PyTorch implementation of our AAAI 2024 paper: *Label-efficient Semantic Segmentation with Unsupervised meta training*.
## Preparation
### Dependencies
This project is originally developed with Python 3.8, PyTorch 1.9, and CUDA 10.2.
- Create the environment:
```
bash install.sh
```
- Build the dependencies:
```
cd model/ops/
bash make.sh
cd ../clusterutils/
bash make.sh
cd ../../
```
                     
### Datasets
- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

### Models
- Download the self-pretrained feature extractors (e.g., MOCO_v2, from [here](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar)) under the folder `UnsupFSS/weights`
- Download the pre-trained backbones from [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) and put them into the `UnsupFSS/initmodel` directory.

## Pseudo Masks Generalization
Run the following comands to create pseudo masks.
```
cd masks
bash create.sh {*root*} {*dataset*}
```

Your directory structure should be prepared as follows before training:
```
${YOUR_PROJ_PATH}$
└── data
    ├── pascal
    |   ├── pascal_moco_pseudo
    |   ├── JPEGImages
    |   └── SegmentationClassAug
    └── COCO2014
        ├── coco_moco_pseudo
        ├── annotations
        ├── train2014
        └── val2014
```

## Train & Testing
Change configuration via the `.yaml` files in `UnsupFSS/config`, then run the `.sh` scripts for training and testing.

- *Unsupervised* Meta-training

  Train the FSS model under the unsupervised meta-training paradigm, using images across all folds.

  ```
  sh train.sh {*dataset*} ssl_resnet50
  ```

- *Supervised* Meta-training

  Fine-tune the model using specific fold and dataset.
    ```
    train.sh {*dataset*} {*model_config*}
    ```
    For example, 
    ```
    sh train.sh pascal split0_resnet50
    ```
    Modify `config` file (specify `data_list` with different sizes, e.g., train_5.txt denotes 5\% of the origin training set.)

- Meta-testing

  Run the following `.sh` script for testing.

  ```
  sh test.sh {*dataset*} {*model_config*}
  ```
  For example, 
  ```
  sh test.sh pascal split0_resnet50
  ```
  Modify `config` file (specify `checkpoint_path`)
