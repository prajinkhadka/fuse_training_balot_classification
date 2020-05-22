# fuse Training Balot Classification

Balot Paper Classification.

Tried 3 Different Architecture.

1. FeedForward - 6 layers.

2. CNN - 7 layers 

3. ResNet18

Best Result was obtained using ResNet 18 test accuracy of 99.92 %, trained for two epochs.

Results are Summarized in this presentation : [Results Presentation](https://docs.google.com/presentation/d/1NMILVamP5lZrXM_Bpj1k-pMxski7NXC-Zz2LvRPVcVM/edit#slide=id.g806adcd27f_0_89)


### For training 

1.clone  the repo : " git clone https://github.com/prajinkhadka/fuse_training_balot_classification.git "

2.download python 3.7.4

3.make virtualenv by specifying path to the python by the given command:
virtualenv -p PATH ENVIRONMENTNAME

e.g. virtualenv -p "C:\Python37_64\python.exe" balot_class

4.install requirements by following command:

    pip install -r requirements.txt

5.run fuse_training_balot_classification/ml/src/models/train_model_resnet18.py


### For training in GPU with Colab 

Run [RenNet18](https://github.com/prajinkhadka/fuse_training_balot_classification/blob/master/notebooks/resnet_balot_final.ipynb) in Google Colab ( with GPU instance Enabled )

It will save the trained model as "checkpoint.pth" in the Currrect colab directory which will be recylced by colab in next runtime. Download it for further use.


### For Prediction :

Download the saved Model from here - 

Run " fuse_training_balot_classification/out/model/model_type/load_predict.py " , with checkpointbest.pth in same directory.

Download saved model from here : [Saved Model](https://drive.google.com/open?id=1JUQwy_v7KpSGxzzw3SlRynT8Zre5Ov3f)
