### Balot Paper Classification
A Balot Paper Classification project.   [Slides](https://docs.google.com/presentation/d/1NMILVamP5lZrXM_Bpj1k-pMxski7NXC-Zz2LvRPVcVM/edit?usp=sharing)

### Project Structure

	
		├── README.md          <- README file.
		|              
		├── ml		     <- Source code for the use in this project.
		│   ├── __init__.py
		│   │
		│   ├── src	          <- Scripts to ML model
		│   │   └── __init__.py	 
		│   │   └── models/train_model_cnn.py       
                    └── models/train_model_feed_forward.py
                    └── models/train_model_resnet18.py
		|   | 
		|   |
		|   |
		│__notebooks      		                    <- Collection of notebooks .
		│   │   └──  CNN_balot.ipynb	                    <- CNN Architecture
		|   |   |__  feed_forward_balot(1).ipynb	    <- Feed Forward Network
		│   │   |__  resnet_balot_final.ipynb	            <- Final resnet18 Model
		|   |
		├── out      <- Scripts to predict from saved_model
                 └──  out/model/model_type/load_predict.py 
            │         
		│── requirements.txt          <-Pip generated requirements file for the project.
		

### Getting Started

##### Installation

**Cloning the repository For code**:
    
		#Get the code 		    
		git clone https://github.com/prajinkhadka/fuse_training_balot_classification.git

**Create the Virtual Environment to avoid any conflicts**:
 
		#Creating Virtual Env
		virtualenv -p python3 .venv
		#Activating virtual env
		source .venv/bin/activate 
 
**Install Dependencies**:

		pip install -r requirement.txt 


### fuse Training Balot Classification - Summary

Balot Paper Classification.

Tried 3 Different Architecture.

    A. FeedForward - 6 layers.

    B. CNN - 7 layers 

    C. ResNet18

Best Result was obtained using ResNet 18 test accuracy of 99.92 %, trained for two epochs.

Results are Summarized in this presentation : [Results Presentation](https://docs.google.com/presentation/d/1NMILVamP5lZrXM_Bpj1k-pMxski7NXC-Zz2LvRPVcVM/edit#slide=id.g806adcd27f_0_89)


### For training 

1.clone  the repo : " git clone https://github.com/prajinkhadka/fuse_training_balot_classification.git "

2.download python 3.7.4

3.make virtualenv.

		#Creating Virtual Env
		virtualenv -p python3 .venv
		#Activating virtual env
		source .venv/bin/activate 
 
4.install requirements by following command:

    pip install -r requirements.txt

5.run fuse_training_balot_classification/ml/src/models/train_model_resnet18.py


### For training in GPU with Colab Notebook

Run [RenNet18](https://colab.research.google.com/drive/1m1Bg_4D1-U3PorB87vX5lqmeUCuv9TIL) in Google Colab ( with GPU instance Enabled )

More Instructions are in the notebook itself.

It will save the trained model as "checkpoint.pth" in the Currrect colab directory which will be recylced by colab in next runtime. Download it for further use.


### For Prediction :

Download the saved Model from here - 

Run " fuse_training_balot_classification/out/model/model_type/load_predict.py " , with checkpoint_best.pth in same directory.

Download saved model from here : [Saved Model](https://drive.google.com/open?id=1JUQwy_v7KpSGxzzw3SlRynT8Zre5Ov3f)

### For Prediction using GPU in Colab.

Go to this Colab Note book [Prediction](https://colab.research.google.com/drive/1l1WLh7OxbWFCd9I-u5IyYn_tVx0e-LZ1#scrollTo=bf8ICTQMiMuz)

    1. Upload the saved model "checkpoint_best.pth"
    
    2. Upload the testset (Zip file).
    
    3. Run the predict Function.
    
    4. More Commands how to run are in the notebook itself.
