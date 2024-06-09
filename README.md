The steps to run the code are as follows:
First, make sure the environment has the pytorch installed in it and all other dependencies that are requried to run any CNN or RESNET code.
Store the data downloaded from the official CIFAR-10 dataset in the folder outside code, and name the folder as 'data'.
Hence, the overall folder structure should be a parent folder which contains a 'code' folder and a 'data' folder. 
Then open the terminal and cd to the parentfolder and run python3 code/main.py --mode train to train the data.
Similarly, python3 code/main.py --mode test to validate and test the data on public test dataset, and python3 code/main.py --mode predict to run the code to predict on private test data. 
Model configurations can be changed using the Configure.py file.
The pre-trained model weight is attached with this folder. To run the code on pre-trained model, change the path in "best_model_path" configuration.
In case of any issues, please feel free to contact 'mohitsarin26@tamu.edu'.