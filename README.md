# [Speech Understanding] Minor Project

This readme file contains detailed explanation of the code provided and instructions on how to run each file for successful reproduction of the reported results.

#### Environment requirements
First, note that in order for the provided code to run successfully, an environment with all the required packages must be installed. Consequently, we have provided a *'package_requirements.txt'* file which contains the name and respective version of all the packages used during this minor project and are necessary to reproduce the results.

Second, note that the provided code is computationally and storage-wise extensive. This implies that it required strong GPU-resources and large storage (for data, models etc.). Consequently, please run the provided code files on such a system only.

#### Pre-requisites
Now, note that there are data-wise and model-wise pre-requisites of the code. This is listed as follows:

##### Data-wise pre-requisites
Now, the required data must be downloaded and located in the specified path.

##### Model-wise pre-requisites
Now, the required model checkpoints must be downloaded and located in the specific path.

Please note that each model has it package requirements provided on its official github repository (requirements.txt) which should be loaded for its successful execution. 

Also, a few folders will be downloaded in the appropriate location following the imports.

### Instructions

Now, note the following set of instructions for the successful reproduction of provided results.

- Run the main.py file with the appropriate hyper-parameters to reproduce the results of the conventional feature extraction approach.
- Run the main_beats.py file followed by the classical.py file with the appropriate path locations to reproduce the resuld of the BEATS feature extraction appoach.
- Run the provided .ipynb files to reproduce the SimCLR based feature extraction approach.


NOTE: Please contact the author in case of any discrepancy.