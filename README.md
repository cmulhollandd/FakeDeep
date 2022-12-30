# FakeDeep
## A toy version of the DeepFake Algorithm
**This is designed as a proof of concept version of the algorithm for experimentation only.**
***

### Quick Intstructions:
1. navigate to the project folder
2. run ```python App.py <path to src1> <path to src2>```
	* Runs the full video pipeline with default parameter values
	* Some parameters can be customized with command line flags
3. Even more customization can be achieved with the instructions for custom workflows below.

***
### Instructions for Custom Workflows:
1. Run ```python ExtractFaces.py <path to src1> <path to src2>```
	* Parameters can be customized using command line flags
2. Run ```python Train.py```
	* Runs the training pipeline with default values
	* Training hyperparameters can be changed using command line flags
	* This saves the generated model to ```/src/dual_model.h5```
