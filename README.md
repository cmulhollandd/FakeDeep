# FakeDeep
## A sort of terrible deepfake implementation
***
This isn't really gonna work that well so theres probably no point but whatever

probably the only redeeming quality of this thing is that theres a cool way to visualize the training process

Also theres no real documentation
if you really want to learn how to use it just look through the source code or this readme, its not that hard

theres probably a better way to do almost everything here so if you want to fix somthing go for it
***

### Quick Intstructions:
1. navigate to the project folder
2. run ```python App.py <path to src1> <path to src2>```
	* this runs the full monty with all the defaults
	* if you want to customize some stuff you can do that with the flags
3. if you want even more customization use the more intense version down below

***
### instructions for more custom workflows:
1. Run ```python ExtractFaces.py <path to src1> <path to src2>```
	* optionally you can use the flags to use some custom parameters
	* not gonna list all the flags here but they're pretty easy
2. Run ```python Train.py```
	* this runs with defaults that are probably pretty bad but whatever
	* again, use the flags to customize stuff
	* this saves the generated model to ```/src/dual_model.h5```
