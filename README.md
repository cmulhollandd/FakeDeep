# FakeDeep
## A kinda shitty deepfake implementation
***
This isn't really gonna work that well so theres probably no point but whatever

probably the only redeeming quality of this thing is that theres a cool way to visualize the training process

Also theres no documentation
if you really want to learn how to use it just look through the source code, its not that hard

theres probably a better way to do almost everything here so if you want to fix somthing go for it
***

### Quick Intstructions:
1. navigate to the project folder
2. run ```python ExtractFaces.py <path to first video> <path to second video> <max number of frames for each video>```
	* this will extract the faces from each frame of the video and save them to :
		1. ```/faces/face{1 or 2}/face/```
3. run ```python Train.py```
	* By default this will use a model thats the same size as the images
	* you can use a custom model by replacing ```/src/dual_model.json``` with a custom keras model saved as a json file
		* this requires the image size to be **64x64**
		* as with all this you can change it yourself and I will probably change it in the future so none of this will matter in like a week

