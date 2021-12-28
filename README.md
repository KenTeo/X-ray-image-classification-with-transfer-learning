# Transfer Learning

Transfer learning is a method of learning using trained neural networks and adding layers of our own at the end of the trained neural network to train a new set of labelled output(images).
The advantage of such a method is existing weights from the trained neural network can be use to extract useful information without any training on our part. Hence, it will allow us to obtain a highly accurate image predicting neural network without needing to train the neural network from stratch. 
Significant amount of processing power and time is saved from using transfer learning for image prediction.

In this project, we have a total of 5856 labelled X-ray images to build a neural network model which classify if a
person has pneumonia or not based on their X-ray image. 624 images are reserved purely for testing the accuracy of the model. 80% of the remaining 5232 images will be use for training while the other 20% will be use for cross-validation. 
ImageDataGenerator from keras is used to create more images for train by transforming existing images. (For example, zoom in on images, shift the image, rotate image, etc..)

We will be applying transfer learning to build a model with VGG16 and another model with resnet152. Finally, the performance of both models will be compared and the one with higher accuracy will be selected. 

The X-ray images are taken from https://data.mendeley.com/datasets/rscbjbr9sj/3 and  https://www.kaggle.com/tolgadincer/labeled-chest-xray-images.

**Refer to following Link: https://paperswithcode.com/sota/image-classification-on-imagenet to stay updated with the best machine vision models available.**

### VGG16
VGG16 is a 16 weighted layered neural network which uses convolution neural net (CNN ) architecture that won the ILSVR(Imagenet) competition in 2014. This model achieves 92.7% top-5 test accuracy in ImageNet, which has a dataset of over 14 million images belonging to 1000 classes. VGG16 effectiveness comes from having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2 reduce the parameters and allow deeper neural network to be modelled.

*See below for the model summary built with transfer learning from VGG16*
```
Model with VGG16 transfer learning
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
 vgg16 (Functional)          (None, 7, 7, 512)         14714688  
                                                                 
 global_average_pooling2d (G  (None, 512)              0         
 lobalAveragePooling2D)                                          
                                                                 
 dense (Dense)               (None, 128)               65664     
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 1)                 129       
                                                                 
=================================================================
Total params: 14,780,481
Trainable params: 65,793
Non-trainable params: 14,714,688
```

Ref: Karen Simonyan∗ & Andrew Zisserman. Very deep convolutional networks for large-scale image reconigtion.

### ResNet-152
ResNet becomes the Winner of ILSVRC 2015 in image classification with a accuracy of 96.4%.
ResNet effectiveness comes from its very deep neural network, the one used has 152 layers.
Before ResNet, neural network model's accuracy tend to plateau when number of layers becomes larger like 30+. This is because during backpropagation, the multiplying of partial derivative will result in exploding or vanishing gradient when derivative >1 or <1. As the more layers there are, the gradients becomes increasing larger or smaller which will result in unstable training or close to zero training.

ResNet resolved the exploding/vanishing gradient problem by introducing shortcut connections is added to add the input x to the output after few weight layers. This allows another path(shortcut connections) to calculate the gradient where derivatives are skipped to update the weights. Therefore, the effect of exploding/vanishing gradient is greatly reduced.

*See below for the model summary built with transfer learning from resnet152*
```
Model with resnet152 transfer learning
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
 resnet152 (Functional)      (None, 7, 7, 2048)        58370944  
                                                                 
 global_average_pooling2d (G  (None, 2048)             0         
 lobalAveragePooling2D)                                          
                                                                 
 dense (Dense)               (None, 128)               262272    
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 1)                 129       
                                                                 
=================================================================
Total params: 58,633,345
Trainable params: 262,401
Non-trainable params: 58,370,944
```
Ref:Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition

### Steps in performing transfer learning
1. Load a good trained model (Example:VGG16, ResNet, etc..)
2. Usually the last layer of model 1. has to be removed as it is a output layer for the original trained categories.
3. Freeze the training of the weights for the trained model.
4. Add in additional layers after the trained model to classify our own required output.
5. Ensure that the input is converted to the required format of the trained model.
6. Train the model with transfer learning.
7. Optimize the model by tuning the parameters.
8. Advisable to perform transfer learning with a few highly effective trained model. Then, select the most suitable one.

### Summary of Result
|             | Model with VGG16|Model with Resnet-152|
|-------------|-----------------|---------------------|
|Test Accuracy|90.06%           |93.75%               |
From the testing results, we can see that Resnet-152 is more effective in classifying the features of the X-ray images. This can be due to it having a deeper neural network. Hence, the model is able to extract more information out of the 
images and allow a better classification of the images.
