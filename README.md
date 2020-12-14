# Image-Captioning
Automated <b> <i> image to text-speech </i ></b> generator.

A model for generating textual description of a given image based on the objects and actions in the image


## Dataset
The dataset used is Flicker8k (https://www.kaggle.com/ming666/flicker8k-dataset).


## Implementation
The image captioning task is divided into following parts:
1. Preparing Data
2. Preprocessing Data
3. Building Vocabulary
4. Image Captioning


### Preparing Data
Data preparation involves mapping captions to their respective image's id.  
Module: Prepare Data.ipynb  
Output: captions.txt


### Preprocessing Data
The captions are augmented with \<start\> token in the beginning and \<end\> token at the end and the images are passed through an InceptionV3 model to generate an encoding for each image.  
Module: Preprocess Data.ipynb  
Output: train_captions.txt, test_captions.txt, train_images.pkl, test_images.pkl  


### Building Vocabulary
A vocabulary is prepared from the augmented captions (the one including \<start\> and \<end\>). A word in the captions with a frequency of more than 10 is added to the vocabulary.  
Module: Build Vocab.ipynb  
Output: vocabulary.txt  
  
  
### Image Captioning
Module: Image Captioning.ipynb  
The image captioning module uses the outputs of the other modules to learn to generate captions for an input image. This step performs the following tasks:  

#### 1. Create mapping:
The words in the vocabulary are mapped to an integer value (or index) and two mappings are created - word_to_index and index_to_word.  

#### 2. Create embedding:
Word embeddings are created using pre-trained GloVe word representations. An embedding matrix is created wherein at each word-index (obtained from word_to_index mapping) the embeddings of the word are stored.  

#### 3. Build model:
The model takes as input an image vector from the training set (train_images.pkl) and a partial caption. The partial caption is initialised as \<start\> and is built successively by a feed forward neural network. The model predicts the next word in the sequence of words forming the partial caption which is, thereafter, added to the partial caption to generate a new partial caption. The process is repeated until \<end\> is generated as the predicted word.  
The architecture is as shown:  
![image](https://user-images.githubusercontent.com/31109495/76164447-fef82f80-6174-11ea-9fc3-cf3a2fed19a9.png)
Loss function: Categorical Cross Entropy  
Optimizer: Adam 
  
#### 4. Make predictions:  
Beam search with a beam width of 3 is used to predict next word in the caption.  


## Results  
| ![image](https://github.com/JiteshGupta17/Image-Captioning/blob/master/Screenshots/76382310-4998c800-637e-11ea-9e42-04891a26f5b8.png) | 
|:--:| 
| *two dogs are playing with each other in the snow .* |
| ![image](https://github.com/JiteshGupta17/Image-Captioning/blob/master/Screenshots/76382497-f3785480-637e-11ea-986c-1af99a1262c7.png) | 
| *a little boy is sitting on a slide in the playground .* |
| ![image](https://github.com/JiteshGupta17/Image-Captioning/blob/master/Screenshots/76382600-43571b80-637f-11ea-975d-e3481df4595f.png) | 
| *two poodles play with each other in the snow .* |
| ![image](https://github.com/JiteshGupta17/Image-Captioning/blob/master/Screenshots/76382753-bcef0980-637f-11ea-8b2a-2884b30ada54.png) | 
| *football players are tackling a football player carrying a football .* |
| ![image](https://github.com/JiteshGupta17/Image-Captioning/blob/master/Screenshots/76382862-0b040d00-6380-11ea-97d7-ec7b0e4d0ee7.png) | 
| *a snowboarder jumps over a snow covered hill .* |
| ![image](https://github.com/JiteshGupta17/Image-Captioning/blob/master/Screenshots/76382877-17886580-6380-11ea-90ae-ab09b6c74428.png) | 
| *a boy in a blue shirt is doing a trick on his skateboard .* |
| ![image](https://github.com/JiteshGupta17/Image-Captioning/blob/master/Screenshots/76382922-3dae0580-6380-11ea-84b4-047bd574bfc0.png) | 
| *a climber is scaling a rock face whilst attached to a rope .* |


## Silent Features  
<ul>
  <li> Weights are saved to the disk after each epoch and reloaded to resume training. This is done to save training time. </li>
  
<li> A custom data generator is used to feed input to the model. This enables us to train the model without loading the entire dataset into memory. </li>
  
<li> Beam search is used instead of greedy search to generate better captions. </li>
</ul>

## Deployment
<ul>
  <li> Used Flask framework and ginger templates for the deployment purpose. Flask is a web framework. It provides us with tools, libraries and technologies that allow us to build a web application </li>
  
<li> Also added an additional feature of text to speech conversion of the generated caption. </li>
</ul>

| ![image](https://github.com/JiteshGupta17/Image-Captioning/blob/master/Screenshots/Deployed.JPG) | 
|:--:| 

## References
<ul>
  <li> https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8 </li>
  <li> https://www.youtube.com/watch?v=RLWuzLLSIgw </li>
</ul>
