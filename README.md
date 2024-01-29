# Interpretation of Doctor's Handwriting


The handwriting recognition system is implemented using Python. Modules such as Opencv and Tensorflow are used.We are going to build a Handwriting Recognition System using a Neural Network to train the dataset. It consists of convolutional NN (CNN) layers, recurrent NN (RNN) layers and a final Connectionist Temporal Classification (CTC) layer.
The IAM Dataset is first preprocessed.The Preprocessing methods will enhance the input images and then simplify the problem for the classifier. 



*The input image which will be uploaded in the form of .jpg or png format. 


*An adaptive method is proposed for doctors handwriting recognition by pre-processing and training the dataset consecutively with CNN and RNN. The input paragraph images are first pre-processed and are split into line images and further line images are processed into word images which are fed into the NN model layers during recognition. After the CNN layers have processed their output, the RNN layers process it further. As a result of the RNN layers, the output text is given to the CTC for decoding. Through the use of consecutive CNNs,RNN and Levenshtein Distance the accuracy of the system improves steadily and the digital text of handwritten input image is generated.


## Screenshots of the system:

![2](https://github.com/sakshiisinghh/Recognition-System/assets/87891878/4d99f9cc-a127-49cf-88b2-37c935f7ed57)









![1](https://github.com/sakshiisinghh/Recognition-System/assets/87891878/47882eb8-85ff-4f45-bd05-5e05d54023d7)
