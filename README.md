# Deep Learning Applications Labs
This repository contains the source code for the three laboratory assignments completed during the "Deep Learning Applications" course taught by Professor Andrew David Bagdanov ([@bagdanov](https://github.com/bagdanov) on Github). The labs cover a variety of topics related to deep learning, including convolutional neural networks, large language models, and reinforcement learning.

## Lab 1 - Convolutional Neural Networks

### Exercise 1.1: A baseline MLP
In the exercise 1.1 we have to implement a *simple Multilayer Perceptr* to classify the 10 digits of *MNIST*, implementing our own training pipeline, training this model to convergence and monitoring the loss and accuracy on the training and validation sets for every epoch.
**Tensorboard** has been used for performance monitoring.
The source code for this lab can be found in the `lab1/` directory.
#### Implementation and Results
In `trainer.py` has been implemented a **Trainer** Class. **Trainer** provides a `train()` method and a `test()` method that can be used not only for MLP, but for convolutional, residual or not, neural networks.
In the `models.py` have been implemented the three models used in this Laboratory: the **MLP**, the **CNN** and the **ResCNN**.
Tensorboard logs can be found in `labe1/model` with the saved models.
The results show the performance of the best *MLP* trained, the one with hidden-layer sizes: **[128, 64, 10]**
INSERISCI FIGURE E COOMENTA IL RISULTATO

### Exercise 1.2: Rinse and Repeat
In the exercise 1.2 we have to repeat the verification we did in exercise 1.1, but with **Convolutional** Neural Networks, showing that **deeper** CNNs *without* residual connections do not always work better whereas **even deeper** ones *with* residual connections.
This time we use *CIFAR10*, since MNIST is *very* easy.
#### Implementation and Results
The same *Trainer* of *MLP* has benn used to train the *CNNs* and the *ResCNNs*. These time has been evaluted differents **depths** to validate the hypothesis.
For the **CNN** has been evaluated: **1, 5, 10, 20, 25 layers deth**
For the **ResCNN** has been evaluted: **1, 5, 10, 20 layers depth**
INSERISCI FIGURE E COOMENTA IL RISULTATO

### Exercise 2.1: Explain why Residual Connections are so effective
In the exercise 2.1 we have to use our CNNs (with and without residual connections) to study and **quantify** why the residual versions of the networks learn more effectively.
So i write a simple trainining loop where I train simultaneously a CNN and a ResCNN with the same layers depth for just 150 batch iterations. During the training I add to the Summary writer the gradient magnitudes passing through the networks during backpropagation in the last layer of the two models (the dense layer).
Different layers size has been compared, all of them show the same results. The gradient magnitudes of the CNN tends to zero, showing vanishing gradient problem, instead the ResNet don't suffer of vanishing, neither exploding, gradients, even with the biggest layers depth evaluated.

INSERISCI FIGURE E COOMENTA IL RISULTATO

### Extra: Exercise 2.3: *Explain* the predictions of a CNN
The exercise 2.3 ask to use the CNN model we trained in Exercise 1.2 and implement [*Class Activation Maps*](http://cnnlocalization.csail.mit.edu/#:~:text=A%20class%20activation%20map%20for,decision%20made%20by%20the%20CNN.):

> B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba. Learning Deep Features for Discriminative Localization. CVPR'16 (arXiv:1512.04150, 2015).

Instead of implementing from scratch the **Class Activation Maps** mechanism, I enjoyed using the original source code of the tecnique:
["zhoubolei/CAM"](https://github.com/zhoubolei/CAM)

#### Results
The original code requires few modifications to work with my custom ResCNN, on CIFAR10.
I have to only ....
INSERISCI FIGURE

## Lab 2 - Large Language Models

The second lab explores large language models using the Hugging Face Transformers library. The source code for this lab can be found in the `lab2/` directory.

## Lab 3 - Reinforcement Learning

The third and final lab covers reinforcement learning, specifically deep Q learning and PPO. The source code for this lab can be found in the `lab3/` directory.

## Getting Started

To run the code in this repository, you will need to have Python 3 installed, as well as several deep learning libraries including TensorFlow, PyTorch, and Hugging Face Transformers.

To get started, clone this repository to your local machine and navigate to the directory of the lab you wish to run. From there, you can run the code using the appropriate commands.

## Contributors

This repository was created by [your name]. If you have any questions or concerns, please contact [your email].

## Acknowledgements

We would like to thank Professor Andrew David Bagdanov for teaching the "Deep Learning Applications" course and providing guidance on these labs. We would also like to acnowledge the creators of the deep learning libraries used in this repository, including TensorFlow, PyTorch, and Hugging Face Transformers.
