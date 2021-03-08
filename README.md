# FruitFlyEmbeddings
tensorflow implementation of the FruitFlyEmbeddings Paper

### Background

Some Researches proposed a model that is strongly based on the neural net of a fruit fly. They asked whether it is possible to create meaningful embeddings with this simple network. 

![model architecture](https://www.google.com/url?sa=i&url=https%3A%2F%2Ftwitter.com%2Fmark_riedl%2Fstatus%2F1351367496914378752&psig=AOvVaw1mQB42HO-cT_qxIA4XTIdF&ust=1614346759538000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCOiqvbuUhe8CFQAAAAAdAAAAABAJ)

### Goal

I want to implement the proposed architecture. Instead of word embeddings I will use market baskets to build product embeddings for an online retailer. 
This decision is solely based on the fact, that I do not want so spend ~1 day with training and therefore came up with an use cases that is a lot easier to interpret whiel having the same nature of embeddings.


### Setup

Download the dataset from [kaggle](https://www.kaggle.com/puneetbhaya/online-retail) and put it into a directory called "data".

Install all requirements:

``
pip3 install -r requirements.txt
``
