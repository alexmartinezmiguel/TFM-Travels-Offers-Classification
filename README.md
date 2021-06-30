# Classification of Travel Offers for Shared-Mobility Applications

This repository contains all the code produced and needed for the development of my master's thesis as a part of the MSc in Principle Fundamentals of Data Science.


## Abstract
Nowadays, there is an increasing amount of available digital data, and therefore there is a need to filter, prioritize and show personalized relevant information to the users. Recommender systems have completely changed the way we interact with many services by solving the problem of information overload. They are tools that search through large volumes of dynamically generated information to provide users with personalized content. Applications range from streaming services to online shopping websites. In this master’s thesis, we build such a system for trip offers: given a mobility request from a user, we seek to provide a personalized ranking of trip alternatives which differ in aspects such as duration, distance, transportation modes, etc. 

This work is organized in two parts. On the one hand, we focus on the characterization of the trip alternatives by defining 11 categories such as quick, reliable or environmentally friendly. We build a system which assigns a score to each one of these categories highlighting the particular compatibility of an alternative with that category. This first part is a direct contribution to Ride2Rail, a European project which we have used as the framework to develop this thesis. On the other hand, we use this trip representation to build a recommender system. In particular, we implement the Bayesian Personalized Ranking algorithm, which was specially designed to work with implicit feedback from the users. This algorithm relies on an underlying model class to make predictions. We take advantage of this fact to compare two approaches. Firstly, we use the Matrix Factorization, a classical collaborative filtering approach which does not take into account the previous categorization. Secondly, we use the Factorization Machines, a model which combines collaborative and content approaches to include the categorization information. The results clearly suggest that using the content information of trips categorization increases the performance of the recommender system, which proves its usefulness. 

## Contributions

- In the [Ride2Rail][r2r] directory, one can find the direct contributions to the Ride2Rail project. In particular, we intend to develope a state-of-the-art Offer Categorizer which enables the description of offers along 11 different categories. These contributions are: a data extractor and two micro-services (Weather-FC and Panoramic-FC) which compute part of the aforementioned categorization. Each one of them has been implemented to be run in different Docker containers, and a thorough description of their usage can be found in each directory. 

- In the [Ranking][ranking] directory, one can find all the code produced to implement the recommender system which creates personalized offer rankings. One the one hand, a simplified version of the Offer Categorizer has been implemented to categorize the offers. Then, using this data, the Bayesian Personalized Ranking (BPR) algorithm has been used comparing two different underlying models: Matrix Factorization and Factorization Machines.


[r2r]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/tree/main/Ride2Rail
[ranking]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/tree/main/BPR
