# Classification of Travel Offers for SharedMobility Applications

This repository contains all the code produced and needed for the development of my master's thesis as a part of the MSc in Principle Fundamentals of Data Science. The work has been developed in framework of the Europen project called Ride2Rail.

The repository is organized as follows:

- In the [Ride2Rail][r2r] directory, one can find the direct contributions to the Ride2Rail project. In particular, we intend to develope a state-of-the-art Offer Categorizer which enables the description of offers along 11 different categories. These contributions are: a data extractor, and two micro-services (Weather-FC and Panoramic-FC) which compute part of the aforementioned categorization. Each one of them has been implemenred to be run in different Docker containers, and a thorough description of their usage can be found in each directory. 

- In the [Ranking][ranking] directory, one can find all the code produced to implement the recommender system which creates personalized offer rankings. One the one hand, a simplified version of the Offer Categorizer has been implemented to categorize the offers. Then, using this data, the Bayesian Personalized Ranking (BPR) algorithm has been implemented using two different underlying models: Matrix Factorization and Factorization Machines.


[r2r]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/tree/main/Ride2Rail
[ranking]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/tree/main/BPR
