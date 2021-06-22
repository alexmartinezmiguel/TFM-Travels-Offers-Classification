# Classification of Travel Offers for Shared-Mobility Applications

This repository contains all the code produced and needed for the development of my master's thesis as a part of the MSc in Principle Fundamentals of Data Science.

The work has been developed in the framework of a European project called Ride2Rail and whose objective is to build an innovative system to classify multimodal mobility solutions along different categories.  On the one hand, we provide a detailed description of the three modules developed in this work which are to be incorporated in the Ride2Rail system. On the other hand, we use a simplified version of this multimodal offer classifier to categorize a set of trip alternatives. Finally, we train a recommender system to learn users’ preferences and generate personalized trip offer rankings. We compare two different approaches: one without including the categorization of the offers (i.e., without content) and one including it. This last approach shows considerably better results and therefore proves the usefulness of this specific categorization.

## Contributions

- In the [Ride2Rail][r2r] directory, one can find the direct contributions to the Ride2Rail project. In particular, we intend to develope a state-of-the-art Offer Categorizer which enables the description of offers along 11 different categories. These contributions are: a data extractor and two micro-services (Weather-FC and Panoramic-FC) which compute part of the aforementioned categorization. Each one of them has been implemented to be run in different Docker containers, and a thorough description of their usage can be found in each directory. 

- In the [Ranking][ranking] directory, one can find all the code produced to implement the recommender system which creates personalized offer rankings. One the one hand, a simplified version of the Offer Categorizer has been implemented to categorize the offers. Then, using this data, the Bayesian Personalized Ranking (BPR) algorithm has been implemented using two different underlying models: Matrix Factorization and Factorization Machines.


[r2r]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/tree/main/Ride2Rail
[ranking]: https://github.com/alexmartinezmiguel/TFM-Travels-Offers-Classification/tree/main/BPR
