#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from math import ceil

class BPR():
    """ 
    Bayesian Personalized Ranking using Matrix Factorization 
    """
    
    def __init__(self,df,test_size=0.1,num_components=5):
        """
        Constructor
        
        - Arguments:
            df: pandas dataframe
            test_size: size (between 0 and 1) of the desired test split
            num_components: dimension of the latent vectors
        """
        self.df = df 
        self.num_components = num_components 
        self.test_size = test_size
        
        # separate positive and negative samples
        positive_df = self.df[self.df['Response']==1].reset_index().drop('index', axis=1)
        negative_df = self.df[self.df['Response']==0].reset_index().drop('index', axis=1)
        # instances of the class
        self.positive = positive_df
        self.negative = negative_df
        
        
        # creation of train and test sets (only positive samples)
        reduced_test = pd.DataFrame()
        reduced_train = self.positive.copy()
        for u in self.positive['user_id'].unique():
            user_df = self.positive[self.positive['user_id']==u]
            test_samples = user_df.sample(n=ceil(self.test_size*len(user_df)), random_state=42, replace=False)
            if len(user_df)>2:
                reduced_test = reduced_test.append(test_samples)
                reduced_train = reduced_train.drop(test_samples.index)
        reduced_train = reduced_train.reset_index().drop('index', axis=1)
        reduced_test = reduced_test.reset_index().drop('index', axis=1)
        self.reduced_train = reduced_train
        self.reduced_test = reduced_test
        
        self.n_users = self.reduced_train.user_id.nunique()
        self.n_items = self.reduced_train.id.nunique()
        
        # complete train (with positive and negative samples)
        train = pd.DataFrame()
        for requestid in self.reduced_train.request_id.unique():
            requestdf = self.df[self.df['request_id']==requestid]
            train = train.append(requestdf)
        self.train = train
        
        # complete the test (with positive and negative samples)
        test = pd.DataFrame()
        for requestid in self.reduced_test.request_id.unique():
            requestdf = self.df[self.df['request_id']==requestid]
            test = test.append(requestdf)
        self.test = test
        
        
        # dictionary from request id to negative offers id
        self.requestid_2_offerid_negative = dict()
        for requestid in self.negative.request_id.unique():
            request_df = self.negative[self.negative['request_id']==requestid]
            self.requestid_2_offerid_negative[requestid] = list(request_df.id.values)
            
        # dictionary from request id to positive offers id
        self.requestid_2_offerid_positive = dict()
        for requestid in self.positive.request_id.unique():
            request_df = self.positive[self.positive['request_id']==requestid]
            self.requestid_2_offerid_positive[requestid] = list(request_df.id.values)
            
        # creation of dictionaries to convert from id to matrix indices
        self.userid_to_idx = dict(zip(np.sort(self.reduced_train['user_id'].unique()),
                                      range(self.reduced_train['user_id'].nunique())))
        
        self.itemid_to_idx = dict(zip(np.sort(self.reduced_train['id'].unique()),
                                      range(self.reduced_train['id'].nunique())))
            
    
    def __sdg__(self):
        """ 
        Stochastic Gradient Descent
        """
        i = 1
        j = 1
        for _ in range(self.iterations):
            # random sample (positive)
            random_sample = self.reduced_train.sample()[['request_id','user_id','id']]
            rs_request_id = random_sample['request_id'].values[0]
            positive_offer_id = random_sample['id'].values[0]
            user_id = random_sample['user_id'].values[0]
            # indeces
            user_index = self.userid_to_idx[user_id]
            positive_offer_index = self.itemid_to_idx[positive_offer_id]

            # prediction
            f_i = self.predict(user_id, positive_offer_id)
            
            # random sample (negative, from the same set of offers)
            negative_offer_id = self.sample_offer_not_picked(rs_request_id)
            # indeces
            negative_offer_index = self.itemid_to_idx[negative_offer_id]
            # prediction
            f_j = self.predict(user_id, negative_offer_id)
            
            f_ij = f_i - f_j
            if f_ij < -10:
                f_ij = -10
            
            #operation shared (only computed once to not repeat the operation)
            exponential = (np.exp(-f_ij))/(1+np.exp(-f_ij))

            # update parameters
            self.user_vecs[user_index,:] += self.learning_rate*(exponential*(self.item_vecs[positive_offer_index,:] -                                                                              self.item_vecs[negative_offer_index,:]) +                                                                 self.lmbda*self.user_vecs[user_index,:])
            
            self.item_vecs[positive_offer_index,:] += self.learning_rate*(exponential* self.user_vecs[user_index,:] +                                                                      self.lmbda*self.item_vecs[positive_offer_index,:])
            
            self.item_vecs[negative_offer_index,:] += self.learning_rate*(-exponential* self.user_vecs[user_index,:] +                                                                      self.lmbda*self.item_vecs[negative_offer_index,:])
            if i%self.compute_metrics==0:
                self.metrics_iterations.append(i)
                print('{}/12'.format(j))
                # accuracy test
                print('Computing metrics...')
                acu_test = self.compute_ACU(self.reduced_test)
                self.acu_test.append(acu_test)
                # accuracy train
                acu_train = self.compute_ACU(self.reduced_train)
                self.acu_train.append(acu_train)
                # recall and MAP train
                recall_at_k_train, MAP_train, pos_bias_train = self.metrics(self.train, [1,5,10])
                self.recall_at_k_train.append(recall_at_k_train)
                self.MAP_train.append(MAP_train)
                self.pos_bias_train.append(pos_bias_train)
                print('Metrics for train completed')
                # recall and MAP test
                recall_at_k_test, MAP_test, pos_bias_test = self.metrics(self.test, [1,5,10])
                self.recall_at_k_test.append(recall_at_k_test)
                self.MAP_test.append(MAP_test)
                self.pos_bias_test.append(pos_bias_test)
                print('Metrics for test completed')
                j += 1
            i += 1
            
                
    def fit(self, single_iterations=1e5, learning_rate = 0.1, lmbda = 0.01):
        """
        Starts the training process
        
        -Input:
            single_iterations: total number of iterations
            learning_rate: hyperparameter of the SGD
            lmbda: regularizer of the SGD"""
        
        self.learning_rate = learning_rate 
        self.lmbda = lmbda 
        self.iterations = int(single_iterations)
        self.compute_metrics = int(single_iterations/12)
            
        # initialize latent vectors
        self.user_vecs = np.random.normal(scale=1./self.num_components,                                          size=(self.n_users, self.num_components))
        self.item_vecs = np.random.normal(scale=1./self.num_components,
                                          size=(self.n_items, self.num_components))
        
            
        #to follow the training process
        self.acu_test = list()
        self.acu_train = list()
        self.metrics_iterations = list()
        self.recall_at_k_train = list()
        self.recall_at_k_test = list()
        self.MAP_train = list()
        self.MAP_test = list()
        self.pos_bias_train = list()
        self.pos_bias_test = list()
        
        self.__sdg__()
        
    def predict(self,user_id,item_id): 
        """
        Predict the score for a given user-offer pair using Factorization Machines
        
        -Input:
            user_id: user identifier
            offer_id: offer identifier
                 
        -Output:
            prediction score
        """
        user_idx = self.userid_to_idx.get(user_id,None) # check if the user is in the training data
        if user_idx is not None:
            item_idx = self.itemid_to_idx.get(item_id,None) # check if the item is in the training data
            if item_idx is not None:
                return np.dot(self.user_vecs[user_idx],self.item_vecs[item_idx].T) # inner product of latent vectors
            else:
                return 0.0 # new item
        else:
            return 0.0 #new user -> p(i>j) = 0.5
    
    def sample_offer_not_picked(self,request_id):
        """
        Sample a negative offer from a given mobility requests
        
        -Input:
            request_id: mobility request identifier
        
        -Output: 
            sampled_offer: negative offer from the mobility request
        """
        
        offers = self.requestid_2_offerid_negative[request_id]
        condition = False
        k = 0
        while condition == False and k<50:
            sampled_offer = np.random.choice(offers)
            if sampled_offer in self.reduced_train.id.unique():
                condition = True
                return sampled_offer
            k += 1
        return 25
    
    
    def compute_ACU(self,data):
        sum_over_users = 0.0
        for user_id in data.user_id.unique():
            user_df = data[data['user_id']==user_id]
            for request_id in user_df.request_id.unique():
                request_df = user_df[user_df['request_id']==request_id]
                offer_id = request_df['id'].values[0]
                f_i = self.predict(user_id,offer_id)
                sum_over_offers = 0.0
                for offer_negative in self.requestid_2_offerid_negative[request_id]:
                    f_j = self.predict(user_id,offer_negative)
                    sum_over_offers += self.Heaviside(f_i, f_j)
                sum_over_users += sum_over_offers/(len(self.requestid_2_offerid_negative[request_id]))
        return sum_over_users/(len(data.request_id.unique()))


 
    
    
    def metrics(self,data,ks):
        """
        Compute Recall@k and MAP
        
        -Input:
            data: pandas dataframe containing the data to evaluate (e.i. train set or test set)
            ks: list containing the different 'k' to evaluate
            
        -Output:
            recall_at_k_average: recall@k for each user (averaged over all the mobility requests
            of each user)
            MAP: Mean Average Precision for each user (averaged over all the mobility requests
            of each user)
        """
        
        votings = dict()
        recall_at_k = dict()
        recall_at_k_average = dict()
        ap = dict()
        MAP = dict()
        pos_bias = dict()
        pos_bias_avg = dict()
        for k in ks:
            recall_at_k.setdefault(k,{})
            recall_at_k_average.setdefault(k,{})
            for userid in data.user_id.unique():
                recall_at_k[k].setdefault(userid, 0)
                recall_at_k_average[k].setdefault(userid, 0)
        for userid in data.user_id.unique():
            votings.setdefault(userid,{})
            ap.setdefault(userid, 0.0)
            pos_bias.setdefault(userid, 0.0)
            pos_bias_avg.setdefault(userid, 0.0)
            MAP.setdefault(userid,0)
            user_df = data[data['user_id']==userid]
            for requestid in user_df.request_id.unique():
                votings[userid].setdefault(requestid, {})
                request_df = user_df[user_df['request_id']==requestid]
                offers = request_df.id.unique()
                for i in range(len(offers)-1):
                    offer_i = offers[i]
                    f_i = self.predict(userid, offer_i)
                    votings[userid][requestid].setdefault(offer_i, 0.0)
                    for j in range(i+1,len(offers)):
                        offer_j = offers[j]
                        votings[userid][requestid].setdefault(offer_j, 0.0)
                        f_j = self.predict(userid, offer_j)
                        f_ij = f_i - f_j
                        if f_ij < -10:
                            f_ij = -10
                        probability = 1./(1.+np.exp(-f_ij))
                        if probability > 0.5:
                            votings[userid][requestid][offer_i] += 1
                        else:
                            votings[userid][requestid][offer_j] += 1
                positive_offer = self.requestid_2_offerid_positive[requestid][0]
                ranking = sorted(votings[userid][requestid],key=votings[userid][requestid].get,reverse=True)
                for k in ks:
                    if positive_offer in ranking[0:k]:
                        recall_at_k[k][userid] += 1
                        if k == 5:
                            ap[userid] += 1.0/float((ranking[0:k].index(positive_offer)+1))
                            pos_bias[userid] += 1.0/(np.log(1+float((ranking[0:k].index(positive_offer)+1))))
            for k in ks:
                recall_at_k_average[k][userid] = recall_at_k[k][userid]/user_df.request_id.nunique()
                if k==5:
                    MAP[userid] = ap[userid]/user_df.request_id.nunique()
                    pos_bias_avg[userid] = pos_bias[userid]/user_df.request_id.nunique()
        return recall_at_k_average, MAP, pos_bias_avg


    
    
    def Heaviside(self,x1,x2):
        if x1 > x2:
            return 1.0
        else:
            return 0.0

        
            

