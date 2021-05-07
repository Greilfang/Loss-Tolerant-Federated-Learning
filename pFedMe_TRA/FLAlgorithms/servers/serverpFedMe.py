import torch
import os

from FLAlgorithms.users.userpFedMe import UserpFedMe
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
 
# Implementation for pFedMe Server

class pFedMe(Server):
    def __init__(self, device,  dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, K, personal_learning_rate, times,selected_rate):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times, selected_rate)

        # Initialize data for all  users
        data = read_data(dataset)
        total_users = len(data[0])
        self.K = K
        self.personal_learning_rate = personal_learning_rate
        for i in range(total_users):
            is_eligible = True if i<int(self.selected_rate*total_users) else False
            print("{} eligible:{}".format(i,is_eligible))
            id, train , test = read_user_data(i, data, dataset)
            user = UserpFedMe(device, id, train, test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer, K, personal_learning_rate,is_eligible)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating pFedMe server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            # send parameters
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            self.evaluate()

            # do update for all users not only selected users
            for user in self.users:
                user.train(self.local_epochs) #* user.train_samples
            
            # choose several users to send back upated model to server
            # self.personalized_evaluate()
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            self.evaluate_personalized_model()
            self.personalized_aggregate_parameters_with_droprate(0.1)
        
        self.send_parameters()
        self.evaluate()
        #print(loss)
        self.save_results()
        self.save_model()

    # def train(self):
    #     loss = []
    #     for glob_iter in range(self.num_glob_iters):
    #         print("-------------Round number: ",glob_iter, " -------------")
    #         # send all parameter for users 
    #         # send parameters
    #         self.send_parameters()

    #         # Evaluate gloal model on user for each interation
    #         print("Evaluate global model")
    #         self.evaluate()

    #         # do update for all users not only selected users
    #         for user in self.users:
    #             user.train(self.local_epochs) #* user.train_samples
            
    #         # choose several users to send back upated model to server
    #         # self.personalized_evaluate()
    #         # self.selected_users = self.select_users(glob_iter,self.num_users)
    #         self.selected_users = self.select_subset_users(glob_iter,self.num_users)

    #         # Evaluate gloal model on user for each interation
    #         #print("Evaluate persionalized model")
    #         #print("")
    #         self.evaluate_personalized_model()
    #         self.personalized_aggregate_parameters()
        
    #     self.send_parameters()
    #     self.evaluate()
    #     #print(loss)
    #     self.save_results()
    #     self.save_model()
    
    
  
