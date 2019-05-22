import sys
import torch
from torch import distributions
import torch.nn.functional as F

class MRF():
    def __init__(self, words, priors, pairwise_potential, verbose=False):
        self.words = []
        for idx, (word, prior) in enumerate(zip(words, priors)):
            self.words.append((word, prior))
        self.pairwise_potential = pairwise_potential
        self.n_words = len(self.words)
        self.verbose = verbose
    def get_univariate_potential_function(self, idx):
        assert (0 <= idx < self.n_words)
        return (lambda x: self.words[idx][1] if x == 1 else 1. - self.words[idx][1])
    
    def get_univariate_potential_array(self, idx):
        assert (0 <= idx < self.n_words)
        return torch.FloatTensor([1-self.words[idx][1], self.words[idx][1]])
    
    def get_pairwise_potential_array(self, source=None, target=None):
        return self.pairwise_potential 
    
    def get_pairwise_potential_function(self, idx1, idx2):
        assert ((0 <= idx1  and idx2 < self.n_words))
        return (lambda x1, x2: self.pairwise_potential[x1][x2])
    def get_initial_messages(self):
        """ key: (from, to) tuple, value: message """
        messages = {} 
        # ENH: rewrite to have parents etc and be less model specific
        for node_idx in range(self.n_words):
            if node_idx >= 1:
                messages[(node_idx - 1, node_idx)] = torch.FloatTensor([0.5, 0.5])
                messages[(node_idx, node_idx - 1)] = torch.FloatTensor([0.5, 0.5])
            if node_idx < self.n_words - 1:
                messages[(node_idx, node_idx + 1)] = torch.FloatTensor([0.5, 0.5])
                messages[(node_idx + 1, node_idx)] = torch.FloatTensor([0.5, 0.5])

        #messages = [self.get_univariate_potential_array(idx) for idx in range(self.n_words)]#messages = torch.tensor([len(words),2,])
        return messages
    def set_initial_messages(self):
        self.messages = self.get_initial_messages()
        return self.messages

    def get_belief(self, index):
        univariate = torch.FloatTensor(self.get_univariate_potential_array(index))
        messages = []
        if index > 0:
            messages.append(self.messages[(index - 1, index)])
        if index < self.n_words - 1:
            messages.append(self.messages[(index + 1, index)])
        for message in messages:
            univariate *= message
        return univariate

    def get_message(self, source_index, dest_index, messages):
        assert ((0 <= source_index  and dest_index < self.n_words) and abs(source_index - dest_index) == 1)
        univariate = self.get_univariate_potential_array(dest_index)
        res_v = [] # The vector
        for target_v in [0,1]:
            s = 0
            for source_v in [0,1]:
                # Start with univariate and pairwise beliefs
                p = (self.get_univariate_potential_array(source_index)[source_v] *
                     self.get_pairwise_potential_array(source_index, dest_index)[source_v][target_v])
                # Also multiply the messages to the source node
                if source_index >= 1:
                    p *= messages[(source_index - 1, source_index)][source_v]
                if source_index < self.n_words - 1:
                    p *= messages[(source_index, source_index + 1)][source_v]
                s += p # ENH: LOG SCALE?
            res_v.append(s)
        return F.normalize(torch.FloatTensor(res_v), p=1,dim=0)       
        
    def make_inference(self, messages=None):
        if messages is None:
            old_messages = self.get_initial_messages()
        else:
            old_messages = messages
        total_dist = sys.maxsize
        n_iter = 0
        while (n_iter < 100 and total_dist > 0.0001):
            if self.verbose: print("Iteration: {0}, distance: {1}".format(n_iter, total_dist))
            total_dist = 0
            # compute all messages from all nodes to each other
            messages = {} 
            # ENH: rewrite to have parents etc and be less model specific
            for node_idx in range(self.n_words):
                if node_idx >= 1:
                    messages[(node_idx - 1, node_idx)] = self.get_message(node_idx - 1, node_idx, old_messages)
                    messages[(node_idx, node_idx - 1)] = self.get_message(node_idx, node_idx - 1, old_messages)
                    total_dist += torch.dist(messages[(node_idx - 1, node_idx)], old_messages[(node_idx - 1, node_idx)])
                    total_dist += torch.dist(messages[(node_idx, node_idx - 1)], old_messages[(node_idx, node_idx - 1)])

                if node_idx < self.n_words - 1:
                    messages[(node_idx, node_idx + 1)] = self.get_message(node_idx, node_idx + 1, old_messages)
                    messages[(node_idx + 1, node_idx)] = self.get_message(node_idx + 1, node_idx, old_messages)
                    total_dist += torch.dist(messages[(node_idx, node_idx + 1)], old_messages[(node_idx, node_idx + 1)])
                    total_dist += torch.dist(messages[(node_idx + 1, node_idx)], old_messages[(node_idx + 1, node_idx)])
            old_messages = messages
            n_iter += 1
        return messages
    
    def make_inference_and_get_beliefs(self):
        """ Return a vector of probabilities """
        messages = self.make_inference()
        beliefs = []
        # ENH: rewrite to use parents and be less model specific
        for node_idx in range(self.n_words):
            belief = self.get_univariate_potential_array(node_idx)
            if node_idx >= 1:
                belief *= messages[(node_idx, node_idx - 1 )]
            if node_idx < self.n_words - 1:
                belief *= messages[(node_idx, node_idx + 1)]
            beliefs.append(belief)
        return F.normalize(torch.stack(beliefs, dim=0), p=1,dim=1)
            
