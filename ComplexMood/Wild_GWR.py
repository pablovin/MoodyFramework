"""
 GWR based on (Marsland et al. 2002)'s Grow-When-Required (Python 3)
@last-modified: 8 September 2018
@author: German I. Parisi (german.parisi@gmail.com)
from: https://github.com/pietromarchesi/neuralgas/blob/master/neuralgas/oss_gwr.py
Please cite this paper: Parisi, G.I., Weber, C., Wermter, S. (2015) Self-Organizing Neural Integration of Pose-Motion Features for Human Action Recognition. Frontiers in Neurorobotics, 9(3).
"""
import logging

import copy
import numpy as np
import networkx as nx
import scipy.spatial.distance as sp

# TODO should have a _get_activation_trajectories method too

class gwr():

    '''
    Growing When Required (GWR) Neural Gas, after [1]. Constitutes the base class
    for the Online Semi-supervised (OSS) GWR.
    [1] Parisi, G. I., Tani, J., Weber, C., & Wermter, S. (2017).
    Emergence of multimodal action representations from neural network
    self-organization. Cognitive Systems Research, 43, 208-221.
    '''

    def __init__(self, act_thr = 0.35, fir_thr = 0.1, eps_b = 0.1,
                 eps_n = 0.01, tau_b = 0.3, tau_n = 0.1, kappa = 1.05,
                  max_age = 100, max_size = 100,
                 random_state = None):
        self.act_thr  = act_thr
        self.fir_thr  = fir_thr
        self.eps_b    = eps_b
        self.eps_n    = eps_n
        self.tau_b    = tau_b
        self.tau_n    = tau_n
        self.kappa    = kappa
        self.max_age  = max_age
        self.max_size = max_size
        if random_state is not None:
            np.random.seed(random_state)

        initialState = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        self._initialize(initialState)

    def _initialize(self, X):
        # print ("Initializing!!!!!!!!!!!!!!!!!!!!!")
        logging.info('Initializing the neural gas.')
        self.G = nx.Graph()
        # TODO: initialize empty labels?
        draw = np.random.choice(X.shape[0], size=2, replace=False)

        # print ("Draw:" + str(draw))
        # print ("Data point:"  + str(X[draw[0],:]))
        self.G.add_node(0 , pos= X[draw[0],:] , fir = 1)
        self.G.add_node(1 , pos= X[draw[1],:] , fir = 1)

        # print ("nx.get_node_attributes(self.G, 'pos') :"  + str(nx.get_node_attributes(self.G, 'pos').values()))


    def get_positions(self):
        pos = np.stack(nx.get_node_attributes(self.G, 'pos').values(), axis=0)
        return pos

    def _get_best_matching(self, x):
        x = np.expand_dims(x, 0)

        pos = self.get_positions()

        print("x:" + str(x.shape))
        print ("Pos:" + str(pos.shape))
        dist = sp.cdist(x, pos, metric='euclidean')
        sorted_dist = np.argsort(dist)
        b = self.G.nodes()[sorted_dist[0,0]]
        s = self.G.nodes()[sorted_dist[0,1]]
        return b, s


    def _get_activation(self, x, b):
        p = self.G.node[b]['pos'][np.newaxis,:]
        dist = sp.cdist(x, p, metric='euclidean')[0,0]
        act = np.exp(-dist)
        return act


    def _make_link(self, b, s):
        # b = list(b)
        print ("Nodes:"  + str(self.G.nodes))
        b = self.G.node[b]

        print ("B:" + str(b))
        self.G.add_edge(self.G[b],self.G[s], age=0)


    def _add_node(self, x, b, s):
        r = max(self.G.nodes()) + 1
        pos_r = 0.5 * (x + self.G.node[b]['pos'])[0,:]
        self.G.add_node(r, pos = pos_r, fir=1)
        self.G.remove_edge(b,s)
        self.G.add_edge(r, b, age=0)
        self.G.add_edge(r, s, age=0)
        return r


    def _update_network(self, x, b):
        dpos_b = self.eps_b * self.G.node[b]['fir']*(x - self.G.node[b]['pos'])
        self.G.node[b]['pos'] = self.G.node[b]['pos'] + dpos_b[0,:]

        neighbors = self.G.neighbors(b)
        for n in neighbors:
            # update the position of the neighbors
            dpos_n = self.eps_n * self.G.node[n]['fir'] * (
                     x - self.G.node[n]['pos'])
            self.G.node[n]['pos'] = self.G.node[n]['pos'] + dpos_n[0,:]

            # increase the age of all edges connected to b
            self.G.edge[b][n]['age'] += 1


    def _update_firing(self, b):
        dfir_b = self.tau_b * self.kappa*(1-self.G.node[b]['fir']) - self.tau_b
        self.G.node[b]['fir'] = self.G.node[b]['fir']  + dfir_b

        neighbors = self.G.neighbors(b)
        for n in neighbors:
            dfir_n = self.tau_n * self.kappa * \
                     (1-self.G.node[b]['fir']) - self.tau_n
            self.G.node[n]['fir'] = self.G.node[n]['fir'] + dfir_n


    def _remove_old_edges(self):
        for e in self.G.edges():
            if self.G[e[0]][e[1]]['age'] > self.max_age:
                self.G.remove_edge(*e)
                for node in self.G.nodes():
                    if len(self.G.edges(node)) == 0:
                        logging.debug('Removing node %s', str(node))
                        self.G.remove_node(node)

    def _check_stopping_criterion(self):
        # TODO: implement this
        pass

    def _training_step(self, x):
        # TODO: do not recompute all positions at every iteration
        b, s = self._get_best_matching(x)

        print ("B and S:" + str(b)+"-"+str(s))
        self._make_link(b, s)
        act = self._get_activation(x, b)
        fir = self.G.node[b]['fir']
        logging.debug('Training step - best matching: %s, %s \n'
                      'Network activation: %s \n'
                      'Firing: %s', str(b), str(s), str(np.round(act,3)),
                      str(np.round(fir,3)))
        if act < self.act_thr and fir < self.fir_thr \
            and len(self.G.nodes()) < self.max_size:
            r = self._add_node(x, b, s)
            logging.debug('GENERATE NODE %s', self.G.node[r])
        else:
            self._update_network(x, b)
        self._update_firing(b)
        self._remove_old_edges()


    def train(self, X, n_epochs=20, warm_start = False, act_thr = 0.35, fir_thr = 0.1,eps_b = 0.1,
                 eps_n = 0.01):
        if not warm_start:
            self._initialize(X)

        self.act_thr = act_thr
        self.fir_thr = fir_thr

        self.eps_b = eps_b
        self.eps_n = eps_n

        for n in range(n_epochs):
            logging.info('>>> Training epoch %s', str(n))
            for i in range(X.shape[0]):
                x = X[i,np.newaxis]
                self._training_step(x[0])
                self._check_stopping_criterion()
        logging.info('Training ended - Network size: %s', len(self.G.nodes()))
