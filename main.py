import numpy as np
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm
import itertools
from collections import deque

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import torch as T
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
import math
import seaborn as sns
import pyproj
import sys
import torch.multiprocessing as mp
import os
class environment:
  def __init__(self,g,data):
      super(environment, self).__init__()
      self.g=g
      self.data=data
      self.enc_node={}
      self.dec_node={}

      for index,nd in enumerate(self.g.nodes):
        self.enc_node[nd]=index
        self.dec_node[index]=nd
            
  def state_enc(self,dst, end):
    n=len(self.g.nodes)
    return dst+n*end

  def state_dec(self,state):
    n=len(self.g.nodes)
    dst = state%n
    end = (state-dst)/n
    return dst, int(end)
  def reset(self):
    self.state=self.state_enc(self.enc_node[1130166767],self.enc_node[1731824802])
    return self.state
  

  def step(self,state,action):
    done=False    
    current_node , end = self.state_dec(state)

    new_state = self.state_enc(action,end)

    rw,link=self.rw_function(current_node,action)

    if not link:
        new_state = state
        return new_state,rw,False  

    elif action == end:
        rw = 10000 #500*12
        done=True
      
    return new_state,rw,done
  
  def wayenc(self,current,new_state,type=1):
    #encoded
    if type==1: #distance
      if new_state in self.g[current]:
        #rw=data[g[current][new_state]['parent']]['distance']*-1
        rw=self.g[current][new_state]['weight']*-1
        return rw,True
      #rw=int(-sys.maxsize - 1)
      rw=-5000
      return rw,False

  def road_distance(self,current,new_node,link):
    if link:
      rw=self.data[self.g[current][new_node]['parent']]['distance']*-1
      return rw
    rw=-5000
    return rw


  def rw_function(self,current,new_state):
    beta=1 #between 1 and 0
    current=self.dec_node[current]
    new_state=self.dec_node[new_state]
    rw0,link=self.wayenc(current,new_state)
    rw1=self.road_distance(current,new_state,link)

    
    frw=rw0*beta+(1-beta)*rw1


    return frw,link
class SharedAdam(T.optim.Adam):
  def __init__(self,params,lr=1e-3,betas=(0.9,0.99),eps=1e-8,
               weight_decay=0):
    super(SharedAdam,self).__init__(params,lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
    for group in self.param_groups:
      for p in group['params']:
        state=self.state[p]
        state['step']=0
        state['exp_avg']=T.zeros_like(p.data)
        state['exp_avg_sq']=T.zeros_like(p.data)
        state['exp_avg'].share_memory_()
        state['exp_avg_sq'].share_memory_()


class ActorCritics(nn.Module):
  def __init__(self,input,n_actions,env,gamma=0.99):
    super(ActorCritics,self).__init__()
    self.gamma=gamma
    self.env=env
    self.n_actions=n_actions
    self.input=input
    self.pi1=nn.Linear(input,128)
    self.v1=nn.Linear(input,128)

    self.pi2=nn.Linear(128,64)
    self.v2=nn.Linear(128,64)

    self.pi=nn.Linear(64,n_actions)
    self.v=nn.Linear(64,1)

    self.rewards=[]
    self.actions=[]
    self.states=[]
  
  #this function takes the values of the state,actions,and reward and append to the memory
  def remember(self,state,action,reward):
    self.actions.append(action)
    self.rewards.append(reward)
    self.states.append(state)
  
  #this function reset the memory each time we are calling the learning function
  def clear_memory(self):
    self.states=[]
    self.actions=[]
    self.rewards=[]

  def forward(self,state):
    pi1=F.relu(self.pi1(state))
    v1=F.relu(self.v1(state))

    pi2=F.relu(self.pi2(pi1))
    v2=F.relu(self.pi2(v1))
    pi=self.pi(pi2)
    v=self.v(v2)
    #pi=self.pi(pi1)
    #v=self.v(v1)
    return pi,v
  
  def calc_returns(self,done):
    # 1-convert the states into tensor
    # 2- send the state into the forward function and get the (policy , value) we are intreseted in the value
    # 3- the return = (the last value in the list wich the value of the terminal state)*(1-done)
    # define a batch return list
    #loop through the inverted reward list
    #return(r)=reward+(gamma*R)
    #append R to the batch return list
    #reverce the batch return list to became the same ordder as the reward list
    list_state=[]
    if len(self.states)>1:
      for lstate in self.states:
        soruce,end=self.env.state_dec(lstate)
        vector=self.convert_vector(soruce,end)
        list_state.append(vector)
      states=T.tensor(list_state,dtype=T.float)

    else:
      soruce,end=self.env.state_dec(self.states[0])
      vector=self.convert_vector(soruce,end)
      states=T.tensor([vector],dtype=T.float)

    #states=T.tensor(self.states,dtype=float)
    #
    p,v=self.forward(states)

    R=v[-1]*(1-int(done))
    batch_return=[]

    for reward in self.rewards[::-1]:
      R=reward+self.gamma*R
      batch_return.append(R)
    batch_return.reverse()
    batch_return=T.tensor(batch_return,dtype=float)
    return batch_return


  def calc_loss(self,done):
    #states=T.tensor(self.states,dtype=T.float)
    list_state=[]
    if len(self.states)>1:
      for lstate in self.states:
        soruce,end=self.env.state_dec(lstate)
        vector=self.convert_vector(soruce,end)
        list_state.append(vector)
      states=T.tensor(list_state,dtype=T.float)
      flag='list'

    else:
      flag='int'
      soruce,end=self.env.state_dec(self.states[0])
      vector=self.convert_vector(soruce,end)
      states=T.tensor([vector],dtype=T.float)

    actions=T.tensor(self.actions,dtype=T.float)


    returns=self.calc_returns(done)

    p,values=self.forward(states)

    values=values.squeeze()

    critic_loss=(returns-values)**2
    probs=T.softmax(p,dim=1)
    dist=Categorical(probs)

    log_probs=dist.log_prob(actions)
    actor_loss=-log_probs*(returns-values)
    total_loss=(critic_loss+actor_loss).mean()
    return total_loss

  def convert_vector(self,lstate,goal_state):
    num_state=self.input
    if isinstance(lstate,list):
      oh_list=[]
      for s in lstate:
        s=int(s)
        vector=[0]*num_state
        vector[s]=1
        vector[goal_state]=1
        oh_list.append(vector)
      vector=oh_list

    else:
      vector=[0]*num_state
      vector[lstate]=1
      vector[goal_state]=1
    return vector

  def choose_action(self,observation,end):
      #print('nn')
      vector=self.convert_vector(observation,end)
      state=T.tensor([vector],dtype=T.float)
      pi,v=self.forward(state)
      probs=T.softmax(pi,dim=1)
      dist=Categorical(probs)
      action=dist.sample().numpy()[0]#take a sample from the categorical dist from 1-22
      return action


class Agent(mp.Process):
 def __init__(self,global_actor_critic,optimizer,input,n_actions
              ,gamma,lr,worker_name,global_episode_index,env,games,T_max):
   super(Agent,self).__init__()
   self.local_actor_critic=ActorCritics(input,n_actions,env,gamma)
   self.global_actor_critic=global_actor_critic
   self.worker_name='w%02i'%worker_name
   self.episode_idx = global_episode_index
   self.env=env
   self.optimizer=optimizer
   self.N_games=games
   self.T_max=T_max
 
 def run(self):
    t_step=1
    max_itr=5000
    #self.episode_idx is a gloabl parametar from MP class and we need to get the value from it
    while self.episode_idx.value < self.N_games:
      itr=0
      plnt=1
      done=False
      observation=self.env.reset()
      score=0
      penalties=0

      self.local_actor_critic.clear_memory()
      while not done:
        soruce,end=self.env.state_dec(observation)
        action=self.local_actor_critic.choose_action(soruce,end)
        observation_,reward,done=self.env.step(observation,action)
        if reward == -5000:
          penalties+=1
        
        score += reward
        self.local_actor_critic.remember(observation,action,reward)
        if t_step% self.T_max==0 or done:
          loss=self.local_actor_critic.calc_loss(done)
          self.optimizer.zero_grad()
          loss.backward()
          #set the current parameters for the workers into the gloabl parameters
          for local_param,global_param in zip(self.local_actor_critic.parameters(),
                                               self.global_actor_critic.parameters()):
            global_param._grad=local_param.grad
          self.optimizer.step()
          self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
          self.local_actor_critic.clear_memory()
        t_step+=1
        observation=observation_
        itr+=1
        
      with self.episode_idx.get_lock():
        self.episode_idx.value+=1
      print(self.worker_name,'episode',self.episode_idx.value,'reward',score,'penalties',penalties,'goal',done,
        'itr_to_done',itr,flush=True)

if __name__ == '__main__':

  def get_neighbors(node):
    neighbors=[i for i in h.neighbors(node)]
    return len(neighbors)


  f = open('final.json')

  data_str_key=json.load(f)

  data={}
  for k,v in data_str_key.items():
    data[int(k)]=v

  #reconstruct the graph
  h = nx.Graph()
  for key in data.keys():
    if data[key]['type']=='way':
      for i in range(len(data[key]['nodes'])-1):
        if 'tags' in data[key] and 'name' in data[key]['tags']:
            h.add_edge(data[key]['nodes'][i],data[key]['nodes'][i+1],parent=data[key]['id'],label=data[key]['tags']['name'])
        else:
          h.add_edge(data[key]['nodes'][i],data[key]['nodes'][i+1],parent=data[key]['id'])

  geod = pyproj.Geod(ellps='WGS84')

  # Compute distance among the two nodes indexed[s] indexed[d] using LON and LAT
  for s,d in h.edges():
    azimuth1, azimuth2, distance = geod.inv(data[s]['lon'],data[s]['lat'],data[d]['lon'],data[d]['lat'])
    h.edges[s,d]['weight'] = distance

  nodex={}
  for node in h.nodes:
    nodex[node]=get_neighbors(node)

  mx = max(nodex.values())
  [k for k, v in nodex.items() if v == mx]

  t=[i for i in h.neighbors(2003461246)]
  t.append(2003461246)

  node_list=[2003461246]
  for node in h.neighbors(2003461246):
    node_list.append(node)
    for snode in h.neighbors(node):
      node_list.append(snode)

  dictt={}
  for node in node_list:
    c=0
    for subnode in node_list:
      if subnode in h[node]:
        c+=1
    dictt[node]=c

  g = h.subgraph(dictt.keys())

  lr=1e-4
  n_actions=len(g.nodes)
  input=len(g.nodes)*(n_actions-1)
  N_games=3000
  T_max=10
  env=environment(g,data)

  global_actor_critic=ActorCritics(input,n_actions,env=env)
  global_actor_critic.share_memory()
  optim=SharedAdam(global_actor_critic.parameters(),lr=lr,betas=(0.92,0.999))
  global_ep=mp.Value('i',0)
  workers=[Agent(global_actor_critic,optim,input,n_actions,gamma=0.99,lr=lr,worker_name=i,
    global_episode_index=global_ep,env=env,games=N_games,T_max=T_max) for i in range(mp.cpu_count())]
  print('start',flush=True)
  [w.start() for w in workers]
  print('join',flush=True)
  [w.join() for w in workers]
  
  obs=env.reset()
  soruce,end=env.state_dec(obs)
  result=[env.dec_node[soruce]]
  done=False
  itr=0
  while not done:
    soruce,end=env.state_dec(obs)
    action=global_actor_critic.choose_action(soruce,end)
    obs_,reward,done=env.step(obs,action)
    obs=obs_
    soruce,end=env.state_dec(obs)
    result.append(env.dec_node[soruce])
    if itr>100:
      break
    else:itr+=1

  