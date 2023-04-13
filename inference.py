import argparse
import logging
import logging.config
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import chain
from logging import critical, debug, error, info, warning
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.distributions as D
from torch.multiprocessing import Process, Queue, Manager, set_start_method, get_start_method, get_context

from pydreamer.data import MlflowEpisodeRepository
from pydreamer.envs import create_env
from pydreamer.models import *
from pydreamer.models.functions import map_structure, cat_structure, cat_structure_torch
from pydreamer.preprocessing import Preprocessor
from pydreamer.tools import *


def main(env_id='MiniGrid-MazeS11N-v0',
         worker_id=99,
         policy_main='remote_network',
         model_reload_interval=600,
         model_conf=dict(),
         q_main=None,
         q_clients=None,
         ):

    configure_logging(prefix=f'[INFER {worker_id}]', info_color=LogColorFormatter.GREEN)
    info(f'INFERENCE started')
    mlflow_init(wait_for_resume=True)#Inference process is spawned to be able to share CUDA tensors so we need to reinit mlflow and resume to the main run
    env = create_env(env_id, model_conf.env_no_terminal, model_conf.env_time_limit, model_conf.env_action_repeat, worker_id)
    print("action_size:", env.action_size)
    # RUN
    info(f'Main policy: {policy_main}')
    policy = create_policy(policy_main, env.action_size, model_conf)

    if model_conf.inference_type == "single":
        single(q_main,q_clients,policy,model_conf,model_reload_interval)
    elif model_conf.inference_type == "multiproc":
        multiproc(q_main,q_clients,policy,model_conf,model_reload_interval)
    elif model_conf.inference_type == "batched":
        batched(q_main,q_clients,policy,model_conf,model_reload_interval)
    else:
        raise NotImplementedError

def multiproc(q_main,q_clients,policy,model_conf,model_reload_interval):
    last_model_load = 0
    manager = Manager()
    q_in = manager.Queue()
    q_out = manager.Queue()
    context = get_context("spawn")

    states = [policy.model.init_state(1) for i in range(model_conf.generator_workers)]
    states = [map_structure(state, lambda x: x.to(policy.device)) for state in states]

    for _ in range(model_conf.inference_data_workers):
        p = context.Process(target=in_process, daemon=True, args=[q_main, q_in, model_conf.inference_batch_size])
        p.start()
    for _ in range(2):
        p = context.Process(target=out_process, daemon=True, args=[q_out, q_clients])
        p.start()
    count,get,transfer1,policy_time,transfer2,put = 0,0,0,0,0,0
    while True:
        if time.time() - last_model_load > model_reload_interval:
            last_model_load = reload(count,get,transfer1,policy_time,transfer2,put,policy)
            count,get,transfer1,policy_time,transfer2,put = 0,0,0,0,0,0

        time_0 = time.time()
        (batch_obs, batch_ids) = q_in.get()
        time_1 = time.time()
        batch_obs_copy = map_structure(batch_obs, lambda x: x.to(policy.device))#Sending CUDA tensors did not work for some reason so send around CPU tensors and waste time transferring them to the GPU in main process :(
        del batch_obs
        batch_state = []
        for id in batch_ids:
            batch_state.append(states[id])
        batch_state = cat_structure(batch_state)
        time_2 = time.time()
        action_distr, metrics, new_states = policy(batch_obs_copy,batch_state)
        time_2_5 = time.time()
        #best practices from https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
        #new_state = map_structure(new_state, lambda x: x.to("cpu"))
        metrics = map_structure(metrics, lambda x: x.to("cpu"))
        logits = action_distr.logits.to("cpu")
        for i in range(len(batch_ids)):
            states[batch_ids[i]] = tuple(x[i:i+1,:] for x in new_states)
        time_3 = time.time()
        q_out.put((batch_ids, logits, metrics))
        time_4 = time.time()
        count+=model_conf.inference_batch_size
        get+=time_1-time_0
        transfer1+=time_2-time_1
        policy_time+=time_2_5-time_2
        transfer2+=time_3-time_2_5
        put+=time_4-time_3

def batched(q_main,q_clients,policy,model_conf,model_reload_interval):
    last_model_load = 0
    states = [policy.model.init_state(1) for i in range(model_conf.generator_workers)]
    states = [map_structure(state, lambda x: x.to(policy.device).share_memory_()) for state in states]
    
    count,get,transfer1,policy_time,transfer2,put = 0,0,0,0,0,0
    while True:
        if time.time() - last_model_load > model_reload_interval:
            last_model_load = reload(count,get,transfer1,policy_time,transfer2,put,policy)
            count,get,transfer1,policy_time,transfer2,put = 0,0,0,0,0,0

        ids = []
        batch_obs = []
        batch_state = []
        time_0 = time.time()
        for i in range(model_conf.inference_batch_size):
            #info(f'Waiting for requests at {q_main}')
            (my_id, obs_model) = q_main.get()
            #info(f'Got a request for queue {q_main}')
            ids.append(my_id)
            batch_obs.append(obs_model)
            batch_state.append(states[my_id])
        
        time_1 = time.time()
        batch_obs = cat_structure_torch(batch_obs,dim=1)#I think B is the second dim here, should be TBICHW
        batch_state = cat_structure(batch_state)
        batch_obs = map_structure(batch_obs, lambda x: x.to(policy.device))

        time_2 = time.time()
        action_distr, mets, new_state = policy(batch_obs,batch_state)
        time_2_5 = time.time()
        mets = map_structure(mets, lambda x: x.to("cpu"))
        logits = action_distr.logits.to("cpu")
        time_3 = time.time()
        for i in range(model_conf.inference_batch_size):
            #info(f'responding to {q_clients[ids[i]]}')
            states[ids[i]] = tuple(x[i:i+1,:] for x in new_state)
            q_clients[ids[i]].put((logits[:,i:i+1,:], {'policy_value': mets["policy_value"][:,i:i+1]}))
        time_4 = time.time()
        count+=model_conf.inference_batch_size
        get+=time_1-time_0
        transfer1+=time_2-time_1
        policy_time+=time_2_5-time_2
        transfer2+=time_3-time_2_5
        put+=time_4-time_3

def single(q_main,q_clients,policy,model_conf,model_reload_interval):
    last_model_load = 0
    states = [policy.model.init_state(1) for i in range(model_conf.generator_workers)]
    states = [map_structure(state, lambda x: x.to(policy.device).share_memory_()) for state in states]
    
    count,get,transfer1,policy_time,transfer2,put = 0,0,0,0,0,0
    while True:
        if time.time() - last_model_load > model_reload_interval:
            last_model_load = reload(count,get,transfer1,policy_time,transfer2,put,policy)
            count,get,transfer1,policy_time,transfer2,put = 0,0,0,0,0,0

        time_0 = time.time()
        (my_id, obs_model) = q_main.get()
        time_1 = time.time()
        state = states[my_id]
        obs_model = map_structure(obs_model, lambda x: x.to(policy.device))
        time_2 = time.time()
        action_distr, metrics, new_state = policy(obs_model,state)
        time_2_5 = time.time()
        metrics = map_structure(metrics, lambda x: x.to("cpu"))
        logits = action_distr.logits.to("cpu")
        time_3 = time.time()
        states[my_id] = new_state
        q_clients[my_id].put((logits, {'policy_value': metrics["policy_value"]}))
        time_4 = time.time()
        count+=1
        get+=time_1-time_0
        transfer1+=time_2-time_1
        policy_time+=time_2_5-time_2
        transfer2+=time_3-time_2_5
        put+=time_4-time_3
        #print("times get,transfer,policy,transfer,put :",time_1-time_0, time_2-time_1,time_2_5-time_2, time_2_5-time_2, time_4-time_3)

def reload(count,get,transfer1,policy_time,transfer2,put,policy):
    while True:
        start_time = time.time()
        #info(f'begin trying to load model checkpoint')
        # takes ~10sec to load checkpoint
        model_step = mlflow_load_checkpoint(policy.model, map_location=policy.device)  # type: ignore
        if model_step:
            info(f'Inference loaded model checkpoint {model_step} t={time.time()-start_time}')
            info(f"times count,get,transfer1,policy,transfer2,put :{count} {get:.2f} {transfer1:.2f} {policy_time:.2f} {transfer2:.2f} {put:.2f}")
            count,get,transfer1,policy_time,transfer2,put = 0,0,0,0,0,0
            return time.time()
        else:
            debug('Inference model checkpoint not found, waiting...')
            time.sleep(10)

def in_process(q_main, q_in, inference_batch_size):
    while True:
        batch_ids = []
        batch_obs = []
        batch_state = []
        
        for i in range(inference_batch_size):
            #info(f'Waiting for requests at {q_main}')
            (my_id, obs_model) = q_main.get()
            #info(f'Got a request for queue {q_main}')
            batch_ids.append(my_id)
            batch_obs.append(obs_model)

            #print("before putting in", my_id, obs_model, states[my_id])
            #state = map_structure(states[my_id], lambda x: x.clone().share_memory_())
            #print("before putting in", my_id, obs_model, state)
            #print("cloning state")
            #for item in state:
            #    item.to(device).share_memory_()
            #batch_state.append(state)
        
        batch_obs = cat_structure_torch(batch_obs,dim=1)#I think B is the second dim here, should be TBICHW
        batch_obs: Dict[str, Tensor] = map_structure(batch_obs, lambda x: x.clone().share_memory_())
        #batch_state = cat_structure(batch_state)
        #print("putting in")
        q_in.put((batch_obs, batch_ids)) 
    
def out_process(q_out, q_clients):
    while True:
        (ids, logits, metrics) = q_out.get()
        #print("post test",ids, logits, metrics, new_state)
        #print("post_processing")
        logits_copy = logits.cpu().clone()
        del logits
        #new_states_clone = map_structure(new_states, lambda x: x.clone())
        #del new_states
        metrics_clone = metrics["policy_value"].clone()
        del metrics
        ids_copy = ids.copy()
        del ids
        for i in range(len(ids_copy)):
            #states[ids_copy[i]] = tuple(x[i:i+1,:] for x in new_states_clone)
            #info(f'responding to {q_clients[ids[i]]}')
            q_clients[ids_copy[i]].put((logits_copy[:,i:i+1,:], {'policy_value': metrics_clone[:,i:i+1]}))

def create_policy(policy_type: str, action_size, model_conf):
    if policy_type == 'remote_network':
        device = torch.device("cuda:7") #TODO: calculate based on worker count or something
        conf = model_conf
        if conf.model == 'dreamer':
            model = Dreamer(conf, device=device)
            print(model)
        else:
            assert False, conf.model
        return NetworkPolicyHost(model, device)
    raise ValueError(policy_type)


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs) -> Tuple[int, dict]:
        return self.action_space.sample(), {}

class NetworkPolicyHost:
    def __init__(self, model: Dreamer, device):
        self.device=device
        self.model = model.to(self.device)

    def __call__(self, obs_model, state) -> Tuple[np.ndarray, dict]:
        #batch = self.preprocess.apply(obs, expandTB=True)
        #obs_model: Dict[str, Tensor] = map_structure(batch, lambda x: torch.from_numpy(x).to(self.device))  # type: ignore
        #obs: Dict[str, Tensor] = map_structure(obs, lambda x: x.to("cuda:1"))  # type: ignore
        #state: Tuple[Tensor,...] = map_structure(state, lambda x: x.to(self.device))

        with torch.no_grad():
            action_distr, new_state, metrics = self.model.inference(obs_model, state, metrics_mean=False)
            #new_state = map_structure(new_state, lambda x: x.to("cpu"))
        #info(f'returning 0 {action_distr}')
        #info(f'returning 1 {metrics}')
        #info(f'returning 2 {new_state}')
        return action_distr, metrics, new_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_main', type=str, required=True)
    parser.add_argument('--worker_id', type=int, default=0)
    args = parser.parse_args()
    main(**vars(args))
