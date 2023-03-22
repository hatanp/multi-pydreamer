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

from pydreamer.data import MlflowEpisodeRepository
from pydreamer.envs import create_env
from pydreamer.models import *
from pydreamer.models.functions import map_structure, cat_structure, cat_structure_torch
from pydreamer.preprocessing import Preprocessor
from pydreamer.tools import *


def main(env_id='MiniGrid-MazeS11N-v0',
         worker_id=99,
         policy_main='remote_network',
         model_reload_interval=120,
         model_conf=dict(),
         q_main=None,
         q_clients=None,
         inference_batch_size=8,
         ):

    configure_logging(prefix=f'[INFER]', info_color=LogColorFormatter.GREEN)
    info(f'INFERENCE started')
    env = create_env(env_id, model_conf.env_no_terminal, model_conf.env_time_limit, model_conf.env_action_repeat, worker_id)
    print("action_size:", env.action_size)
    # RUN

    last_model_load = 0

    info(f'Main policy: {policy_main}')
    policy = create_policy(policy_main, env.action_size, model_conf)
    is_prefill_policy = False

    while True:
        if time.time() - last_model_load > model_reload_interval:
            while True:
                # takes ~10sec to load checkpoint
                model_step = mlflow_load_checkpoint(policy.model, map_location=policy.device)  # type: ignore
                if model_step:
                    info(f'Inference loaded model checkpoint {model_step}')
                    last_model_load = time.time()
                    break
                else:
                    debug('Inference model checkpoint not found, waiting...')
                    time.sleep(10)

        ids = []
        batch_obs = []
        batch_state = []

        for i in range(inference_batch_size):
            #info(f'Waiting for requests at {q_main}')
            #test = q_main.get()
            (my_id, obs_model, state) = q_main.get()
            #info(f'Got a request for queue {q_main}')
            ids.append(my_id)
            batch_obs.append(obs_model)
            batch_state.append(state)
        #print("shapes:", batch_obs[0].keys(),batch_obs[0]["image"].shape ,batch_state[0][0].shape)
        batch_obs = cat_structure_torch(batch_obs,dim=1)#I think B is the second dim here, should be TBICHW
        batch_state = cat_structure(batch_state)
        #print("shapes:", batch_obs.keys(),batch_obs["image"].shape ,batch_state[0].shape)
        logits, mets, new_state = policy(batch_obs,batch_state)
        #print("shapes:", logits.shape,mets["policy_value"].shape ,new_state[0].shape,new_state[1].shape)
        #print("action_distr.logits:", action_distr.logits.shape, action_distr.logits.device)
        #print("action_distr.probs:", action_distr.probs.shape)
        #print("action_distr.mean:", action_distr.mean.shape)
        #print("action_distr.mode:", action_distr.mode.shape)
        #print("action_distr.param_shape:", action_distr.param_shape)
        #print("returning:", {'policy_value': mets["policy_value"][:,i:i+1]})
        for i in range(inference_batch_size):
            #info(f'responding to {q_clients[ids[i]]}')
            q_clients[ids[i]].put((logits[:,i:i+1,:], {'policy_value': mets["policy_value"][:,i:i+1]}, tuple(x[i:i+1,:] for x in new_state)))
            

def create_policy(policy_type: str, action_size, model_conf):
    if policy_type == 'remote_network':
        conf = model_conf
        if conf.model == 'dreamer':
            model = Dreamer(conf)
        else:
            assert False, conf.model
        preprocess = Preprocessor(image_categorical=conf.image_channels if conf.image_categorical else None,
                                  image_key=conf.image_key,
                                  map_categorical=conf.map_channels if conf.map_categorical else None,
                                  map_key=conf.map_key,
                                  action_dim=action_size,  # type: ignore
                                  clip_rewards=conf.clip_rewards)
        return NetworkPolicyHost(model, preprocess)
    raise ValueError(policy_type)


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, obs) -> Tuple[int, dict]:
        return self.action_space.sample(), {}

class NetworkPolicyHost:
    def __init__(self, model: Dreamer, preprocess: Preprocessor):
        self.device=torch.device("cuda:3")
        self.model = model.to(self.device)
        self.preprocess = preprocess

    def __call__(self, obs_model, state) -> Tuple[np.ndarray, dict]:
        #batch = self.preprocess.apply(obs, expandTB=True)
        #obs_model: Dict[str, Tensor] = map_structure(batch, lambda x: torch.from_numpy(x).to(self.device))  # type: ignore
        #obs: Dict[str, Tensor] = map_structure(obs, lambda x: x.to("cuda:1"))  # type: ignore
        obs_model: Dict[str, Tensor] = map_structure(obs_model, lambda x: x.to(self.device))
        state: Tuple[Tensor,...] = map_structure(state, lambda x: x.to(self.device))

        with torch.no_grad():
            action_distr, new_state, metrics = self.model.inference(obs_model, state, metrics_mean=False)
            metrics = map_structure(metrics, lambda x: x.to("cpu"))
            new_state = map_structure(new_state, lambda x: x.to("cpu"))
        #info(f'returning 0 {action_distr}')
        #info(f'returning 1 {metrics}')
        #info(f'returning 2 {new_state}')
        return action_distr.logits.to("cpu"), metrics, new_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_main', type=str, required=True)
    parser.add_argument('--worker_id', type=int, default=0)
    args = parser.parse_args()
    main(**vars(args))
