import argparse
import json
import os
import time
from logging import info
from distutils.util import strtobool
from multiprocessing import Process, Queue, Manager, get_context
from typing import List

import generator
import inference
import train
from pydreamer.tools import (configure_logging, mlflow_log_params,
                             mlflow_init, print_once, read_yamls)


def launch():
    configure_logging('[launcher]')
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()

    # Config from YAML

    conf = {}
    configs = read_yamls('./config')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])

    # Override config from command-line

    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = lambda x: bool(strtobool(x))
        parser.add_argument(f'--{key}', type=type_, default=value)
    conf = parser.parse_args(remaining)

    # Mlflow

    worker_type, worker_index = get_worker_info()
    is_main_worker = worker_type is None or worker_type == 'learner'#No idea what this is for when implementing multi-node for DDP.
    mlrun = None
    #Can be modified somehow to use just single run in DDP...
    """if(conf.learner_workers > 1):#Ignore is_main_worker here as no idea what it is for
        nodeid = int(os.environ["SLURM_NODEID"])
        if nodeid != 0:
            time.sleep(10)
            while True:
                try:
                    mlrun = mlflow_init(wait_for_resume=True)
                    break
                except:
                    info('secondary launch mlflow run not found, waiting...')
                    time.sleep(10)
        else:
            mlrun = mlflow_init(wait_for_resume=False)
    else:"""
    mlrun = mlflow_init(wait_for_resume=not is_main_worker)
    artifact_uri = mlrun.info.artifact_uri
    mlflow_log_params(vars(conf))

    subprocesses: List[Process] = []
    # Launch policy inference thread

    manager = Manager()
    q_main = None
    policy_main = "network"
    if conf.policy_inference == "remote":
        q_main = manager.Queue()
        policy_main = "remote_network"
    q_clients = []

    # Launch train+eval generators

    for i in range(conf.generator_workers):
        q_self = None
        if conf.policy_inference == "remote":
            q_self = manager.Queue()
            q_clients.append(q_self)
            info(f'generator {i} queue {q_self}')
        if belongs_to_worker('generator', i):
            info(f'Launching train+eval generator {i}')
            p = launch_generator(
                conf.env_id,
                conf,
                save_uri=f'{artifact_uri}/episodes/{i}',
                save_uri2=f'{artifact_uri}/episodes_eval/{i}',
                num_steps=conf.n_env_steps // conf.env_action_repeat // conf.generator_workers,
                limit_step_ratio=conf.limit_step_ratio / conf.generator_workers,
                worker_id=i,
                policy_main=policy_main,
                policy_prefill=conf.generator_prefill_policy,
                num_steps_prefill=conf.generator_prefill_steps // conf.generator_workers,
                split_fraction=0.05,
                q_main=q_main,
                q_self=q_self,
            )
            subprocesses.append(p)

    # Launch train generators

    for i in range(conf.generator_workers_train):
        q_self = None
        if conf.policy_inference == "remote":
            q_self = manager.Queue()
            q_clients.append(q_self)
            info(f'generator {i} queue {q_self}')
        if belongs_to_worker('generator_train', i):
            info(f'Launching train generator {i}')
            p = launch_generator(
                conf.env_id,
                conf,
                f'{artifact_uri}/episodes/{i}',
                num_steps=conf.n_env_steps // conf.env_action_repeat // conf.generator_workers,
                limit_step_ratio=conf.limit_step_ratio / conf.generator_workers,
                worker_id=i,
                policy_main=policy_main,
                policy_prefill=conf.generator_prefill_policy,
                num_steps_prefill=conf.generator_prefill_steps // conf.generator_workers,
                q_main=q_main,
                q_self=q_self,
            )
            subprocesses.append(p)

    # Launch eval generators

    for i in range(conf.generator_workers_eval):
        q_self = None
        if conf.policy_inference == "remote":
            q_self = manager.Queue()
            q_clients.append(q_self)
            info(f'generator {i} queue {q_self}')
        if belongs_to_worker('generator_eval', i):
            info(f'Launching eval generator {i}')
            p = launch_generator(
                conf.env_id_eval or conf.env_id,
                conf,
                f'{artifact_uri}/episodes_eval/{i}',
                worker_id=conf.generator_workers + i,
                policy_main=policy_main,
                metrics_prefix='agent_eval',
                q_main=q_main,
                q_self=q_self,
            )
            subprocesses.append(p)

    # Launch inference process

    for i in range(conf.inference_workers):
        if conf.policy_inference == "remote":
            info(f'Launching remote inference {i} {q_main}')
            p = launch_inference(
                conf.env_id,
                conf,
                worker_id=i,
                policy_main=policy_main,
                q_main=q_main,
                q_clients=q_clients,
            )
            subprocesses.append(p)

    # Launch learner

    for i in range(conf.learner_workers):
        if belongs_to_worker('learner', i):
            info(f'Launching learner {i}')
            p = launch_learner(
                conf,
                worker_id=i
            )
            subprocesses.append(p)

    # Wait & watch

    try:
        while len(subprocesses) > 0:
            check_subprocesses(subprocesses)
            time.sleep(1)
    finally:
        for p in subprocesses:
            p.kill()  # Non-daemon processes (learner, inference) need to be killed


def launch_learner(conf, worker_id):
    p = Process(target=train.run, daemon=False, args=[conf], kwargs=dict(worker_id=worker_id,))
    p.start()
    return p

def launch_inference(env_id,
                     conf,
                     policy_main='remote_network',
                     worker_id=0,
                     q_main=None,
                     q_clients=None,
                     ):
    
    context = get_context("spawn")
    p = context.Process(target=inference.main,
                daemon=False,
                kwargs=dict(
                    env_id=env_id,
                    policy_main=policy_main,
                    worker_id=worker_id,
                    model_conf=conf,
                    q_main=q_main,
                    q_clients=q_clients,
                ))
    p.start()
    return p

def launch_generator(env_id,
                     conf,
                     save_uri,
                     save_uri2=None,
                     policy_main='network',
                     policy_prefill='random',
                     worker_id=0,
                     num_steps=int(1e9),
                     num_steps_prefill=0,
                     limit_step_ratio=0,
                     split_fraction=0.0,
                     metrics_prefix='agent',
                     log_mlflow_metrics=True,
                     q_main=None,
                     q_self=None,
                     ):
    p = Process(target=generator.main,
                daemon=True,
                kwargs=dict(
                    env_id=env_id,
                    save_uri=save_uri,
                    save_uri2=save_uri2,
                    env_time_limit=conf.env_time_limit,
                    env_action_repeat=conf.env_action_repeat,
                    env_no_terminal=conf.env_no_terminal,
                    limit_step_ratio=limit_step_ratio,
                    policy_main=policy_main,
                    policy_prefill=policy_prefill,
                    num_steps=num_steps,
                    num_steps_prefill=num_steps_prefill,
                    worker_id=worker_id,
                    model_conf=conf,
                    log_mlflow_metrics=log_mlflow_metrics,
                    split_fraction=split_fraction,
                    metrics_prefix=metrics_prefix,
                    metrics_gamma=conf.gamma,
                    q_main=q_main,
                    q_self=q_self,
                ))
    p.start()
    return p


def check_subprocesses(subprocesses):
    subp_finished = []
    for p in subprocesses:
        if not p.is_alive():
            if p.exitcode == 0:
                subp_finished.append(p)
                info(f'Generator process {p.pid} finished')
            else:
                raise Exception(f'Generator process {p.pid} died with exitcode {p.exitcode}')
    for p in subp_finished:
        subprocesses.remove(p)


def belongs_to_worker(work_type, work_index):
    """
    In case of distributed workers, checks if this work should execute on this worker.
    If not distributed, return True.
    """
    worker_type, worker_index = get_worker_info()
    return (
        (worker_type is None or worker_type == work_type) and
        (worker_index is None or worker_index == work_index)
    )


def get_worker_info():
    worker_type = None
    worker_index = None

    if 'TF_CONFIG' in os.environ:
        # TF_CONFIG indicates Google Vertex AI run
        tf_config = json.loads(os.environ['TF_CONFIG'])
        print_once('TF_CONFIG is set:', tf_config)
        if tf_config['cluster'].get('worker'):
            # If there are workers in the cluster, then it's a distributed run
            worker_type = {
                'chief': 'learner',
                'worker': 'generator',
            }[str(tf_config['task']['type'])]
            worker_index = int(tf_config['task']['index'])
            print_once('Distributed run detected, current worker is:', f'{worker_type} ({worker_index})')

    return worker_type, worker_index


if __name__ == '__main__':
    launch()
