import random
import time
import wandb
import os

WANDB_INIT=False
import logging

log = logging.getLogger(__name__)

def init_wandb(cfg, name, dir, group=None, project="motion-ssl"):
    wandb.login(key=os.environ["WANDB_KEY"])
    attempts = 5
    while True:
        try:
            run = wandb.init(project=project, 
                            entity='motion-ssl-team',
                            config=cfg,
                            name=name,
                            group=group,
                            dir=dir,
                            resume='auto')
            break
        except Exception as e:
            attempts -= 1
            log.error(f"Error in wandb init, retrying ({attempts} attempts left)")
            time.sleep(random.randint(15, 60))
            if attempts <= 0:
                raise e
    
    global WANDB_INIT
    WANDB_INIT=True
    return run

def wandb_log(*args, **kwargs):
    if WANDB_INIT:
        wandb.log(*args, **kwargs)