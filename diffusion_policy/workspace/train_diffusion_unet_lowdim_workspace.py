if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import shutil

import time
import traceback
import platform
import json

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

# ============== debug helpers (no logic changes) ==============
def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _log(msg):
    # single place to control flushing/format
    print(f"[{_now()}] {msg}", flush=True)

def _tensor_summary(x, name="tensor"):
    try:
        if not torch.is_tensor(x):
            return f"{name}: (not tensor) type={type(x)}"
        return (f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
                f"device={x.device} requires_grad={x.requires_grad}")
    except Exception as e:
        return f"{name}: <summary failed: {e}>"

def _dict_summary(d, name="dict", max_items=20):
    try:
        if not isinstance(d, dict):
            return f"{name}: (not dict) type={type(d)}"
        keys = list(d.keys())
        head = keys[:max_items]
        parts = [f"{name}: nkeys={len(keys)} keys(head)={head}"]
        for k in head:
            v = d[k]
            if torch.is_tensor(v):
                parts.append("  " + _tensor_summary(v, name=f"{name}[{k}]"))
            else:
                parts.append(f"  {name}[{k}]: type={type(v)}")
        return "\n".join(parts)
    except Exception as e:
        return f"{name}: <dict summary failed: {e}>"

def _mps_info():
    try:
        return {
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        }
    except Exception as e:
        return {"mps_info_error": str(e)}

def _cuda_info():
    try:
        return {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        }
    except Exception as e:
        return {"cuda_info_error": str(e)}

def _env_info():
    # safe, small diagnostics
    keys = [
        "PYTHONUNBUFFERED",
        "HYDRA_FULL_ERROR",
        "PYGAME_HIDE_SUPPORT_PROMPT",
        "SDL_VIDEODRIVER",
        "SDL_AUDIODRIVER",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "CONDA_PREFIX",
    ]
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}

def _time_block(name):
    # lightweight manual timing without contextlib deps
    t0 = time.time()
    _log(f"[TIMER] start {name}")
    return t0

def _time_block_end(name, t0):
    dt = time.time() - t0
    _log(f"[TIMER] end {name} dt={dt:.3f}s")

# %%
class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        _log("[INIT] workspace init begin")
        _log(f"[INIT] python={platform.python_version()} torch={torch.__version__} platform={platform.platform()}")
        _log(f"[INIT] mps={_mps_info()} cuda={_cuda_info()}")
        _log(f"[INIT] env={json.dumps(_env_info(), indent=2)}")
        _log(f"[INIT] output_dir={self.output_dir}")

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        _log(f"[INIT] seed set to {seed}")

        # configure model
        self.model: DiffusionUnetLowdimPolicy
        t0 = _time_block("[INIT] instantiate cfg.policy")
        self.model = hydra.utils.instantiate(cfg.policy)
        _time_block_end("[INIT] instantiate cfg.policy", t0)
        try:
            nparams = sum(p.numel() for p in self.model.parameters())
            _log(f"[INIT] model instantiated. n_params={nparams}")
        except Exception as e:
            _log(f"[INIT] model param count failed: {e}")

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            t0 = _time_block("[INIT] deepcopy model for EMA")
            self.ema_model = copy.deepcopy(self.model)
            _time_block_end("[INIT] deepcopy model for EMA", t0)
            _log("[INIT] EMA model created")
        else:
            _log("[INIT] EMA disabled")

        # configure training state
        t0 = _time_block("[INIT] instantiate optimizer")
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        _time_block_end("[INIT] instantiate optimizer", t0)
        _log(f"[INIT] optimizer instantiated: {type(self.optimizer)}")

        self.global_step = 0
        self.epoch = 0
        _log("[INIT] workspace init done")

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        _log("[RUN] run() begin")
        _log(f"[RUN] cfg.training.device={cfg.training.device}")
        _log(f"[RUN] cfg.training.num_epochs={cfg.training.num_epochs}")
        _log(f"[RUN] cfg.training.max_train_steps={cfg.training.max_train_steps} "
             f"max_val_steps={cfg.training.max_val_steps}")
        _log(f"[RUN] rollout_every={cfg.training.rollout_every} "
             f"val_every={cfg.training.val_every} sample_every={cfg.training.sample_every} "
             f"checkpoint_every={cfg.training.checkpoint_every}")

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            _log(f"[RUN] resume=True latest_ckpt_path={lastest_ckpt_path}")
            if lastest_ckpt_path.is_file():
                _log(f"[RUN] Resuming from checkpoint {lastest_ckpt_path}")
                t0 = _time_block("[RUN] load_checkpoint")
                self.load_checkpoint(path=lastest_ckpt_path)
                _time_block_end("[RUN] load_checkpoint", t0)
            else:
                _log("[RUN] resume requested but latest checkpoint not found")
        else:
            _log("[RUN] resume=False")

        # configure dataset
        dataset: BaseLowdimDataset
        t0 = _time_block("[RUN] instantiate cfg.task.dataset")
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        _time_block_end("[RUN] instantiate cfg.task.dataset", t0)
        assert isinstance(dataset, BaseLowdimDataset)
        _log(f"[RUN] dataset instantiated: {type(dataset)}")
        try:
            _log(f"[RUN] dataset len={len(dataset)}")
        except Exception as e:
            _log(f"[RUN] dataset len failed: {e}")

        t0 = _time_block("[RUN] create train_dataloader")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        _time_block_end("[RUN] create train_dataloader", t0)
        try:
            _log(f"[RUN] train_dataloader len={len(train_dataloader)} cfg.dataloader={OmegaConf.to_container(cfg.dataloader, resolve=True)}")
        except Exception as e:
            _log(f"[RUN] train_dataloader len/cfg dump failed: {e}")

        t0 = _time_block("[RUN] dataset.get_normalizer()")
        normalizer = dataset.get_normalizer()
        _time_block_end("[RUN] dataset.get_normalizer()", t0)
        _log(f"[RUN] normalizer type={type(normalizer)}")

        # configure validation dataset
        t0 = _time_block("[RUN] dataset.get_validation_dataset()")
        val_dataset = dataset.get_validation_dataset()
        _time_block_end("[RUN] dataset.get_validation_dataset()", t0)
        _log(f"[RUN] val_dataset type={type(val_dataset)}")
        try:
            _log(f"[RUN] val_dataset len={len(val_dataset)}")
        except Exception as e:
            _log(f"[RUN] val_dataset len failed: {e}")

        t0 = _time_block("[RUN] create val_dataloader")
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        _time_block_end("[RUN] create val_dataloader", t0)
        try:
            _log(f"[RUN] val_dataloader len={len(val_dataloader)} cfg.val_dataloader={OmegaConf.to_container(cfg.val_dataloader, resolve=True)}")
        except Exception as e:
            _log(f"[RUN] val_dataloader len/cfg dump failed: {e}")

        self.model.set_normalizer(normalizer)
        _log("[RUN] model normalizer set")
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
            _log("[RUN] ema_model normalizer set")

        # configure lr scheduler
        num_training_steps = (len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every
        _log(f"[RUN] lr_scheduler: num_training_steps={num_training_steps} warmup={cfg.training.lr_warmup_steps} last_epoch={self.global_step-1}")
        t0 = _time_block("[RUN] get_scheduler")
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=self.global_step-1
        )
        _time_block_end("[RUN] get_scheduler", t0)

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            t0 = _time_block("[RUN] instantiate cfg.ema")
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            _time_block_end("[RUN] instantiate cfg.ema", t0)
            _log(f"[RUN] ema instantiated: {type(ema)}")

        # configure env runner
        env_runner: BaseLowdimRunner
        t0 = _time_block("[RUN] instantiate cfg.task.env_runner")
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        _time_block_end("[RUN] instantiate cfg.task.env_runner", t0)
        assert isinstance(env_runner, BaseLowdimRunner)
        _log(f"[RUN] env_runner instantiated: {type(env_runner)}")
        try:
            _log(f"[RUN] env_runner.max_steps={getattr(env_runner,'max_steps',None)}")
        except Exception:
            pass

        # configure logging
        t0 = _time_block("[RUN] wandb.init")
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        _time_block_end("[RUN] wandb.init", t0)
        try:
            wandb.config.update({"output_dir": self.output_dir})
        except Exception as e:
            _log(f"[RUN] wandb.config.update failed: {e}")
        _log("[RUN] wandb configured")

        # configure checkpoint
        _log(f"[RUN] checkpoint.topk.monitor_key={cfg.checkpoint.topk.monitor_key}")
        t0 = _time_block("[RUN] TopKCheckpointManager init")
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )
        _time_block_end("[RUN] TopKCheckpointManager init", t0)

        # device transfer
        device = torch.device(cfg.training.device)
        _log(f"[RUN] device transfer begin: device={device}")
        t0 = _time_block("[RUN] model.to(device)")
        self.model.to(device)
        _time_block_end("[RUN] model.to(device)", t0)
        try:
            _log(f"[RUN] train device: {device} param: {next(self.model.parameters()).device}")
        except Exception as e:
            _log(f"[RUN] printing param device failed: {e}")

        if self.ema_model is not None:
            t0 = _time_block("[RUN] ema_model.to(device)")
            self.ema_model.to(device)
            _time_block_end("[RUN] ema_model.to(device)", t0)

        t0 = _time_block("[RUN] optimizer_to(device)")
        optimizer_to(self.optimizer, device)
        _time_block_end("[RUN] optimizer_to(device)", t0)
        _log("[RUN] device transfer done")

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            _log("[RUN] cfg.training.debug=True -> overriding training knobs (existing behavior)")
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        _log(f"[RUN] JsonLogger path={log_path}")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                _log(f"[EPOCH] begin local_epoch_idx={local_epoch_idx} self.epoch={self.epoch} global_step={self.global_step}")
                step_log = dict()

                # ========= train for this epoch ==========
                train_losses = list()
                t0_epoch_train = _time_block(f"[TRAIN] epoch={self.epoch} dataloader_iter")
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        if batch_idx == 0:
                            _log(f"[TRAIN] first batch fetched batch_idx={batch_idx}")
                            _log(_dict_summary(batch, name="train_batch", max_items=30))

                        # device transfer
                        t0 = time.time()
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        dt = time.time() - t0
                        if batch_idx == 0:
                            _log(f"[TRAIN] first batch moved to device dt={dt:.4f}s")
                            _log(_dict_summary(batch, name="train_batch_on_device", max_items=30))

                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                            _log("[TRAIN] cached train_sampling_batch")

                        # compute loss
                        t0 = time.time()
                        raw_loss = self.model.compute_loss(batch)
                        dt_loss = time.time() - t0

                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        t0 = time.time()
                        loss.backward()
                        dt_bwd = time.time() - t0

                        # step optimizer
                        did_step = False
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            t0 = time.time()
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            did_step = True
                            dt_step = time.time() - t0
                        else:
                            dt_step = 0.0

                        # update ema
                        if cfg.training.use_ema:
                            t0 = time.time()
                            ema.step(self.model)
                            dt_ema = time.time() - t0
                        else:
                            dt_ema = 0.0

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)

                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        if batch_idx % 1 == 0:
                            _log(f"[TRAIN] batch_idx={batch_idx} global_step={self.global_step} "
                                 f"loss={raw_loss_cpu:.6f} did_step={did_step} "
                                 f"dt(loss)={dt_loss:.4f}s dt(bwd)={dt_bwd:.4f}s dt(opt)={dt_step:.4f}s dt(ema)={dt_ema:.4f}s")

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            _log(f"[TRAIN] hit max_train_steps={cfg.training.max_train_steps} at batch_idx={batch_idx} -> break")
                            break

                _time_block_end(f"[TRAIN] epoch={self.epoch} dataloader_iter", t0_epoch_train)

                # at the end of each epoch
                train_loss = np.mean(train_losses) if len(train_losses) > 0 else float("nan")
                step_log['train_loss'] = train_loss
                _log(f"[TRAIN] epoch end self.epoch={self.epoch} mean_train_loss={train_loss}")

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()
                _log(f"[EVAL] policy set to eval. use_ema={cfg.training.use_ema}")

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    _log(f"[ROLL] start epoch={self.epoch} rollout_every={cfg.training.rollout_every} "
                         f"max_steps={getattr(env_runner,'max_steps',None)} runner={type(env_runner)}")
                    t0 = _time_block(f"[ROLL] env_runner.run(epoch={self.epoch})")
                    try:
                        runner_log = env_runner.run(policy)
                        _log(f"[ROLL] done epoch={self.epoch} keys={list(runner_log.keys())}")
                        step_log.update(runner_log)
                    except Exception as e:
                        _log(f"[ROLL] EXCEPTION: {e}")
                        _log(traceback.format_exc())
                        raise
                    finally:
                        _time_block_end(f"[ROLL] env_runner.run(epoch={self.epoch})", t0)
                else:
                    _log(f"[ROLL] skipped epoch={self.epoch} rollout_every={cfg.training.rollout_every}")

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    _log(f"[VAL] start epoch={self.epoch} val_every={cfg.training.val_every} max_val_steps={cfg.training.max_val_steps}")
                    t0_val = _time_block(f"[VAL] val_dataloader(epoch={self.epoch})")
                    try:
                        with torch.no_grad():
                            val_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                           leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    if batch_idx == 0:
                                        _log("[VAL] first batch fetched")
                                        _log(_dict_summary(batch, name="val_batch", max_items=30))
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    loss = self.model.compute_loss(batch)
                                    val_losses.append(loss)
                                    if batch_idx % 1 == 0:
                                        try:
                                            _log(f"[VAL] batch_idx={batch_idx} loss={loss.item():.6f}")
                                        except Exception:
                                            _log(f"[VAL] batch_idx={batch_idx} loss=<non-scalar?>")

                                    if (cfg.training.max_val_steps is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                        _log(f"[VAL] hit max_val_steps={cfg.training.max_val_steps} at batch_idx={batch_idx} -> break")
                                        break

                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                step_log['val_loss'] = val_loss
                                _log(f"[VAL] epoch end val_loss={val_loss}")
                            else:
                                _log("[VAL] no val batches processed (val_losses empty)")
                    except Exception as e:
                        _log(f"[VAL] EXCEPTION: {e}")
                        _log(traceback.format_exc())
                        raise
                    finally:
                        _time_block_end(f"[VAL] val_dataloader(epoch={self.epoch})", t0_val)
                else:
                    _log(f"[VAL] skipped epoch={self.epoch} val_every={cfg.training.val_every}")

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    _log(f"[SAMPLE] start epoch={self.epoch} sample_every={cfg.training.sample_every}")
                    t0 = _time_block(f"[SAMPLE] policy.predict_action(epoch={self.epoch})")
                    try:
                        with torch.no_grad():
                            batch = train_sampling_batch
                            if batch is None:
                                _log("[SAMPLE] train_sampling_batch is None -> skipping sample")
                            else:
                                obs_dict = {'obs': batch['obs']}
                                gt_action = batch['action']

                                _log(_tensor_summary(obs_dict['obs'], name="sample_obs"))
                                _log(_tensor_summary(gt_action, name="sample_gt_action"))

                                result = policy.predict_action(obs_dict)
                                _log(_dict_summary(result, name="sample_result", max_items=50))

                                if cfg.pred_action_steps_only:
                                    pred_action = result['action']
                                    start = cfg.n_obs_steps - 1
                                    end = start + cfg.n_action_steps
                                    gt_action = gt_action[:, start:end]
                                else:
                                    pred_action = result['action_pred']

                                mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                                step_log['train_action_mse_error'] = mse.item()
                                _log(f"[SAMPLE] mse={mse.item()} pred_action={_tensor_summary(pred_action,'pred_action')} gt_action={_tensor_summary(gt_action,'gt_action')}")

                                # release RAM
                                del batch
                                del obs_dict
                                del gt_action
                                del result
                                del pred_action
                                del mse
                    except Exception as e:
                        _log(f"[SAMPLE] EXCEPTION: {e}")
                        _log(traceback.format_exc())
                        raise
                    finally:
                        _time_block_end(f"[SAMPLE] policy.predict_action(epoch={self.epoch})", t0)
                else:
                    _log(f"[SAMPLE] skipped epoch={self.epoch} sample_every={cfg.training.sample_every}")

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    _log(f"[CKPT] start epoch={self.epoch} checkpoint_every={cfg.training.checkpoint_every}")
                    t0 = _time_block(f"[CKPT] checkpoint(epoch={self.epoch})")
                    try:
                        if cfg.checkpoint.save_last_ckpt:
                            _log("[CKPT] save_last_ckpt=True -> save_checkpoint()")
                            self.save_checkpoint()
                        else:
                            _log("[CKPT] save_last_ckpt=False")

                        if cfg.checkpoint.save_last_snapshot:
                            _log("[CKPT] save_last_snapshot=True -> save_snapshot()")
                            self.save_snapshot()
                        else:
                            _log("[CKPT] save_last_snapshot=False")

                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value

                        _log(f"[CKPT] metric_dict keys={list(metric_dict.keys())}")
                        _log(f"[CKPT] monitor_key={topk_manager.monitor_key if hasattr(topk_manager,'monitor_key') else cfg.checkpoint.topk.monitor_key}")

                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        _log(f"[CKPT] topk_ckpt_path={topk_ckpt_path}")

                        if topk_ckpt_path is not None:
                            _log("[CKPT] saving topk checkpoint")
                            self.save_checkpoint(path=topk_ckpt_path)
                    except Exception as e:
                        _log(f"[CKPT] EXCEPTION: {e}")
                        _log(traceback.format_exc())
                        raise
                    finally:
                        _time_block_end(f"[CKPT] checkpoint(epoch={self.epoch})", t0)
                else:
                    _log(f"[CKPT] skipped epoch={self.epoch} checkpoint_every={cfg.training.checkpoint_every}")

                # ========= eval end for this epoch ==========
                policy.train()
                _log("[EPOCH] policy back to train()")

                # end of epoch
                _log(f"[EPOCH] logging end-of-epoch step_log keys={list(step_log.keys())}")
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                _log(f"[EPOCH] end self.epoch={self.epoch} global_step={self.global_step}")

        _log("[RUN] run() end")

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
