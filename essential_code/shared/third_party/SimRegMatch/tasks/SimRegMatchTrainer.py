import os, warnings, random, gc, time, csv
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from math import isnan
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from dataloaders import make_semi_loader
from models.resnet_proposed import resnet50
from models.dinov2_proposed import dinov2_vits14

from utils.saver import Saver
from utils.tqdm_config import get_tqdm_config
from torch.utils.data import DataLoader
from baselines.utkface_data import BenchmarkContext, DictDataset
from baselines.benchmark_io import checkpoint_size, write_history, write_json, write_npz
from data_processing.utkface_protocol import dataloader_generator, seed_dataloader_worker, loader_metadata, runtime_metadata


class SimRegMatchTrainer(object):
    def __init__(self, args):
        self.args = args
        
        self.benchmark = bool(self.args.benchmark_manifest)
        if self.benchmark:
            self.experiment_dir = self.args.benchmark_output_dir
            os.makedirs(self.experiment_dir, exist_ok=True)
            self.context = BenchmarkContext(os.environ['UTKFACE_BENCHMARK_ROOT'], self.args.data_dir, self.args.benchmark_manifest)
            self.y_mean, self.y_std = self.context.mean, self.context.std
        else:
            self.saver = Saver(self.args)
            self.saver.save_experiment_config(self.args)
            self.experiment_dir = self.saver.experiment_dir
        print(self.experiment_dir)
        
        self.result_dir = os.path.join(self.experiment_dir, 'csv')
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.experiment_dir)
        if self.benchmark:
            def loader(dataset, role, shuffle, drop_last):
                return DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle,
                                  num_workers=self.args.num_workers, drop_last=drop_last,
                                  worker_init_fn=seed_dataloader_worker,
                                  generator=dataloader_generator(self.args.seed, role), pin_memory=True)
            self.labeled_loader = loader(DictDataset(self.context, 'labeled'), 'labeled', True, True)
            self.unlabeled_loader = loader(DictDataset(self.context, 'unlabeled', 'weak_strong'), 'unlabeled', True, True)
            self.valid_loader = loader(DictDataset(self.context, 'validation', 'evaluation'), 'validation', False, False)
            self.test_loader = loader(DictDataset(self.context, 'test', 'evaluation'), 'test', False, False)
            self.benchmark_loaders = {
                role: loader_metadata(seed=self.args.seed, role=role, num_workers=self.args.num_workers,
                                      batch_size=self.args.batch_size, shuffle=shuffle, drop_last=drop_last,
                                      sampler='RandomSampler' if shuffle else 'SequentialSampler', pin_memory=True)
                for role, shuffle, drop_last in (
                    ('labeled', True, True), ('unlabeled', True, True),
                    ('validation', False, False), ('test', False, False))
            }
        else:
            self.labeled_loader, self.unlabeled_loader, self.valid_loader, self.test_loader = make_semi_loader(self.args, num_workers=0)
        
        model_factory = dinov2_vits14 if self.args.model == 'dinov2_vits14' else resnet50
        model_kwargs = {'dropout': self.args.dropout}
        if self.args.model == 'resnet50': model_kwargs['pretrained'] = self.benchmark
        self.model = model_factory(**model_kwargs).to(self.args.cuda)
        
        formal_dataset = self.context.dataset if self.benchmark else None
        if self.args.model == 'dinov2_vits14':
            self.optimizer = torch.optim.AdamW([
                {'params': self.model.backbone.parameters(), 'lr': 1e-5},
                {'params': self.model.linear.parameters(), 'lr': 1e-4},
            ], weight_decay=0.05)
        elif formal_dataset == 'chexchonet_lvidd':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        elif formal_dataset == 'rsna_bone_age':
            head = list(self.model.linear.parameters())
            head_ids = {id(p) for p in head}
            encoder = [p for p in self.model.parameters() if id(p) not in head_ids]
            self.optimizer = torch.optim.Adam([{'params': encoder, 'lr': 1e-4},
                                               {'params': head, 'lr': 1e-3}], weight_decay=0.0)
        elif self.args.optimizer.lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                              lr=self.args.lr,
                                              momentum=self.args.momentum,
                                              weight_decay=self.args.weight_decay)
        self.scheduler = None
        if formal_dataset in ('chexchonet_lvidd', 'rsna_bone_age'):
            self.scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)
                              if self.args.model == 'dinov2_vits14' else
                              torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1))

        if self.args.loss=='mse':
            self.criterion = nn.MSELoss().to(self.args.cuda)
            self.criterion_unlabel = nn.MSELoss(reduction='none').to(self.args.cuda)
        elif self.args.loss=='l1':
            self.criterion = nn.L1Loss().to(self.args.cuda)
            self.criterion_unlabel = nn.L1Loss(reduction='none').to(self.args.cuda)
        
        self.args.best_valid_loss = np.inf
        self.args.best_valid_epoch = 0
        self.cnt_train, self.cnt_valid = 0, 0
        self.history = []
        self.audit_epochs = {
            int(value) for value in self.args.reliability_audit_epochs.split(',')
            if value.strip()
        }
        if self.args.reliability_audit_dir is not None and not self.benchmark:
            raise ValueError('--reliability_audit_dir requires --benchmark_manifest')
        self.started = time.time()
        if self.args.cuda.type == 'cuda': torch.cuda.reset_peak_memory_stats(self.args.cuda)

    def reorient_labeled_batches(self):
        """Supply ground truth for every canonical unlabeled-loader step."""
        if len(self.labeled_loader) == 0:
            raise ValueError('Reorient requires a nonempty labeled loader')
        labeled_iterator = iter(self.labeled_loader)
        for _ in range(len(self.unlabeled_loader)):
            try:
                batch = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(self.labeled_loader)
                batch = next(labeled_iterator)
            yield batch

    def train(self, epoch):
        self.model.train()
        losses_t, losses_l, losses_u = 0.0, 0.0, 0.0
        iter_unlabeled = iter(self.unlabeled_loader)
        
        total_steps = (len(self.unlabeled_loader) if self.args.reorient else
                       len(self.labeled_loader.dataset) // self.args.batch_size)
        with tqdm(**get_tqdm_config(total=total_steps,
                  leave=True, color='green')) as pbar:
            labeled_batches = self.reorient_labeled_batches() if self.args.reorient else self.labeled_loader
            for idx, samples_labeled in enumerate(labeled_batches):
                # Load data
                inputs_l = samples_labeled['input'].to(self.args.cuda)
                labels_l = samples_labeled['label'].to(self.args.cuda)
                if self.benchmark: labels_l = (labels_l - self.y_mean) / self.y_std

                try:
                    samples_unlabeled = next(iter_unlabeled)
                except:
                    iter_unlabeled = iter(self.unlabeled_loader)
                    samples_unlabeled = next(iter_unlabeled)

                inputs_u = samples_unlabeled['weak'].to(self.args.cuda)
                inputs_s = samples_unlabeled['strong'].to(self.args.cuda)
                
                # Predict labeled examples 
                preds_x, vecs_x = self.model(inputs_l)

                # Predict strong-augmented examples 
                preds_w, vecs_w = [], []
                for _ in range(self.args.iter_u):
                    tmp_preds, tmp_vecs = self.model(inputs_u)
                    preds_w.append(tmp_preds.unsqueeze(-1))
                    vecs_w.append(tmp_vecs.unsqueeze(-1))
                
                preds_w = torch.cat(preds_w, dim=-1)
                vecs_w = torch.cat(vecs_w, dim=-1)
                
                # Predict strong-augmented examples
                preds_s, _ = self.model(inputs_s)
                
                # loss calculation for labeled examples
                loss_x = self.criterion(preds_x, labels_l)
                
                # loss calculation for unlabeled examples
                v_mean = torch.mean(preds_w, axis=2) # pseudo labeling based on our model
                vecs_w = torch.mean(vecs_w, axis=2)

                # similarity distribution
                vecs_w, vecs_x = F.normalize(vecs_w, dim=1), F.normalize(vecs_x, dim=1) 
                similaritys = vecs_w @ vecs_x.T
                similaritys = torch.softmax(similaritys/self.args.t, dim=1)
                
                # similarity-based pseudo-label
                v_similarity = similaritys @ labels_l
                
                # pseudo-label calibration
                v_mean = self.args.beta*v_mean + (1-self.args.beta)*v_similarity # beta*modelPL + (1-beta)*simPL
                
                # uncertainty estimation
                v_uncertainty = torch.pow(torch.std(preds_w, axis=2), 2)
                v_uncertainty = torch.sum(v_uncertainty, axis=1)
                
                # pseudo-label filtering
                threshold_used = self.args.threshold
                mask = (v_uncertainty < threshold_used)
                loss_u_per_example = self.criterion_unlabel(v_mean.detach(), preds_s).sum(axis=1)
                loss_u = (loss_u_per_example * mask).sum()/(int(mask.sum()))
                
                self.args.threshold = np.percentile(v_uncertainty.detach().cpu().numpy(), q=self.args.percentile)           
                self.writer.add_scalar(
                    'Threshold',
                    self.args.threshold,
                    global_step=self.cnt_train
                )
                           
                if isnan(loss_u.item()):
                    loss_u = torch.tensor(0).to(self.args.cuda)
                
                loss = loss_x + self.args.lambda_u*loss_u

                del(inputs_s, inputs_u, preds_w, preds_s)
                gc.collect()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if epoch == 1 and idx == 0 and self.benchmark:
                    write_json(os.path.join(self.experiment_dir, 'early_health.json'), {
                        'pid': os.getpid(), 'epoch': epoch, 'step': idx,
                        'loss': loss.item(), 'supervised_loss': loss_x.item(),
                        'unlabeled_loss': loss_u.item(), 'percentile': self.args.percentile,
                        'full_unlabeled_steps': total_steps,
                        'physical_gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'allocated_cuda_bytes': torch.cuda.memory_allocated(self.args.cuda),
                        'reserved_cuda_bytes': torch.cuda.memory_reserved(self.args.cuda),
                        'timestamp_unix': time.time(),
                    })
                
                losses_t += loss.item()
                losses_l += loss_x.item()
                losses_u += loss_u.item()
                
                self.writer.add_scalars(
                    'Train steps',
                    {'Total Loss': losses_t/(idx+1),
                    'Labeled Loss': losses_l/(idx+1),
                    'Unlabeled Loss': losses_u/(idx+1)},
                    global_step=self.cnt_train
                )
                self.cnt_train += 1
                
                description = "%-7s(%3d/%3d) Total: %.4f| Labeled: %.4f| Unlabeled: %.4f"%(
                    'SEMI', idx, total_steps, losses_t/(idx+1), losses_l/(idx+1), losses_u/(idx+1)
                )
                pbar.set_description(description)
                pbar.update(1)  

    def collect_reliability_audit(self, epoch):
        """Evaluate the full ordered unlabeled cohort without changing training state."""
        if self.args.reliability_audit_dir is None or epoch not in self.audit_epochs:
            return
        audit_dir = self.args.reliability_audit_dir
        os.makedirs(audit_dir, exist_ok=True)
        destination = os.path.join(audit_dir, 'diagnostics.csv')
        unlabeled = DictDataset(self.context, 'unlabeled', 'weak_strong')
        labeled = DictDataset(self.context, 'labeled')
        audit_generators = []

        def audit_loader(dataset, generator_seed):
            generator = torch.Generator().manual_seed(generator_seed)
            audit_generators.append(generator)
            return DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=False,
                num_workers=self.args.num_workers, drop_last=False,
                worker_init_fn=seed_dataloader_worker, generator=generator,
                pin_memory=True,
            )

        seed_base = 8_000_000 + self.args.seed * 10_000 + epoch * 10
        unlabeled_loader = audit_loader(unlabeled, seed_base)
        labeled_loader = audit_loader(labeled, seed_base + 1)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        buffer_state = {name: value.detach().clone() for name, value in self.model.named_buffers()}
        was_training = self.model.training
        rows, offset = [], 0
        threshold = float(self.args.threshold)
        cuda_devices = [self.args.cuda.index or 0] if self.args.cuda.type == 'cuda' else []
        try:
            with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                torch.manual_seed(seed_base + 2)
                self.model.train()
                labeled_iter = iter(labeled_loader)
                for samples_u in unlabeled_loader:
                    try:
                        samples_l = next(labeled_iter)
                    except StopIteration:
                        labeled_iter = iter(labeled_loader)
                        samples_l = next(labeled_iter)
                    inputs_u = samples_u['weak'].to(self.args.cuda, non_blocking=True)
                    inputs_s = samples_u['strong'].to(self.args.cuda, non_blocking=True)
                    inputs_l = samples_l['input'].to(self.args.cuda, non_blocking=True)
                    labels_l = samples_l['label'].to(self.args.cuda, non_blocking=True)
                    labels_l = (labels_l - self.y_mean) / self.y_std
                    _, vecs_x = self.model(inputs_l)
                    predictions, representations = [], []
                    for _ in range(self.args.iter_u):
                        prediction, representation = self.model(inputs_u)
                        predictions.append(prediction.unsqueeze(-1))
                        representations.append(representation.unsqueeze(-1))
                    predictions = torch.cat(predictions, dim=-1)
                    representations = torch.cat(representations, dim=-1)
                    strong_prediction, _ = self.model(inputs_s)
                    model_pseudo = predictions.mean(dim=2)
                    mean_representation = representations.mean(dim=2)
                    similarities = F.softmax(
                        F.normalize(mean_representation, dim=1)
                        @ F.normalize(vecs_x, dim=1).T / self.args.t,
                        dim=1,
                    )
                    calibrated_pseudo = (
                        self.args.beta * model_pseudo
                        + (1 - self.args.beta) * (similarities @ labels_l)
                    )
                    uncertainty = predictions.std(dim=2).square().sum(dim=1)
                    mask = uncertainty < threshold
                    loss_values = self.criterion_unlabel(
                        calibrated_pseudo.detach(), strong_prediction
                    ).sum(dim=1)
                    count = len(samples_u['idx'])
                    expected_ids = np.asarray(unlabeled.indices[offset:offset + count])
                    sample_ids = samples_u['idx'].numpy().reshape(-1)
                    if not np.array_equal(sample_ids, expected_ids):
                        raise RuntimeError('Reliability cohort order changed')
                    offset += count
                    true_labels = samples_u['label'].numpy().reshape(-1)
                    uncertainty_values = uncertainty.cpu().numpy().reshape(-1)
                    selected = mask.cpu().numpy().astype(np.int8).reshape(-1)
                    pseudo_years = calibrated_pseudo.cpu().numpy().reshape(-1) * self.y_std + self.y_mean
                    strong_years = strong_prediction.cpu().numpy().reshape(-1) * self.y_std + self.y_mean
                    rows.extend(zip(
                        sample_ids, [epoch] * count, true_labels,
                        uncertainty_values, -uncertainty_values,
                        [threshold] * count, selected, pseudo_years,
                        strong_years, np.abs(pseudo_years - true_labels),
                        loss_values.cpu().numpy().reshape(-1), selected,
                    ))
                    # Preserve the canonical audited q=0.95 behavior exactly.
                    threshold = float(np.percentile(
                        uncertainty_values, q=self.args.percentile
                    ))
        finally:
            for name, value in self.model.named_buffers():
                value.copy_(buffer_state[name])
            self.model.train(was_training)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        if offset != len(unlabeled.indices):
            raise RuntimeError('Reliability audit cohort was not evaluated exactly once')
        previous = []
        if os.path.isfile(destination):
            with open(destination, newline='') as handle:
                previous = list(csv.reader(handle))[1:]
            previous = [row for row in previous if int(row[1]) != epoch]
        temporary = destination + '.tmp'
        with open(temporary, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow([
                'sample_id', 'epoch', 'true_label', 'uncertainty',
                'reliability_score', 'threshold', 'selection_mask',
                'pseudo_label', 'strong_prediction', 'abs_error',
                'unweighted_consistency_loss', 'effective_weight',
            ])
            writer.writerows(previous)
            writer.writerows(rows)
        os.replace(temporary, destination)

    def validation(self, epoch):
        self.model.eval()
        losses_t = 0.0
        
        total_steps = len(self.valid_loader.dataset) // self.args.batch_size + 1
        with tqdm(**get_tqdm_config(total=total_steps,
                                    leave=True, color='blue')) as pbar:
            with torch.no_grad():
              for idx, samples in enumerate(self.valid_loader):
                inputs_l = samples['input'].to(self.args.cuda)
                labels_raw = samples['label'].to(self.args.cuda)
                labels_l = (labels_raw - self.y_mean) / self.y_std if self.benchmark else labels_raw
                
                preds, _ = self.model(inputs_l)
                loss = self.criterion(preds, labels_l)
                                
                losses_t += loss.item()
                
                if idx == 0:
                    labels_total = labels_raw.detach().cpu()
                    preds_total = (preds.detach().cpu() * self.y_std + self.y_mean) if self.benchmark else preds.detach().cpu()
                else:
                    labels_total = torch.cat((labels_total, labels_raw.detach().cpu()), dim=0)
                    preds_total = torch.cat((preds_total,
                                             (preds.detach().cpu() * self.y_std + self.y_mean) if self.benchmark else preds.detach().cpu()), dim=0)

                r2, mae, rmse = self.regression_metrics(labels_total, preds_total)
                self.writer.add_scalars(
                    'Validation steps',
                    {'Loss': losses_t/(idx+1),
                     'MAE': mae,
                     'RMSE': rmse,
                     'R2': r2},
                    global_step=self.cnt_valid
                )
                self.cnt_valid += 1

                desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Valid", idx, total_steps, losses_t/(idx+1), r2, mae, rmse)
                pbar.set_description(desc)
                pbar.update(1)

            desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Valid", epoch, self.args.epochs, losses_t/(idx+1), r2, mae, rmse)
            pbar.set_description(desc)

            losses_t /= (idx+1)
            self.history.append({'epoch': epoch, 'training_loss': '', 'validation_loss': losses_t,
                                 'validation_mae': mae, 'validation_r2': r2,
                                 'learning_rate': self.optimizer.param_groups[0]['lr'],
                                 'best_so_far': int(mae < self.args.best_valid_loss),
                                 'elapsed_seconds': time.time() - self.started})
            if self.args.best_valid_loss > mae:
                self.args.best_valid_loss = mae
                self.args.best_valid_epoch = epoch
                
                self.args.valid_r2 = str(r2)
                self.args.valid_mae = str(mae)
                self.args.valid_rmse = str(rmse)
                
                labels_total, preds_total = labels_total.numpy(), preds_total.numpy()

                labels_total, preds_total = pd.DataFrame(labels_total), pd.DataFrame(preds_total)
                df = pd.concat([labels_total, preds_total], axis=1)
                df.columns = ['Real', 'Pred']
                
                df.to_csv(os.path.join(self.result_dir, f'valid_{str(epoch)}.csv'), index=False)
                if self.benchmark:
                    torch.save({'model': self.model.state_dict(), 'epoch': epoch,
                                'validation_mae_years': mae, 'mean': self.y_mean, 'std': self.y_std,
                                'optimizer': self.optimizer.state_dict(), 'config': vars(self.args),
                                'loader_metadata': self.benchmark_loaders},
                               os.path.join(self.experiment_dir, 'best.pt'))
                else:
                    torch.save(self.model.state_dict(), os.path.join(self.experiment_dir, 'best_model.pth'))
                
                if not self.benchmark: self.saver.save_experiment_config(self.args)

    def inference(self, epoch):
        checkpoint_path = os.path.join(self.experiment_dir, 'best.pt' if self.benchmark else 'best_model.pth')
        weight = torch.load(checkpoint_path, map_location=self.args.cuda, weights_only=False)
        self.model.load_state_dict(weight['model'] if self.benchmark else weight)
        if self.benchmark:
            assert weight['epoch'] == self.args.best_valid_epoch
            assert weight['validation_mae_years'] == self.args.best_valid_loss
        
        self.model.eval()
        losses_t = 0.0
        
        total_steps = len(self.test_loader.dataset) // self.args.batch_size +1
        with tqdm(**get_tqdm_config(total=total_steps,
                                    leave=True, color='red')) as pbar:
            with torch.no_grad():
              for idx, samples in enumerate(self.test_loader):
                inputs_l = samples['input'].to(self.args.cuda)
                labels_raw = samples['label'].to(self.args.cuda)
                labels_l = (labels_raw - self.y_mean) / self.y_std if self.benchmark else labels_raw
                
                preds, _ = self.model(inputs_l)
                loss = self.criterion(preds, labels_l)
                                
                losses_t += loss.item()
                
                if idx == 0:
                    labels_total = labels_raw.detach().cpu()
                    preds_total = (preds.detach().cpu() * self.y_std + self.y_mean) if self.benchmark else preds.detach().cpu()
                else:
                    labels_total = torch.cat((labels_total, labels_raw.detach().cpu()), dim=0)
                    preds_total = torch.cat((preds_total,
                                             (preds.detach().cpu() * self.y_std + self.y_mean) if self.benchmark else preds.detach().cpu()), dim=0)

                r2, mae, rmse = self.regression_metrics(labels_total, preds_total)
                desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Test", idx, total_steps, losses_t/(idx+1), r2, mae, rmse)
                pbar.set_description(desc)
                pbar.update(1)

            desc = "%-7s(%5d/%5d) Loss: %.4f| R^2: %.4f| MAE: %.4f| RMSE: %.4f "%("Test", epoch, self.args.epochs, losses_t/(idx+1), r2, mae, rmse)
            pbar.set_description(desc)

            r2, mae, rmse = self.regression_metrics(labels_total, preds_total)
            losses_t /= (idx+1)
            
            self.args.best_test_loss = losses_t
            
            self.args.test_r2 = str(r2)
            self.args.test_mae = str(mae)
            self.args.test_rmse = str(rmse)
            
            labels_array, predictions_array = labels_total.numpy(), preds_total.numpy()

            labels_frame = pd.DataFrame(labels_array)
            predictions_frame = pd.DataFrame(predictions_array)
            df = pd.concat([labels_frame, predictions_frame], axis=1)
            df.columns = ['Real', 'Pred']
            
            df.to_csv(os.path.join(self.result_dir, f'test_{str(epoch)}.csv'), index=False)
            
            if self.benchmark:
                indices = np.asarray(self.context.indices('test'), dtype=np.int64)
                write_npz(os.path.join(self.experiment_dir, 'test_predictions.npz'), cohort_indices=indices,
                          relative_paths=np.asarray([self.context.relative_path(i) for i in indices]),
                          predictions=predictions_array[:, 0], targets=labels_array[:, 0],
                          predictions_years=(predictions_array[:, 0] if self.context.target_unit == 'years' else np.asarray([], dtype=np.float32)),
                          targets_years=(labels_array[:, 0] if self.context.target_unit == 'years' else np.asarray([], dtype=np.float32)))
                write_npz(os.path.join(self.experiment_dir, 'analysis_snapshot.npz'),
                          cohort_indices=np.asarray([], dtype=np.int64), note=np.asarray(['post-hoc diagnostics deferred']))
                write_history(os.path.join(self.experiment_dir, 'history.csv'), self.history)
                write_json(os.path.join(self.experiment_dir, 'config.json'), vars(self.args))
                write_json(os.path.join(self.experiment_dir, 'metrics.json'), {'best_epoch': self.args.best_valid_epoch,
                           'validation_mae': self.args.best_valid_loss, 'test_mae': mae, 'test_r2': r2})
                write_json(os.path.join(self.experiment_dir, 'metadata.json'), {
                    'method': 'simregmatch', 'algorithm_identity': 'SimRegMatch official algorithm with audited benchmark adapter.',
                    'status': 'complete',
                    'plugin': 'reorient' if self.args.reorient else None,
                    'supervised_recycling': bool(self.args.reorient),
                    'preserve_full_unlabeled_trajectory': bool(self.args.reorient),
                    'combined_same_optimizer_update': True,
                    'additional_supervised_optimizer_step': False,
                    'percentile_argument_to_numpy': self.args.percentile,
                    'labeled_ratio': self.args.labeled_ratio,
                    'completed_epochs': len(self.history),
                    'manifest_path': str(self.context.manifest_path), 'checkpoint_reloaded': True,
                    'test_evaluations': 1, 'test_used_for_selection': False,
                    'test_model_inference_count': 1,
                    'best_epoch': self.args.best_valid_epoch, 'validation_mae': self.args.best_valid_loss,
                    'checkpoint_size_bytes': checkpoint_size(checkpoint_path), 'runtime': runtime_metadata(),
                    'wall_clock_seconds': time.time() - self.started,
                    'peak_allocated_cuda_bytes': torch.cuda.max_memory_allocated(self.args.cuda),
                    'peak_reserved_cuda_bytes': torch.cuda.max_memory_reserved(self.args.cuda),
                    'model': (self.model.pretrained_identifier if self.args.model == 'dinov2_vits14'
                              else 'ImageNet-pretrained official ResNet-50 with compatible torchvision backbone weights'),
                    'target_backbone': self.args.model,
                    'pretrained_backbone_identifier': (self.model.pretrained_identifier if self.args.model == 'dinov2_vits14'
                                                       else 'torchvision.ResNet50_Weights.IMAGENET1K_V1'),
                    'native_views': 'weak+strong', 'loaders': self.benchmark_loaders,
                    **self.context.protocol_metadata(),
                })
            else:
                self.saver.save_experiment_config(self.args)

    @staticmethod
    def regression_metrics(reals, preds):
        reals, preds = reals.numpy(), preds.numpy()
        
        r2, mae = r2_score(reals, preds), mean_absolute_error(reals, preds)
        rmse = np.sqrt(mean_squared_error(reals, preds))
        
        return r2, mae, rmse
