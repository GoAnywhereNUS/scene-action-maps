import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl

from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix

import numpy as np

def log_sinkhorn(log_alpha, n_iter, converge=False, return_full=False):
    prev_alpha = None

    zero_padding = nn.ZeroPad2d((0, 1, 0, 1))
    log_alpha_padded = zero_padding(log_alpha)

    for it in range(n_iter):
        # Row normalization
        log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
            dim=1)

        # Column normalization
        log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
            dim=2)

        if converge:
            if prev_alpha is not None:
                abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                abs_delta = torch.sum(abs_dev, dim=[1, 2])
                print(it, " Sinkhorn delta: ", torch.max(abs_delta))
                if torch.max(abs_delta) < 1e-6:
                    break
            prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

    if return_full:
        return log_alpha_padded.exp()

    perm_mat = log_alpha_padded[:, :-1, :-1]
    return perm_mat.exp()

def matching(alpha):
    row, col = linear_sum_assignment(-alpha)
    permutation_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
    return torch.from_numpy(permutation_matrix)

def hard_sinkhorn_perms(log_alphas, n_iter):
    soft_perms = log_sinkhorn(log_alphas, n_iter=n_iter, converge=True)
    hard_perms = torch.round(soft_perms)
    # print("@@@@@@@@@@@@@@@")
    # print(torch.round(soft_perms, decimals=2))
    return hard_perms, soft_perms

class EdgeMatcher(pl.LightningModule):
    def __init__(self, misc_hparams=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.im_feature_size = 1000
        self.K = 10
        self.n_behaviours = 6
        self.n_sink_iter = 20
        self.reg_coeff = 0.5
        # self.neg_scale_factor = misc_hparams.neg_scale_factor

        self.adapter = nn.Sequential(
            # nn.Conv2d(2, 3, 1, padding=0)
            nn.Conv2d(4, 3, 1, padding=0)
        )
        # self.backbone = torch.hub.load('pytorch/vision:v0.12.0', 'mobilenet_v2', pretrained=False)
        self.backbone = torch.hub.load('pytorch/vision:v0.12.0', 'resnet50', pretrained=False)
        self.scoring = nn.Sequential(
            nn.Linear(self.im_feature_size, 256, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.ReLU(True),
            nn.BatchNorm1d(16),
            nn.Linear(16, self.n_behaviours)
        )
        # self.thresholding = nn.Sequential(
        #     nn.Linear(self.n_behaviours, self.n_behaviours + 1, bias=False)
        # )

        # self.cost_mat_dim = (self.n_behaviours + 1) * (self.K + 1)
        # self.thresholding = nn.Sequential(
        #     nn.Linear(self.cost_mat_dim, self.K * (self.n_behaviours + 1)),
        #     nn.ReLU(True),
        #     nn.Linear(self.K * (self.n_behaviours + 1), self.K * (self.n_behaviours + 1))
        # )

        # self.criterion = nn.MSELoss(reduction='sum')
        # self.xent = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch):
        map_im, keypoint_ims = batch
        keypoint_ims = keypoint_ims.transpose(0, 1).contiguous()

        cost_mat = []
        for kp_im in keypoint_ims:
            kp_im = torch.unsqueeze(kp_im, 1)
            inputs = torch.cat([map_im, kp_im], dim=1)
            # inputs = torch.stack([map_im, kp_im], dim=1)
            inputs = self.adapter(inputs)
            feats = self.backbone(inputs)
            scores = self.scoring(feats)
            cost_mat.append(scores)

        log_alphas = torch.stack(cost_mat, dim=1)
        if self.training:
            perms = log_sinkhorn(log_alphas, n_iter=self.n_sink_iter, return_full=True)

            if torch.any(torch.isnan(log_alphas)):
                raise Exception("NaN detected in cost matrix")
            if torch.any(torch.isnan(perms)):
                raise Exception("NaN detected in permutation matrix")

            return perms[:, :-1, :-1], log_alphas.exp()

            # unnorm_scores = self.thresholding(perms.view(-1, self.n_behaviours))
            # unnorm_scores = self.thresholding(perms.view(-1, self.cost_mat_dim)).view(-1, self.n_behaviours + 1)
            # return perms[:, :-1, :-1], unnorm_scores
        else:
            # matched = [matching(alpha) for alpha in log_alphas.cpu().detach().numpy()]
            # perms = torch.stack(matched).float().to(log_alphas.device)
            
            perms, soft_perms = hard_sinkhorn_perms(log_alphas, n_iter=500)
            return perms, soft_perms

            # # unnorm_scores = self.thresholding(perms.view(-1, self.n_behaviours))
            # unnorm_scores = self.thresholding(perms.view(-1, self.cost_mat_dim)).view(-1, self.n_behaviours + 1)
            # norm_scores = F.softmax(unnorm_scores.view(-1, self.K, self.n_behaviours + 1), dim=2)
            # return perms, soft_perms, norm_scores

        # return perms, log_alphas.exp()

    def configure_optimizers(self):
        # opt = optim.Adam(self.parameters(), lr=1e-3)
        opt = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        return opt

    def training_step(self, batch, batch_idx):
        maps, keypoints, matches = batch
        perms, unnorm_scores = self.forward((maps, keypoints))
        matches = matches.bool()
        
        # Compute loss
        log_perms = torch.log(perms)
        # masked1 = log_perms * matches
        # loss = -torch.sum(masked1) / torch.sum(matches)

        masked_pos = log_perms * matches
        masked_neg = perms * ~matches
        # print(torch.sum(masked_neg) / torch.sum(~matches), torch.sum(masked_pos) / torch.sum(matches))
        loss = (self.neg_scale_factor * torch.sum(masked_neg) / torch.sum(~matches)) \
             - torch.sum(masked_pos) / torch.sum(matches)

        # masked2 = log_perms * torch.cat((matches[:, 3:], matches[:, :3]), dim=1)
        # loss = -0.5 * (torch.sum(masked1) + torch.sum(masked2)) / torch.sum(matches)

        # match_scores = matches.view(-1, self.n_behaviours)
        # not_matched = match_scores < 0.5
        # match_scores = torch.cat((match_scores, torch.all(not_matched, 1, keepdim=True)), dim=1)
        # match_indices = match_scores.argmax(1)
        # xent_loss = self.xent(unnorm_scores, match_indices)
        # # print("Xent: ", xent_loss.detach().cpu().numpy(), loss.detach().cpu().numpy())
        # loss += xent_loss

        # mse_normal = torch.sum(torch.sqrt((perms - matches)**2))
        # mse_flipped = torch.sum(torch.sqrt((torch.cat((perms[:, 3:], perms[:, :3]), dim=1) - matches)**2))
        # mse = 0.5 * (mse_normal + mse_flipped)

        # # mse = torch.sum(torch.sqrt((perms - matches)**2))
        # reg = -self.reg_coeff * torch.sum(perms) * ((self.K + self.n_behaviours) / (self.K * self.n_behaviours))
        # loss = mse + reg
        # print(">>> ", mse, torch.sum(perms), loss)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        perms, soft_perms = self.forward(batch)

        # _, soft_perms, norm_scores = self.forward(batch)
        # max_scores = norm_scores.argmax(2)
        # perms = torch.zeros(norm_scores.shape, device=self.device).scatter(2, max_scores.unsqueeze(2), 1.0)
        # perms = perms[:, :, :-1]
        
        return perms, soft_perms

    def test_step(self, batch, batch_idx):
        maps, keypoints, matches = batch
        perms, soft_perms = self.forward((maps, keypoints))

        # _, soft_perms, norm_scores = self.forward((maps, keypoints))
        # max_scores = norm_scores.argmax(2)
        # perms = torch.zeros(norm_scores.shape, device=self.device).scatter(2, max_scores.unsqueeze(2), 1.0)
        # # perms = perms[:, :, :-1]
        
        for i in range(matches.shape[0]):
            print(">>>>>>>")
            # print(norm_scores[i])
            print(perms[i].T)
            print(matches[i].T)
            print("<<<<<<<")

        soft_perms = soft_perms.cpu().numpy()
        perms = perms.cpu().numpy()
        matches = matches.cpu().numpy()
        # norm_scores = norm_scores.cpu().numpy()
        # np.savez('results.npz', soft_perms=soft_perms, perms=perms, matches=matches, norm_scores=norm_scores)
        np.savez('results.npz', soft_perms=soft_perms, perms=perms, matches=matches)

        return 0.0


class EdgeMatcherRepr(pl.LightningModule):
    def __init__(self, temperature=0.1, lr=1e-2, gamma=0.98, misc_hparams=None):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.gamma = gamma
        self.temperature = temperature
        self.feature_size = 1000

        self.adapter = nn.Sequential(
            nn.Conv2d(4, 3, 1, padding=0)
        )
        # self.encoder = torch.hub.load('pytorch/vision:v0.12.0', 'mobilenet_v2', pretrained=False)
        self.encoder = torch.hub.load('pytorch/vision:v0.12.0', 'resnet50', pretrained=False)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size, bias=False)
        )

    def forward(self, batch):
        inputs = self.adapter(batch)
        feats = self.encoder(inputs)
        projs = self.projection_head(feats)
        return projs

    def configure_optimizers(self):
        # opt = optim.Adam(self.parameters(), lr=self.lr)
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        sch = optim.lr_scheduler.ExponentialLR(opt, gamma=self.gamma, verbose=True)
        return [opt], [sch]
        # return opt

    def supcon_loss(self, positives_mask, feats, temp=0.1, reducer="mean"):
        feats_norm = F.normalize(feats, p=2.0, dim=1)
        logits = torch.matmul(feats_norm, feats_norm.T) / temp

        # Sum over all positive elements
        positives_sum = torch.sum(positives_mask * logits, dim=1)
        positives_count = torch.sum(positives_mask, dim=1)

        # Compute the log denominator for each sample, by summing over
        # dot products with all other samples excluding itself (i.e.
        # excluding the diagonal elements)
        logits.fill_diagonal_(-torch.inf)
        log_denom = torch.logsumexp(logits, dim=1)
        sample_loss = log_denom - (positives_sum / positives_count)
        
        if reducer == "mean":
            loss = torch.mean(sample_loss)
        elif reducer == "sum":
            loss = torch.sum(sample_loss)
        else:
            raise NotImplementedError

        return loss         

    def training_step(self, batch, batch_idx):
        views, _, positives_mask, num_views = batch
        batch_size = views.shape[0]

        feats = self.forward(views)
        loss = self.supcon_loss(positives_mask, feats, temp=self.temperature)
        
        self.log(
            "train_loss_step", loss, on_step=True, on_epoch=True, 
            prog_bar=True, logger=True, batch_size=batch_size
        )
        return loss


class ChangepointNetXent(pl.LightningModule):
    def __init__(self, encoder=None, misc_hparams=None):
        super().__init__()
        self.save_hyperparameters()

        self.feature_size = 1000
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = torch.hub.load('pytorch/vision:v0.12.0', 'mobilenet_v2', pretrained=False)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        # samples = batch.unsqueeze(1).repeat(1, 3, 1, 1)
        # feats = self.encoder(samples)
        feats = self.encoder(batch)
        scores = self.fc(feats)
        return F.softmax(scores, dim=1)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-4)
        return opt

    def xent_loss(self, batch):
        samples, labels = batch
        # feats = self.encoder(samples.unsqueeze(1).repeat(1, 3, 1, 1))
        feats = self.encoder(samples)
        scores = self.fc(feats)
        loss = self.criterion(scores, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.xent_loss(batch)
        self.log("train_loss_step", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        # samples = batch.unsqueeze(1).repeat(1, 3, 1, 1)
        # feats = self.encoder(samples)
        feats = self.encoder(batch)
        scores = self.fc(feats)
        return F.softmax(scores, dim=1)


class ChangepointNetRepr(pl.LightningModule):
    def __init__(self, temperature=0.1, lr=1e-2, gamma=0.98, misc_hparams=None):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.gamma = gamma
        self.temperature = temperature
        self.feature_size = 1000
        self.encoder = torch.hub.load('pytorch/vision:v0.12.0', 'mobilenet_v2', pretrained=False)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size, bias=False)
        )

    def forward(self, batch):
        feats = self.encoder(batch)
        projs = self.projection_head(feats)
        return projs

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        sch = optim.lr_scheduler.ExponentialLR(opt, gamma=self.gamma, verbose=True)
        return [opt], [sch]

    def supcon_loss(self, labels, feats, temp=0.1, reducer="mean"):
        batch_size = feats.shape[0]
        feats_norm = F.normalize(feats, p=2.0, dim=1)
        logits = torch.matmul(feats_norm, feats_norm.T) / temp

        # Mask out positive (similar) elements for each sample in the batch
        # but exclude self-similarity (i.e. diagonal elements)
        labels_2d = torch.unsqueeze(labels, dim=0)
        positives_mask = torch.logical_not(torch.logical_xor(labels_2d, labels_2d.T)).type(torch.float)
        positives_mask.fill_diagonal_(0.0)

        # Sum over all positive elements
        positives_sum = torch.sum(positives_mask * logits, dim=1)
        positives_count = torch.sum(positives_mask, dim=1)

        # Compute the log denominator for each sample, by summing over
        # dot products with all other samples excluding itself (i.e.
        # excluding the diagonal elements)
        logits.fill_diagonal_(-torch.inf)
        log_denom = torch.logsumexp(logits, dim=1)
        sample_loss = log_denom - (positives_sum / positives_count)
        
        if reducer == "mean":
            loss = torch.mean(sample_loss)
        elif reducer == "sum":
            loss = torch.sum(sample_loss)
        else:
            raise NotImplementedError

        return loss

    def training_step(self, batch, batch_idx):
        ims, labels = batch
        feats = self.forward(ims)
        loss = self.supcon_loss(labels, feats, temp=self.temperature)
        self.log("train_loss_step", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

