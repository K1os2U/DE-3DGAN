import os
import torch
import logging
import argparse
import numpy as np
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from collections import Counter
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from pathlib import Path
from torch import autograd
from sklearn import model_selection
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from preprocessors import DEAPDataset, Sequence
from preprocessors import BinaryLabel
from preprocessors import Raw2TNCF, RemoveBaseline, TNCF2NCF, ChannelToLocation

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0, help='Fold index.')
args = parser.parse_args()

class CFG:
    NUM_EPOCHS = 300
    BATCH_SIZE  = 64
    C_LR        = 1e-5
    WD          = 5e-4
    TEST_SIZE   = 0.2
    FOLD = args.fold

# EEG signal preprocessing
RECEIVED_PARAMS = { "weight_ssl": 0.5}
TRAIL_ID = 'DE-3DGAN_finetune_Classifier_valence' + str(CFG.FOLD)
logger = logging.getLogger(TRAIL_ID)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('logs/{}.log'.format(TRAIL_ID))
logger.addHandler(console_handler)
logger.addHandler(file_handler)
DATASET_BASE_DIR = Path('./eeg_dataset')
DATASET_FOLD_DIR = DATASET_BASE_DIR / 'DEAP'
PREPROCESSED_EEG_DIR = DATASET_FOLD_DIR / 'data_preprocessed_python'
label_preprocessors = {'label': Sequence([BinaryLabel()])}
feature_preprocessors = {
    'feature':
    Sequence([Raw2TNCF(),
              RemoveBaseline(),
              TNCF2NCF(),
              ChannelToLocation()])
}
preprocessors_results = DEAPDataset(
    PREPROCESSED_EEG_DIR, label_preprocessors,
    feature_preprocessors)('./dataset/deap_binary_valence_dataset.pkl')

# Normalization operation
def robust_norm(x, lower=0.01, upper=0.99):
    ql = x.quantile(lower)
    qu = x.quantile(upper)
    x = x.clamp(ql, qu)
    mu = x.mean()
    sigma = x.std() + 1e-6
    return (x - mu) / sigma, mu, sigma

# DE features and raw signal loading
class EEGDataset5D(Dataset):
    def __init__(self, preprocessors_results, de_dir='./DE_feature/'):
        de_list = []
        for f in sorted(os.listdir(de_dir)):
            if not f.endswith('.mat'): continue
            m = loadmat(os.path.join(de_dir,f))
            d = m['data']               
            d = d.reshape(40,120,4,9,9) 
            de_list.append(d)
        de = np.concatenate(de_list,axis=0)

        raw_list, lbl_list = [], []
        for trail in preprocessors_results.keys():
            feat = preprocessors_results[trail]['feature']  
            lab  = preprocessors_results[trail]['label']   
            feat = feat.reshape(40, 60, 128, 9, 9)
            lab = lab.reshape(40, 60)
            raw_list.append(feat)
            lbl_list.append(lab)
        raw = np.concatenate(raw_list,axis=0)
        lbl = np.concatenate(lbl_list,axis=0)

        self.de_seg  = de
        self.raw_seg = raw
        self.lbl_seg = lbl

        assert len(self.de_seg)==len(self.raw_seg)==len(self.lbl_seg)

    def __len__(self):
        return self.de_seg.shape[0]

    def __getitem__(self, idx):
        de = torch.from_numpy(self.de_seg[idx]).float()
        de = de.permute(1,0,2,3).contiguous()
        raw   = torch.from_numpy(self.raw_seg[idx]).float()
        label = torch.tensor(self.lbl_seg[idx]).long()

        return de, raw, label

# Generate signals and real signals loading
class FrameDataset(Dataset):
    def __init__(self, trial_ds, device='cuda'):

        self.trial_ds = trial_ds
        self.device   = device
        G = Generator3D_UNet()
        self.G= G.cuda()
        gan_model_state_dict = torch.load(
            './parameters/DE-3DGAN_3D_UNet_GAN_valence.pth')
        self.G.load_state_dict(gan_model_state_dict['g_model'])
        self.G.eval()

        self.gen_all   = []
        self.raw_all   = []
        self.label_all = []

        with torch.no_grad():
            for idx in range(len(trial_ds)):
                de, raw, label = trial_ds[idx]  
                de, de_mu, de_sigma = robust_norm(de)
                de = de.unsqueeze(0).cuda()
                gen = self.G(de)           
                gen = (gen * de_sigma + de_mu).squeeze(0).cpu()    
                raw = raw                      
                label = label                   

                for t in range(gen.shape[0]):
                    self.gen_all.append(gen[t])     
                    self.raw_all.append(raw[t])     
                    self.label_all.append(label[t]) 

        self.gen   = torch.stack(self.gen_all,   dim=0) 
        self.raw   = torch.stack(self.raw_all,   dim=0)
        self.lbl = torch.tensor(self.label_all)     

    def __len__(self):
        return self.gen.shape[0]

    def __getitem__(self, idx):
        return self.gen[idx], self.raw[idx], self.lbl[idx]

class ResidualConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.SELU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch)
        )
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.act  = nn.SELU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))

# generator structure
class Generator3D_UNet(nn.Module):
    def __init__(self, in_ch=4, base_ch=32, time_len=60):
        super().__init__()
        self.time_len = time_len
        self.enc1 = ResidualConv3d(in_ch,    base_ch)
        self.enc2 = ResidualConv3d(base_ch,  base_ch*2)
        self.enc3 = ResidualConv3d(base_ch*2,base_ch*4)
        self.enc4 = ResidualConv3d(base_ch*4,base_ch*8)
        self.pool = nn.MaxPool3d((2,2,2),(2,2,2))
        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, 2, 2)
        self.dec3 = ResidualConv3d(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, 2)
        self.dec2 = ResidualConv3d(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=(2,2,2), stride=(2,2,2), output_padding=(0,1,1))
        self.dec1 = ResidualConv3d(base_ch*2, base_ch)
        self.out = nn.Conv3d(base_ch, 128, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.act = nn.SELU(inplace=True)

    def forward(self, x):
        mask = (x.abs().sum(dim=(1,2), keepdim=True) > 0).float()
        e1 = self.enc1(x)      # (B, b, 120, 9, 9)
        p1 = self.pool(e1)     # (B, b,  60, 4, 4)
        e2 = self.enc2(p1)     # (B,2b,  60, 4, 4)
        p2 = self.pool(e2)     # (B,2b,  30, 2, 2)
        e3 = self.enc3(p2)     # (B,4b,  30, 2, 2)
        p3 = self.pool(e3)     # (B,4b,  15, 1, 1)
        e4 = self.enc4(p3)     # (B,8b,  15, 1, 1)
        d3 = self.up3(e4)      # (B,4b,  30, 2, 2)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)     # (B,4b,  30, 2, 2)
        d2 = self.up2(d3)      # (B,2b,  60, 4, 4)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)     # (B,2b,  60, 4, 4)
        d1 = self.up1(d2)      # (B, b, 120, 8, 8)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)     # (B, b, 120, 8, 8)
        out = self.out(d1)     # (B,128,120, 8, 8)
        out = F.interpolate(
            out,
            size=(self.time_len, 9, 9),
            mode='trilinear',
            align_corners=False
        )                      # (B,128, 60, 9,9)

        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out * self.alpha
        return out * mask  # (B, 60,128, 9,9)

# Multi scale convolution module
class InceptionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv5x5 = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2,
                                 bias=bias)
        self.conv3x3 = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=bias)
        self.conv1x1 = nn.Conv2d(in_channels,
                                 out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=bias)

    def forward(self, x):
        return self.conv5x5(x) + self.conv3x3(x) + self.conv1x1(x)

class SeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=True):
        super().__init__()
        self.depth = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               groups=in_channels,
                               bias=bias)
        self.point = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=1,
                               stride=stride,
                               padding=0,
                               bias=bias)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return x

# SE-Block module
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              # (B,C,1,1)
            nn.Conv2d(channels, channels//reduction, 1),
            nn.SELU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)   # (B,C,1,1)
        return x * w

# Spatial attention module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size,padding=kernel_size//2,bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        maxc,_ = x.max(dim=1,keepdim=True)  # (B,1,H,W)
        avgc   = x.mean(dim=1,keepdim=True) # (B,1,H,W)
        att    = self.sig(self.conv(torch.cat([maxc,avgc],dim=1)))
        return x * att

# Classifier structure
class Classifier(nn.Module):
    def __init__(self, num_classes, in_channels=4):
        super(Classifier, self).__init__()
        self.layer1 = nn.Conv2d(in_channels,
                                256,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=True)
        self.layer2 = nn.Conv2d(256,
                                128,
                                kernel_size=5,
                                stride=1,
                                padding=2,
                                bias=True)
        self.layer3 = nn.Conv2d(128,
                                64,
                                kernel_size=5,
                                stride=1,
                                padding=2,
                                bias=True)
        self.layer4 = SeparableConv2d(64,
                                      32,
                                      kernel_size=5,
                                      stride=1,
                                      padding=2,
                                      bias=True)
        self.layer5 = InceptionConv2d(32, 16)
        self.se = SEBlock(16, reduction=4)
        self.freq_att = SpatialAttention()
        self.drop = nn.Sequential(nn.Dropout(), nn.SELU())
        self.fc1 = nn.Sequential(nn.Linear(9 * 9 * 16, 1024, bias=True), nn.SELU())
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.drop(out)
        out = self.layer2(out)
        out = self.drop(out)
        out = self.layer3(out)
        out = self.drop(out)
        out = self.layer4(out)
        out = self.drop(out)
        out = self.layer5(out)
        out = self.se(out)
        out = self.freq_att(out)
        out = self.drop(out)
        out = out.view(out.size(0), -1)
        feat = self.fc1(out)
        out = self.fc2(feat)
        return out, feat


def gradient_penalty(model, real, fake):
    device = real.device
    real = real.data
    fake = fake.data
    alpha = torch.rand(real.size(0), *([1] * (len(real.shape) - 1))).to(device)
    inputs = alpha * real + ((1 - alpha) * fake)
    inputs.requires_grad_()
    outputs = model(inputs)
    gradient = autograd.grad(outputs=outputs,
                             inputs=inputs,
                             grad_outputs=torch.ones_like(outputs).to(device),
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]

    gradient = gradient.flatten(1)
    return ((gradient.norm(2, dim=1) - 1)**2).mean()

# Training data separation
def train_test_split(dataset,
                     kfold_split_index_path='./dataset/kfold_split_index.pkl',
                     fold=0,
                     n_splits=5,
                     shuffle=True,
                     seed=520):
    if not os.path.exists(kfold_split_index_path):
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        kfold = model_selection.StratifiedKFold(n_splits=n_splits,
                                                shuffle=shuffle,
                                                random_state=seed)
        index_dict = {}
        for i, (train_index, test_index) in enumerate(
                kfold.split(indices, dataset.label_list)):
            index_dict[i] = {
                'train_index': train_index,
                'test_index': test_index
            }
        with open(kfold_split_index_path, 'wb') as file:
            pkl.dump(index_dict, file)
    else:
        with open(kfold_split_index_path, 'rb') as file:
            index_dict = pkl.load(file)

    index_split = index_dict[fold]
    train_index, test_index = index_split['train_index'], index_split[
        'test_index']
    trian_dataset = Subset(dataset, train_index)
    test_dataset = Subset(dataset, test_index)

    return trian_dataset, test_dataset

# Fine tune the classifier using artificial data obtained from the generator
class Trainer():
    def __init__(self, c_model, g_model, ds, trainer_kwargs={'max_epochs':10}):
        super().__init__()
        self.c_model = c_model.cuda()
        self.g_model = g_model.cuda()

        self._loss_fn_ce  = nn.CrossEntropyLoss()
        self._loss_fn_mse = nn.MSELoss()
        self._optimizer_c_model = torch.optim.Adam(
            c_model.parameters(),
            lr=CFG.C_LR,
            weight_decay=CFG.WD
        )
        self._trainer_kwargs = trainer_kwargs
        self.all_preds = []
        self.all_labels = []
        self.metric_history = []

        tr_ds, val_ds = train_test_split(ds, fold=CFG.FOLD)
        self._train_dataloader = DataLoader(
            tr_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=False)
        self._val_dataloader   = DataLoader(
            val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, drop_last=False)

    def _accuracy(self, logits, target):
        preds = torch.argmax(logits, dim=1)
        return (preds == target).float().mean().item()

    def training_step_c_model(self, batch, batch_idx):
        for p in self.c_model.parameters():
            p.requires_grad = True
        self._optimizer_c_model.zero_grad()
        de, raw, label = batch
        de, raw, label = de.cuda(), raw.cuda(), label.cuda()

        y_hat, x_feat = self.c_model(raw)
        loss = self._loss_fn_ce(y_hat, label)
        _, aug_feat= self.c_model(de)
        ssl = F.mse_loss(x_feat, aug_feat, reduction='none').mean(dim=1)
        loss_ssl = ssl.mean()
        loss = loss + RECEIVED_PARAMS['weight_ssl'] * loss_ssl
        loss.backward()
        self._optimizer_c_model.step()
        return loss

    def validation_step(self, batch, batch_idx):
        z, x, y = batch
        z, x, y = z.cuda(), x.cuda(), y.cuda()
        with torch.no_grad():
            logits, _ = self.c_model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]  # 取正类概率
        self.all_preds.append(probs.cpu().numpy())
        self.all_labels.append(y.cpu().numpy())
        return logits.cpu(), z.cpu()

    def validation_epoch_end(self):
        preds = np.concatenate(self.all_preds)
        labels = np.concatenate(self.all_labels)
        pred_labels = (preds >= 0.5).astype(int)
        acc = (pred_labels == labels).mean()
        f1 = f1_score(labels, pred_labels)
        auc = roc_auc_score(labels, preds)

        self.all_preds.clear()
        self.all_labels.clear()
        metrics = {'acc': acc, 'f1': f1, 'auc': auc}
        self.metric_history.append(metrics)
        logger.info(f"[VAL] Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        return metrics

    def _validate(self, epoch_idx=-1):
        outputs = []
        for i, batch in enumerate(self._val_dataloader):
            outputs.append(self.validation_step(batch, i))
        return self.validation_epoch_end()

    def _train(self, epoch_idx=-1):
        pbar = tqdm(total=len(self._train_dataloader))
        pbar.set_description(f"[TRAIN] Epoch {epoch_idx + 1}")
        for i, batch in enumerate(self._train_dataloader):
            loss_c = self.training_step_c_model(batch, i)
            pbar.update(1)
            pbar.set_postfix(ordered_dict={'loss_c_model': f"{loss_c.item():.3f}"})

    def fit(self):
        for i in tqdm(range(self._trainer_kwargs['max_epochs'])):
            self._train(i)
            self._validate(i)
        last_n = 100
        subset = self.metric_history[-last_n:]
        accs = np.array([m['acc'] for m in subset])
        f1s = np.array([m['f1'] for m in subset])
        aucs = np.array([m['auc'] for m in subset])
        logger.info(f"\n=== Statistics of the last {len(subset)} validation metrics ===")
        logger.info(f"Accuracy: ACC {accs.mean():.4f}, MAX {accs.max():.4f}")
        logger.info(f"F1-Score: ACC {f1s.mean():.4f}, MAX {f1s.max():.4f}")
        logger.info(f"AUC: ACC {aucs.mean():.4f}, MAX {aucs.max():.4f}")

    def save(self, param_path):
        torch.save({
            'c_model': self.c_model.state_dict(),
            'g_model': self.g_model.state_dict()
        }, param_path)

    def load(self):
        gan_model_state_dict = torch.load(
            './parameters/DE-3DGAN_3D_UNet_GAN_valence.pth')
        self.g_model.load_state_dict(gan_model_state_dict['g_model'])

        if os.path.exists('./parameters/DE-3DGAN_SA-ST_Net_Classifier_valence' + str(CFG.FOLD) + '.pth'):
            c_model_state_dict = torch.load(
                './parameters/DE-3DGAN_SA-ST_Net_Classifier_valence' + str(CFG.FOLD) +
                '.pth')
            self.c_model.load_state_dict(c_model_state_dict['c_model'])

if __name__=='__main__':
    ds = EEGDataset5D(preprocessors_results)
    ds = FrameDataset(ds)
    G  = Generator3D_UNet()
    C  = Classifier(num_classes=2, in_channels=128)
    trainer = Trainer(C, G, ds, trainer_kwargs={'max_epochs':CFG.NUM_EPOCHS})
    trainer.load()
    trainer.fit()
    trainer.save('./parameters/' + TRAIL_ID + '.pth')
