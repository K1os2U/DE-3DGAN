import os
import numpy as np
from scipy.io import loadmat
from collections import Counter
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn import model_selection
from tqdm import tqdm
from pathlib import Path
from preprocessors import DEAPDataset, Sequence
from preprocessors import BinaryLabel
from preprocessors import Raw2TNCF, RemoveBaseline, TNCF2NCF, ChannelToLocation

class CFG:
    EPOCHS    = 300
    BATCH_SIZE= 8
    TEST_SIZE = 0.2
    G_LR      = 1e-4
    D_LR      = 1e-5
    WD        = 5e-4
    W_GP      = 1.0
    REC_W     = 0.5

TRAIL_ID = 'DE-3DGAN_3D_UNet_GAN_valence'
logger = logging.getLogger(TRAIL_ID)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('logs/{}.log'.format(TRAIL_ID))

# EEG signal preprocessing
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
    return ((x - mu) / sigma)

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
        # —— Encoder ——
        self.enc1 = ResidualConv3d(in_ch,    base_ch)
        self.enc2 = ResidualConv3d(base_ch,  base_ch*2)
        self.enc3 = ResidualConv3d(base_ch*2,base_ch*4)
        self.enc4 = ResidualConv3d(base_ch*4,base_ch*8)
        self.pool = nn.MaxPool3d((2,2,2),(2,2,2))
        # —— Decoder ——
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
        mask = (x.abs().sum(dim=(1,2), keepdim=True) > 0).float()  # x: (B,  4, 120, 9, 9)
        e1 = self.enc1(x)     
        p1 = self.pool(e1)    
        e2 = self.enc2(p1)   
        p2 = self.pool(e2)   
        e3 = self.enc3(p2)   
        p3 = self.pool(e3)   
        e4 = self.enc4(p3)    
        d3 = self.up3(e4)    
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)   
        d2 = self.up2(d3)     
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)    
        d1 = self.up1(d2)     
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)   
        out = self.out(d1)  

        out = F.interpolate(
            out,
            size=(self.time_len, 9, 9),
            mode='trilinear',
            align_corners=False
        )                     
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out * self.alpha
        return out * mask

# SE-Block module
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          
            nn.Conv2d(channels, channels//reduction, 1),
            nn.SELU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)  
        return x * w

# Spatial attention module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2,1,kernel_size,padding=kernel_size//2,bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        maxc,_ = x.max(dim=1,keepdim=True)  
        avgc   = x.mean(dim=1,keepdim=True) 
        att    = self.sig(self.conv(torch.cat([maxc,avgc],dim=1)))
        return x * att

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

# discriminator structure
class Discriminator(nn.Module):
    def __init__(self, num_classes, in_channels=4):
        super(Discriminator, self).__init__()
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
        self.drop = nn.Sequential(nn.SELU())
        self.fc1 = nn.Sequential(nn.Linear(9 * 9 * 16, 1024, bias=True),
                                 nn.SELU())
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
        out = self.fc1(out)
        out = self.fc2(out)
        return out

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

def train_test_split5D(ds, test_size=0.2, seed=520):
    idx = np.arange(len(ds))
    tr, te = model_selection.train_test_split(idx,
                test_size=test_size, random_state=seed, shuffle=True)
    return Subset(ds, tr), Subset(ds, te)

# GAN framework training process
class Trainer5D:
    def __init__(self, G, D, ds, cfg):
        self.G, self.D = G.cuda(), D.cuda()
        tr, ts = train_test_split5D(ds, test_size=cfg.TEST_SIZE)
        self.loader = DataLoader(tr, batch_size=cfg.BATCH_SIZE,
                                 shuffle=True, drop_last=True)
        self.ts_loader = DataLoader(ts, batch_size=cfg.BATCH_SIZE,
                                 shuffle=True, drop_last=True)
        self.optG = torch.optim.Adam(G.parameters(), lr=cfg.G_LR, weight_decay=cfg.WD)
        self.optD = torch.optim.Adam(D.parameters(), lr=cfg.D_LR, weight_decay=cfg.WD)
        self.mse  = nn.MSELoss()
        self.cfg  = cfg

    def train_epoch(self, epoch):
        self.G.train(); self.D.train()
        lg = ld = 0.0
        for de, raw, label in tqdm(self.loader, desc=f"Epoch {epoch}"):
            de, raw = de.cuda(), raw.cuda() 
            de = robust_norm(de)
            raw = robust_norm(raw)

            # Generator step
            self.optG.zero_grad()
            gen = self.G(de)  # gen: (B,60,128,9,9)
            gen_flat = gen.view(-1, 128, 9, 9)
            raw_flat = raw.view(-1, 128, 9, 9)
            loss_adv = -self.D(gen_flat).mean()
            loss_rec = self.mse(gen, raw)

            loss_g = loss_adv + self.cfg.REC_W * loss_rec
            loss_g.backward()
            self.optG.step()
            lg += loss_g.item()

            self.optD.zero_grad()
            d_fake = self.D(gen_flat.detach()).mean()
            d_real = self.D(raw_flat).mean()
            gp = self.cfg.W_GP * gradient_penalty(self.D, raw_flat, gen_flat.detach())
            loss_d = d_fake - d_real + gp
            if epoch % 5 == 0:
                loss_d.backward()
                self.optD.step()
            ld += loss_d.item()
        print(f"[{epoch}] G Loss: {lg/len(self.loader):.4f} | D Loss: {ld/len(self.loader):.4f}")

    def evaluate(self):
        self.G.eval()
        total_mse = 0.0
        total_n = 0
        with torch.no_grad():
            for de_raw, raw_raw,label in self.ts_loader:
                de = robust_norm(de_raw.cuda())
                raw = robust_norm(raw_raw.cuda())

                gen = self.G(de)
                loss = self.mse(gen, raw) * de.size(0)
                total_mse += loss
                total_n += de.size(0)

        avg_mse = total_mse / total_n
        print(f"⇒ [Eval] Avg Reconstruction MSE: {avg_mse:.4f}")
        return avg_mse

    def fit(self):
        print("Starting GAN training with 2D Discriminator...")
        for e in range(1, self.cfg.EPOCHS + 1):
            self.train_epoch(e)
            if e % 20 == 0 or e == self.cfg.EPOCHS:
                self.evaluate()

    def save(self, param_path):
        torch.save(
            {
                'g_model': self.G.state_dict(),
                'd_model': self.D.state_dict()
            }, param_path)


if __name__ == '__main__':
    ds = EEGDataset5D(preprocessors_results,de_dir='./DE_feature/',)
    G = Generator3D_UNet(in_ch=4)
    D = Discriminator(num_classes=1, in_channels=128)
    trainer = Trainer5D(G, D, ds, CFG)
    trainer.fit()
    trainer.save('./parameters/' + TRAIL_ID + '.pth')
