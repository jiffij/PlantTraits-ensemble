import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as imageio
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch
from transformers import SwinModel
import glob
import torchmetrics
import time
import psutil
import os
import logging
import random
import argparse

# logging = logging.getlogging(__name__)
if __name__ == "__main__":
    
    tqdm.pandas()

    # Configure logging
    logging.basicConfig(filename='output.txt', level=logging.INFO)

    # In[3]:

    parser = argparse.ArgumentParser(description='Resume training script.')

    # Add the --resume argument
    parser.add_argument('--resume', type=str, help='Path to the checkpoint file to resume training from')
    parser.add_argument('--model', type=str, default='MHA', help='MHA, Model, simpleModel, vitmha')
    parser.add_argument('--csv', type=str, default='train.csv', help='train csv path')
    parser.add_argument('--test_only', action='store_true')

    # Parse the arguments
    args = parser.parse_args()

    logging.info(f'GPU Name: {torch.cuda.get_device_name(0)}')

    EMB_SIZE = 512

    class Config():
        IMAGE_SIZE = 384
        BACKBONE = '../transformers' #'microsoft/swin-large-patch4-window12-384-in22k'
        TARGET_COLUMNS = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
        N_TARGETS = len(TARGET_COLUMNS)
        BATCH_SIZE = 10
        LR_MAX = 3e-3
        WEIGHT_DECAY = 0.01
        N_EPOCHS = 20
        TRAIN_MODEL = True
        IS_INTERACTIVE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive'
        COL_SIZE = 30
        VIT_OUTPUT_SIZE = 256
        DROPOUT = 0.3
        MODEL = "simpleModel"  # "simpleModel", "Model", MHA
        ONLY_MLP = False
        HIDDEN_SIZE = 100

    CONFIG = Config()
    CONFIG.MODEL = args.model

    class Dataset(Dataset):
        def __init__(self, X_jpeg_bytes,X_MLP, y, transforms=None):
            self.X_jpeg_bytes = X_jpeg_bytes
            self.X_MLP = X_MLP
            self.y = y
            self.transforms = transforms

        def __len__(self):
            return len(self.X_jpeg_bytes)

        def __getitem__(self, index):
            X_sample = self.transforms(
                image=imageio.imread(self.X_jpeg_bytes[index]),
            )['image']
            X_MLP_data = torch.tensor(self.X_MLP[index])

            y_sample = torch.tensor(self.y[index])

            return X_sample, X_MLP_data, y_sample


    def save_model(file, m, o, s, e):
        torch.save({
            'epoch': e,
            'model_state_dict': m.state_dict(),
            'optimizer_state_dict': o.state_dict(),
            'scheduler_state_dict': s.state_dict(),
        }, file)


    def load_model(file, model, optimizer, LR_SCHEDULER):
        # global 
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        LR_SCHEDULER.load_state_dict(checkpoint['scheduler_state_dict'])
        # model.vit.cuda()
        model.cuda()
        print(f"{file} loaded.")
        return start_epoch


    def load_swin_model(m, file):
        # global model, optimizer, LR_SCHEDULER, start_epoch
        checkpoint = torch.load(file)
        # start_epoch = checkpoint['epoch'] + 1
        m.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # LR_SCHEDULER.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"{file} loaded.")
        
        
    def load_mha_model(m, file):
        # global model, optimizer, LR_SCHEDULER, start_epoch
        checkpoint = torch.load(file)
        # start_epoch = checkpoint['epoch'] + 1
        m.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # LR_SCHEDULER.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"{file} loaded.")


    class Swin(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = SwinModel.from_pretrained(CONFIG.BACKBONE, local_files_only=True if CONFIG.BACKBONE != "microsoft/swin-large-patch4-window12-384-in22k" else False)
            self.fc1 = nn.Linear(1536, 512)
            self.fc2 = nn.Linear(512, CONFIG.N_TARGETS)

        def forward(self, inputs):
            output = self.backbone(inputs).pooler_output
            output = self.fc1(output)
            return self.fc2(output)


    class mha(nn.Module):
        def __init__(self, input_dim, num_heads, hidden_dim, dropout=0.1):
            super(mha, self).__init__()
            self.num_heads = num_heads
            self.hidden_dim = hidden_dim
            self.head_dim = hidden_dim // num_heads

            self.linear_q = nn.Linear(input_dim, hidden_dim)
            self.linear_k = nn.Linear(input_dim, hidden_dim)
            self.linear_v = nn.Linear(input_dim, hidden_dim)

            self.attention = nn.MultiheadAttention(1, num_heads, dropout=dropout, batch_first=True)
            # print(input_dim)
            self.linear1 = nn.Linear(hidden_dim, 36)
            self.linear_out = nn.Linear(36, 6)
            self.batchnorm = nn.BatchNorm1d(36)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
            q = self.linear_q(x).unsqueeze(-1)
            k = self.linear_k(x).unsqueeze(-1)
            v = self.linear_v(x).unsqueeze(-1)

            # q = q.view(q.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            # k = k.view(k.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            # v = v.view(v.size(0), -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            attention_output, _ = self.attention(q, k, v)

            attention_output = attention_output.squeeze()

            out = self.linear1(attention_output)
            if len(out.size()) == 1:
                out = out.unsqueeze(0)
            out = self.batchnorm(out)
            out = self.dropout(out)
            out = self.linear_out(out)
            return out


    class vitmha(nn.Module):
        def __init__(self):
            super().__init__()
            # if not CONFIG.ONLY_MLP:
            self.vit = Swin()
            load_swin_model(self.vit, "../checkpoint/latest_model.pth")
            # self.vit.fc2 = nn.Identity()  # remove the last layer
            # self.fc3 = nn.Linear(512, CONFIG.VIT_OUTPUT_SIZE)
            for param in self.vit.parameters():  # freeze the swin model
                param.requires_grad = False
            
            self.MHA = mha(30, 1, CONFIG.HIDDEN_SIZE)
            load_mha_model(self.MHA, "../checkpoint/MHA_model.pth")
            
            for param in self.MHA.parameters():  # freeze the swin model
                param.requires_grad = False
            
            self.vit = self.vit.cuda()
            self.MHA = self.MHA.cuda()
            self.out = nn.Linear(12, 6)

                
        def forward(self, image, metadata):
            vitout = self.vit(image)
            mhaout = self.MHA(metadata)
            # print(vitout.size(), mhaout.size())
            if len(vitout.size()) > 2: 
                vitout = vitout.squeeze()
            if len(mhaout.size()) > 2:
                mhaout = mhaout.squeeze()
            output = self.out(torch.cat((vitout, mhaout), dim=1))
            return output



    class simpleModel(nn.Module):
        def __init__(self, col_size):
            super().__init__()
            if not CONFIG.ONLY_MLP:
                self.vit = Swin()
                load_swin_model(self.vit, "../checkpoint/latest_model.pth")
                self.vit.fc2 = nn.Identity()  # remove the last layer
                self.fc3 = nn.Linear(512, CONFIG.VIT_OUTPUT_SIZE)
                for param in self.vit.parameters():  # freeze the swin model
                    param.requires_grad = False
            self.vit = self.vit.cuda()
            self.join = nn.Linear( (CONFIG.VIT_OUTPUT_SIZE if not CONFIG.ONLY_MLP else 0) + col_size, 150)
            self.ff1 = nn.Linear(150, 75)
            self.ff2 = nn.Linear(75, 25)
            self.ff3 = nn.Linear(25, 6)
            self.drop1 = nn.Dropout(CONFIG.DROPOUT)
            self.drop2 = nn.Dropout(CONFIG.DROPOUT)
            self.drop3 = nn.Dropout(CONFIG.DROPOUT)
            self.norm1 = nn.BatchNorm1d(150)
            self.norm2 = nn.BatchNorm1d(75)
            self.activation = nn.GELU()
                
        def forward(self, image, metadata):
            if not CONFIG.ONLY_MLP:
                vitout = self.fc3(self.vit(image))
                # print(vitout.size(), metadata.size())
                if len(vitout.size()) > 1: 
                    vitout = vitout.squeeze(0)
                    output = self.join(torch.cat((vitout, metadata), dim=0))
                else:
                    output = self.join(torch.cat((vitout, metadata), dim=1))
            else:
                output = self.join(metadata)
            output = self.norm1(output)
            output = self.activation(output)
            output = self.drop1(output)
            output = self.ff1(output)
            output = self.norm2(output)
            output = self.activation(output)
            output = self.drop2(output)
            output = self.ff2(output)
            output = self.activation(output)
            output = self.drop3(output)
            return self.ff3(output)
                

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = Swin()
            load_swin_model(self.vit, "../checkpoint/latest_model.pth")
            self.vit.fc2 = nn.Identity()  # remove the last layer
            self.fc3 = nn.Linear(512, 256)
            for param in self.vit.parameters():  # freeze the swin model
                param.requires_grad = False

            self.mlp = MLP(169,350,512,350,169)
            self.vitmlp = MLP(256,350,512,350,256)
            self.d_in = 169
            self.d_out_kq = 256
            self. d_out_v = 169
            self.d2_in = 256
            self.d2_out_kq = 169
            self.d2_out_v = 256
            self.mha1 = MultiHeadAttentionWrapper(
                self.d_in, self.d_out_kq, self.d_out_v, num_heads=4
            )
            self.mha2 = MultiHeadAttentionWrapper(
                self.d2_in, self.d2_out_kq, self.d2_out_v, num_heads=4
            )
            self.joining = JoiningTwoOutputnFlatten(self.d_out_v*4,self.d2_out_v*4, self.d_out_v,self.d2_out_v)
            self.fcc = FullyConnectedLayer(169+256, 6)
        def forward(self, image, metadata):
            vit = self.vit
            mlp = self.mlp
            fc3 = self.fc3
            vitmlp = self.vitmlp
            joining = self.joining
            fcc = self.fcc
            mha1 = self.mha1
            mha2 = self.mha2
            outvit = fc3(vit(image))
            outmlp = mlp(metadata)
            outvitmlp = vitmlp(outvit)
            outmha1 = mha1(outvitmlp, outmlp)
            outmha2 = mha2(outmlp, outvitmlp)
            out = joining(outmha1,outmha2, outmlp,outvitmlp)
            return fcc(out)

    # class ViT(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.backbone = timm.create_model(
    #                 Config.BACKBONE,
    #                 num_classes=256,
    #                 pretrained=True)
    #
    #     def forward(self, inputs):
    #         return self.backbone(inputs)

    class MLP(nn.Module):
        def __init__(self, inputdim, outputdim1,outputdim2, outputdim3, outputdim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(inputdim, outputdim1),
                nn.GELU(),
                nn.Linear(outputdim1, outputdim2),
                # nn.ReLU(),
                # nn.Linear(outputdim2,outputdim3),
                nn.GELU(),
                nn.Linear(outputdim2, outputdim)
            )
        def forward(self,input):
            return self.layers(input)
    class CrossAttention(nn.Module):
        def __init__(self, d_in, d_out_kq, d_out_v):
            super().__init__()
            self.d_out_kq = d_out_kq
            self.W_query = nn.Parameter(torch.rand(d_out_kq, d_in))
            self.W_key   = nn.Parameter(torch.rand(d_in, d_in))
            self.W_value = nn.Parameter(torch.rand(d_in, d_in))
        def forward(self, x_1, x_2):
            queries_1 = x_1 @ self.W_query

            keys_2 = x_2 @ self.W_key
            values_2 = x_2 @ self.W_value

            attn_scores = queries_1 @ keys_2.T
            attn_weights = torch.softmax(
                attn_scores / self.d_out_kq**0.5, dim=-1)

            context_vec = attn_weights @ values_2
            return context_vec

    class MultiHeadAttentionWrapper(nn.Module):

        def __init__(self, d_in, d_out_kq, d_out_v, num_heads):
            super().__init__()
            self.heads = nn.ModuleList(
                [CrossAttention(d_in, d_out_kq, d_out_v)
                for _ in range(num_heads)]
            )

        def forward(self, input1, input2):
            return torch.cat([head(input1,input2) for head in self.heads], dim=-1)


    class JoiningTwoOutputnFlatten(nn.Module):
        def __init__(self,d1nhead1dim, d2nhead2dim, qx1dim,qx2dim ):
            super().__init__()
            self.linear = nn.Linear(d1nhead1dim, qx1dim)
            self.linear2 = nn.Linear(d2nhead2dim, qx2dim)

        def forward(self,x1,x2,qx1,qx2):
            # x1 (1, d1*head)and qx1 (1,d1) , x2 (1, d2*head) and qx2 (1,d2)
            x1 = self.linear(x1)
            x2 = self.linear2(x2)
            # x1 and qx1 (1,d1) , x2 and qx2 (1,d2)
            x = x1+qx1
            y = x2+qx2
            out = torch.cat((x,y), dim=1) # output (1, (d1+d2) )

            return out
    class FullyConnectedLayer(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim,1024),
                nn.GELU(),
                nn.Linear(1024,512),
                nn.GELU(),
                nn.Linear(512,256),
                nn.GELU(),
                nn.Linear(256,output_dim)
            )
        def forward(self,x):
            return self.layers(x)
    class AverageMeter(object):
        def __init__(self):
            self.reset()

        def reset(self):
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val):
            self.sum += val.sum()
            self.count += val.numel()
            self.avg = self.sum / self.count


    def get_lr_scheduler(optimizer, CONFIG):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=CONFIG.N_STEPS,
            eta_min=1e-7
        )
        # return torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=optimizer,
        #     max_lr=CONFIG.LR_MAX,
        #     total_steps=CONFIG.N_STEPS,
        #     pct_start=0.1,
        #     anneal_strategy='cos',
        #     div_factor=1e1,
        #     final_div_factor=1e1,
        # )
    def r2_loss(y_pred, y_true, Y_MEAN, EPS):
        ss_res = torch.sum((y_true - y_pred)**2, dim=0)
        ss_total = torch.sum((y_true - Y_MEAN)**2, dim=0)
        ss_total = torch.maximum(ss_total, EPS)
        r2 = torch.mean(ss_res / ss_total)
        return r2

    def random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def main():
        random_seed(100)
        logging.warning("started")

        train = pd.read_csv(f'../planttraits2024/{args.csv}')

        logging.warning("Reading data")
        train['file_path'] = train['id'].apply(lambda s: f'../planttraits2024/train_images/{s}.jpeg')
        train['jpeg_bytes'] = train['file_path'].progress_apply(lambda fp: open(fp, 'rb').read())
        train.to_pickle('./train.pkl')
        logging.warning("train data gaodim")
        for column in CONFIG.TARGET_COLUMNS:
            lower_quantile = train[column].quantile(0.005)
            upper_quantile = train[column].quantile(0.985)
            train = train[(train[column] >= lower_quantile) & (train[column] <= upper_quantile)]
        logging.warning("preprocess train data")
        CONFIG.N_TRAIN_SAMPLES = len(train)
        CONFIG.N_STEPS_PER_EPOCH = (CONFIG.N_TRAIN_SAMPLES // CONFIG.BATCH_SIZE)
        CONFIG.N_STEPS = CONFIG.N_STEPS_PER_EPOCH * CONFIG.N_EPOCHS + 1
        logging.warning("config gaodim")
        logging.warning("Reading test data")

        test = pd.read_csv('../planttraits2024/test_outlier_preprocessed.csv')
        test['file_path'] = test['id'].apply(lambda s: f'../planttraits2024/test_images/{s}.jpeg')
        test['jpeg_bytes'] = test['file_path'].progress_apply(lambda fp: open(fp, 'rb').read())
        test.to_pickle('./test.pkl')
        logging.warning(" test data gaodim")
        logging.info(f'N_TRAIN_SAMPLES: {len(train)}, N_TEST_SAMPLES: {len(test)}')
        LOG_FEATURES = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
        logging.warning(" transforming y_train")
        y_train = np.zeros_like(train[CONFIG.TARGET_COLUMNS], dtype=np.float32)
        for target_idx, target in enumerate(CONFIG.TARGET_COLUMNS):
            v = train[target].values
            if target in LOG_FEATURES:
                v = np.log10(v)
            y_train[:, target_idx] = v

        SCALER = StandardScaler()
        y_train = SCALER.fit_transform(y_train)
        logging.warning(" y_train gaodim")
        logging.warning(" x_mlp_train")
        DROP_COLUMNS = ['id','X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean','file_path','jpeg_bytes', 'X4_sd','X11_sd','X18_sd','X26_sd','X50_sd','X3112_sd']
        columns_to_keep = train.columns[~train.columns.isin(DROP_COLUMNS)]
        x_MLP_train = train[columns_to_keep]
        x_MLP_train = x_MLP_train.astype('float32')
        
        

        mlp_scaler_train = StandardScaler()
        x_MLP_scaled_train = mlp_scaler_train.fit_transform(x_MLP_train)
        logging.info(x_MLP_scaled_train.shape)
        logging.warning(" x_mlp_test")
        DROP_COLUMNS = ['id','X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean','file_path','jpeg_bytes', 'X4_sd','X11_sd','X18_sd','X26_sd','X50_sd','X3112_sd']
        columns_to_keep = test.columns[~test.columns.isin(DROP_COLUMNS)]
        x_MLP_test = test[columns_to_keep]
        x_MLP_test = x_MLP_test.astype('float32')

        mlp_scaler_test = StandardScaler()
        x_MLP_scaled_test = mlp_scaler_test.fit_transform(x_MLP_test)
        logging.info(x_MLP_scaled_test.shape)
        logging.warning(" all gaodim")
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        

        y_train = np.nan_to_num(y_train, nan=0)
        x_MLP_scaled_train = np.nan_to_num(x_MLP_scaled_train, nan=0)
        x_MLP_scaled_test = np.nan_to_num(x_MLP_scaled_test, nan=0)
        
        CONFIG.COL_SIZE = x_MLP_train.shape[1]

        TRAIN_TRANSFORMS = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomSizedCrop(
                    [448, 512],
                    CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE, w2h_ratio=1.0, p=0.75),
                A.Resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
                A.ImageCompression(quality_lower=85, quality_upper=100, p=0.25),
                A.ToFloat(),
                A.Normalize(mean=MEAN, std=STD, max_pixel_value=1),
                ToTensorV2(),
            ])

        TEST_TRANSFORMS = A.Compose([
                A.Resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE),
                A.ToFloat(),
                A.Normalize(mean=MEAN, std=STD, max_pixel_value=1),
                ToTensorV2(),
            ])
        train_dataset = Dataset(
        train['jpeg_bytes'].values,
        x_MLP_scaled_train,
        y_train,
        TRAIN_TRANSFORMS,
        )

        train_dataloader = DataLoader(
                train_dataset,
                batch_size=CONFIG.BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                num_workers=2,
        )

        test_dataset = Dataset(
            test['jpeg_bytes'].values,
            x_MLP_scaled_test,
            test['id'].values,
            TEST_TRANSFORMS,
        )
        
        logging.warning("data gaodim")
        
        # print(x_MLP_scaled_train.shape)
        # print(CONFIG.COL_SIZE)
        
        if CONFIG.MODEL == "simpleModel":
            model = simpleModel(CONFIG.COL_SIZE)
        elif CONFIG.MODEL == "Model":
            model = Model()
        elif CONFIG.MODEL == "MHA":
            model = mha(CONFIG.COL_SIZE, 1, CONFIG.HIDDEN_SIZE)
        elif CONFIG.MODEL == "vitmha":
            model = vitmha()
        else:
            print("Wrong model name.")
            
        model = model.cuda()
        
        total_params = sum(p.numel() for p in model.parameters())
        total_memory_for_weights = total_params * 4
        logging.warning("Total memory needed: "+ str(total_memory_for_weights))

        logging.info(model)

        MAE = torchmetrics.regression.MeanAbsoluteError().cuda()
        R2 = torchmetrics.regression.R2Score(num_outputs=CONFIG.N_TARGETS, multioutput='uniform_average').to('cuda')
        LOSS = AverageMeter()

        Y_MEAN = torch.tensor(y_train).mean(dim=0).cuda()
        EPS = torch.tensor([1e-6]).cuda()
        LOSS_FN = nn.SmoothL1Loss() # r2_loss

        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=CONFIG.LR_MAX,
            weight_decay=CONFIG.WEIGHT_DECAY,
        )

        LR_SCHEDULER = get_lr_scheduler(optimizer, CONFIG)
        
        start_epoch = 1
        if args.resume:
            start_epoch = load_model(args.resume, model, optimizer, LR_SCHEDULER)


        logging.warning("Training")
        # new
        print("Start Training:")
        logging.info("Start Training:")
        del train_dataset
        best_loss = 1000000000
        
        if not args.test_only:
            for epoch in range(start_epoch, CONFIG.N_EPOCHS+1):
                MAE.reset()
                R2.reset()
                LOSS.reset()
                model.train()
                logging.info(f"Starting Epoch {epoch}...")
                # Assuming optimizer is your optimizer (e.g., SGD, Adam)
                max_grad_norm = 1.0  # Set the maximum gradient norm

                for step, (X_batch, MLP_batch, y_true) in enumerate(train_dataloader):
                    if CONFIG.MODEL != "MHA":
                        X_batch = X_batch.cuda()
                    # mlp_batch = MLP_batch[:, :30]
                    mlp_batch = MLP_batch.cuda()
                    y_true = y_true.cuda()
                    t_start = time.perf_counter_ns()
                    if CONFIG.MODEL != "MHA":
                        y_pred = model(X_batch, mlp_batch)
                    else:
                        y_pred = model(mlp_batch)
                    loss = LOSS_FN(y_pred, y_true)
                    LOSS.update(loss)
                    loss.backward()
                    
                    # Calculate the gradient norm
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    LR_SCHEDULER.step()
                    MAE.update(y_pred, y_true)
                    R2.update(y_pred, y_true)

                    if step%100 == 0:
                        # logging.info(MLP_batch[0], y_true[0])
                        logging.info(f'EPOCH {epoch:02d}, {step + 1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                                    f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                                    f'step: {(time.perf_counter_ns() - t_start) * 1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}')

                        print(
                                f'EPOCH {epoch+1:02d}, {step+1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                                f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                            f'step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}')
                    # elif CONFIG.IS_INTERACTIVE:
                    #     print(
                    #         f'\rEPOCH {epoch+1:02d}, {step+1:04d}/{CONFIG.N_STEPS_PER_EPOCH} | ' +
                    #         f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                    #         f'step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s, lr: {LR_SCHEDULER.get_last_lr()[0]:.2e}',
                    #         end='\n' if (step + 1) == CONFIG.N_STEPS_PER_EPOCH else '', flush=True,
                    #     )

                if LOSS.avg < best_loss:
                    best_loss = LOSS.avg
                    save_model(f"{args.model}_model.pth", model, optimizer, LR_SCHEDULER, epoch)

                save_model(f"{args.model}_latest_model.pth", model, optimizer, LR_SCHEDULER, epoch)

        SUBMISSION_ROWS = []
        model.eval()

        for X_sample_test, mlp_sample_test, test_id in tqdm(test_dataset):
            # if mlp_sample_test.size(0) < 36:
            #     temp = torch.zeros(1, 36)
            #     temp[0, :mlp_sample_test.size(0)] = mlp_sample_test
            #     mlp_sample_test = temp
            with torch.no_grad():
                if args.model == "MHA":
                    y_pred = model(mlp_sample_test.cuda()).detach().cpu().numpy()
                else:
                    y_pred = model(X_sample_test.unsqueeze(0).cuda(), mlp_sample_test.cuda()).detach().cpu().numpy()

            y_pred = SCALER.inverse_transform(y_pred).squeeze()
            row = {'id': test_id}

            for k, v in zip(CONFIG.TARGET_COLUMNS, y_pred):
                if k in LOG_FEATURES:
                    row[k.replace('_mean', '')] = 10 ** v
                else:
                    row[k.replace('_mean', '')] = v

            SUBMISSION_ROWS.append(row)

        submission_df = pd.DataFrame(SUBMISSION_ROWS)
        submission_df.to_csv('submission.csv', index=False)
        print("Submit!")
        # output = joining(mha1(input)) #try this

        # mlp = mlp.to('cuda')
        # print(mlp)
        # print(model)
        # print(mha1)
        # print(mha2)
        # print(joining)
        # print(fcc)

    main()








