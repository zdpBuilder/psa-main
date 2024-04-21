import torch
import numpy as np
from torch import nn
import os
import torchaudio
import importlib
import torchaudio.compliance.kaldi as Kaldi
from PSA.model.ERes2Net import ERes2Net
from sklearn.feature_selection import mutual_info_regression
class FBank(object):
    def __init__(self,
        n_mels,
        sample_rate,
        mean_nor: bool = False,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mean_nor = mean_nor

    def __call__(self, wav, dither=0):
        sr = 16000
        assert sr==self.sample_rate
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # select single channel
        if wav.shape[0] > 1:
            wav = wav[0, :]
        assert len(wav.shape) == 2 and wav.shape[0]==1
        feat = Kaldi.fbank(wav, num_mel_bins=self.n_mels,
            sample_frequency=sr, dither=dither)
        # feat: [T, N]
        if self.mean_nor:
            feat = feat - feat.mean(0, keepdim=True)
        return feat

class eres2net2PSA(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.pretrained_state = torch.load(path, map_location='cpu')
        self.model = ERes2Net(**{'feat_dim': 80,'embedding_size': 512,'m_channels': 64})
        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

        if torch.cuda.is_available():
            msg = 'Using gpu for inference.'
            print(f'[INFO]: {msg}')
            self.device = torch.device('cuda')
        else:
            msg = 'No cuda device is detected. Using cpu.'
            print(f'[INFO]: {msg}')
            self.device = torch.device('cpu')
        self.model.load_state_dict(self.pretrained_state)
        self.model.to(self.device)
        self.model.eval()


    def compute_embedding(self,wav_file, embedding_dir,save=True):
        # load wav
        wav = self.load_wav(wav_file)
        # compute feat
        feat = self.feature_extractor(wav).unsqueeze(0).to(self.device)
        # compute embedding
        with torch.no_grad():
            embedding = self.model(feat).detach().cpu().numpy()

        if save:
            save_path = embedding_dir / (
                    '%s.npy' % (os.path.basename(wav_file).rsplit('.', 1)[0]))
            np.save(save_path, embedding)
            print(f'[INFO]: The extracted embedding from {wav_file} is saved to {save_path}.')

        return embedding

    def load_wav(self,wav_file, obj_fs=16000):
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
            wav, fs = torchaudio.sox_effects.apply_effects_tensor(
                wav, fs, effects=[['rate', str(obj_fs)]]
            )
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        return wav

    def dynamic_import(self,import_path):
        module_name, obj_name = import_path.rsplit('.', 1)
        m = importlib.import_module(module_name)
        return getattr(m, obj_name)

    def forward(self, wav_file, embedding_dir, save=True):
        # extract embeddings
        return self.compute_embedding(wav_file, embedding_dir, save)

    def get_one_wav_embedding(self, wav_file,embedding_dir="",save=False):
        with torch.no_grad():
            embedding = self.forward(wav_file,embedding_dir,save)
        return embedding

    def get_similarity_score_by_cosine(self,source_wav_path,target_wav_path):
        source_embedding = self.get_one_wav_embedding(source_wav_path)
        target_embedding = self.get_one_wav_embedding(target_wav_path)
        similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        return similarity(torch.from_numpy(source_embedding), torch.from_numpy(target_embedding)).item()[0]

    def get_similarity_result_by_MI(self,source_wav_path,target_wav_path):
        source_embedding = self.get_one_wav_embedding(source_wav_path)
        target_embedding = self.get_one_wav_embedding(target_wav_path)
        return mutual_info_regression(source_embedding.flatten().reshape(-1, 1),target_embedding.flatten())
    def get_PSA_Score(self,source_wav_path,target_wav_path):
        result_score = self.get_similarity_result_by_MI(source_wav_path,target_wav_path)
        if not result_score == 0:
            result_score =  self.get_similarity_score_by_cosine(source_wav_path,target_wav_path)
        return result_score