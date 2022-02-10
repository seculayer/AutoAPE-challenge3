import sys
from pathlib import Path
from datetime import timedelta

import dateutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from TaPR_pkg import etapr


def dataframe_from_csv(target):
	return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
	return pd.concat([dataframe_from_csv(x) for x in targets])


def extract_feature(DF_RAW, RAW_INDEX, nts):
	temp_train = []
	temp_train_2 = []
	for i in range(len(nts)):
		temp_line = []
		for j in range(len(nts[i])):
			if i == 0: temp_line.append(0)
			else: temp_line.append(nts[i, j] - nts[i-1, j])
		temp_train.append(temp_line)
	for i in range(len(nts)):
		temp_line = []
		for j in range(len(nts[i])):
			if i == 0: temp_line.append(0)
			else:
				if nts[i, j] >= nts[i-1, j]: temp_line.append(1)
				elif nts[i, j] < nts[i-1, j]: temp_line.append(-1)
		temp_train_2.append(temp_line)

	ext_list_1 = []
	ext_list_2 = []
	for i in range(len(temp_train[1])):
		col_n = 'D' + str(i+1)
		ext_list_1.append(col_n)
	for i in range(len(temp_train_2[1])):
		col_n = 'E' + str(i+1)
		ext_list_2.append(col_n)

	train_d_df = pd.DataFrame(temp_train, index=RAW_INDEX, columns=ext_list_1)
	train_e_df = pd.DataFrame(temp_train_2, index=RAW_INDEX, columns=ext_list_2)
	FIN_DF_RAW = pd.concat([DF_RAW, train_d_df, train_e_df], axis=1)
	return FIN_DF_RAW


def normalize(df):
	ndf = df.copy()
	for c in df.columns:
		if TAG_MIN[c] == TAG_MAX[c]:
			ndf[c] = df[c] - TAG_MIN[c]
		else:
			ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
	return ndf


TEST_DATASET = sorted([x for x in Path("./test/").glob("*.csv")])

TRAIN_DATASET = sorted([x for x in Path("./train/").glob("*.csv")])
VALIDATION_DATASET = sorted([x for x in Path("./validation/").glob("*.csv")])
TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)
VALIDATION_DF_RAW = dataframe_from_csvs(VALIDATION_DATASET)
VALIDATION_DF_RAW = VALIDATION_DF_RAW.loc[:, ['timestamp', 'C03', 'C12', 'C71', 'C73', 'C75', 'C76', 'attack']]
TRAIN_DF_RAW = TRAIN_DF_RAW.loc[:, ['timestamp', 'C03', 'C12', 'C71', 'C73', 'C75', 'C76']]
# ##############################################################################################################
# TRAIN_DATASET = sorted([x for x in Path("./train/").glob("*.csv")])
# TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)			# 전처리 추가 상승 하강 양상, 상승폭 하강폭 양상
# TRAIN_RAW_INDEX = TRAIN_DF_RAW.index
# train_nts = TRAIN_DF_RAW.iloc[:, 1:]
# train_nts = train_nts.values
# TRAIN_DF_RAW = extract_feature(TRAIN_DF_RAW, TRAIN_RAW_INDEX, train_nts)
# print(TRAIN_DF_RAW)
#
# VALIDATION_DATASET = sorted([x for x in Path("./validation/").glob("*.csv")])
# VALIDATION_RAW = dataframe_from_csvs(VALIDATION_DATASET)
# VALID_RAW_INDEX = VALIDATION_RAW.index
# valid_nts_df = VALIDATION_RAW.iloc[:, 1:-1]
# valid_nts = valid_nts_df.values
# VALIDATION_DF_RAW = extract_feature(VALIDATION_RAW.iloc[:, :-1], VALID_RAW_INDEX, valid_nts)
# VALIDATION_DF_RAW = pd.concat([VALIDATION_DF_RAW, VALIDATION_RAW.iloc[:, -1]], axis=1)
# print(VALIDATION_DF_RAW)
#
# corr = VALIDATION_DF_RAW.corr()
# corr_label = corr['attack']
# corr_label = corr_label.dropna(axis=0, how='any')
# print(corr_label.sort_values(ascending=False))
#
# corrl_1_1 = corr_label <= 0.1
# corrl_2_2 = corrl_1_1 >= -0.1
#
# #df_corr1_1_attack = corr_label[corrl_1_1]
# df_corr2_2_attack = corr_label[corrl_2_2]
#
# #corr1_1_index_list = list([ item for item in df_corr1_1_attack.index])
# corr2_2_index_list = list([ item for item in df_corr2_2_attack.index])
# corr_index_list = sorted(set(corr2_2_index_list)) #corr1_1_index_list +
# print(corr_index_list)
#
# VALIDATION_DF_RAW = VALIDATION_DF_RAW.loc[:, ['timestamp'] + corr_index_list]
# TRAIN_DF_RAW = TRAIN_DF_RAW.loc[:, ['timestamp'] + corr_index_list[:-1]]
# print(VALIDATION_DF_RAW)
# print(TRAIN_DF_RAW)
# ############################################################################################################

TIMESTAMP_FIELD = "timestamp"
IDSTAMP_FIELD = 'id'
ATTACK_FIELD = "attack"
VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop([TIMESTAMP_FIELD])
print(VALID_COLUMNS_IN_TRAIN_DATASET)

TAG_MIN = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].min()
TAG_MAX = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].max()

TRAIN_DF = normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
print(TRAIN_DF)


def boundary_check(df):
	x = np.array(df, dtype=np.float32)
	return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))

boundary_check(TRAIN_DF)


WINDOW_GIVEN = 89
WINDOW_SIZE = 90


class HaiDataset(Dataset):
	def __init__(self, timestamps, df, stride=1, attacks=None):
		self.ts = np.array(timestamps)
		self.tag_values = np.array(df, dtype=np.float32)
		self.valid_idxs = []
		for L in trange(len(self.ts) - WINDOW_SIZE + 1):
			R = L + WINDOW_SIZE - 1
			if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
					self.ts[L]
			) == timedelta(seconds=WINDOW_SIZE - 1):
				self.valid_idxs.append(L)
		self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
		self.n_idxs = len(self.valid_idxs)
		print(f"# of valid windows: {self.n_idxs}")
		if attacks is not None:
			self.attacks = np.array(attacks, dtype=np.float32)
			self.with_attack = True
		else:
			self.with_attack = False

	def __len__(self):
		return self.n_idxs

	def __getitem__(self, idx):
		i = self.valid_idxs[idx]
		last = i + WINDOW_SIZE - 1
		item = {"attack": self.attacks[last]} if self.with_attack else {}
		item["ts"] = self.ts[i + WINDOW_SIZE - 1]
		item["given"] = torch.from_numpy(self.tag_values[i : i + WINDOW_GIVEN])
		item["answer"] = torch.from_numpy(self.tag_values[last])
		return item


HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, stride=5)		# stride
print(HAI_DATASET_TRAIN[0])


N_HIDDENS = 100
N_LAYERS = 3
BATCH_SIZE = 512
class StackedGRU(torch.nn.Module):
	def __init__(self, n_tags):
		super().__init__()
		self.rnn = torch.nn.GRU(
			input_size=n_tags,
			hidden_size=N_HIDDENS,
			num_layers=N_LAYERS,
			bidirectional=True,
			dropout=0,
		)
		self.fc = torch.nn.Linear(N_HIDDENS * 2, n_tags)

	def forward(self, x):
		x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
		self.rnn.flatten_parameters()
		outs, _ = self.rnn(x)
		out = self.fc(outs[-1])
		return x[0] + out

MODEL = StackedGRU(n_tags=TRAIN_DF.shape[1])
MODEL.cuda()


def train(dataset, model, batch_size, n_epochs):
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	optimizer = torch.optim.AdamW(model.parameters())
	loss_fn = torch.nn.MSELoss()
	epochs = trange(n_epochs, desc="training")
	best = {"loss": sys.float_info.max}
	loss_history = []
	for e in epochs:
		epoch_loss = 0
		for batch in dataloader:
			optimizer.zero_grad()
			given = batch["given"].cuda()
			guess = model(given)
			answer = batch["answer"].cuda()
			loss = loss_fn(answer, guess)
			loss.backward()
			epoch_loss += loss.item()
			optimizer.step()
		loss_history.append(epoch_loss)
		epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
		if epoch_loss < best["loss"]:
			best["state"] = model.state_dict()
			best["loss"] = epoch_loss
			best["epoch"] = e + 1
	return best, loss_history


#%%time
MODEL.train()
BEST_MODEL, LOSS_HISTORY = train(HAI_DATASET_TRAIN, MODEL, BATCH_SIZE, 15)		# 30
print(BEST_MODEL["loss"])
print(BEST_MODEL["epoch"])

with open("model_2.pt", "wb") as f:
	torch.save(
		{
			"state": BEST_MODEL["state"],
			"best_epoch": BEST_MODEL["epoch"],
			"loss_history": LOSS_HISTORY,
		},
		f,
	)


with open("model_2.pt", "rb") as f:
	SAVED_MODEL = torch.load(f)

MODEL.load_state_dict(SAVED_MODEL["state"])


plt.figure(figsize=(16, 4))
plt.title("Training Loss Graph")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.yscale("log")
plt.plot(SAVED_MODEL["loss_history"])
plt.show()



VALIDATION_DF = normalize(VALIDATION_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET])

boundary_check(VALIDATION_DF)

HAI_DATASET_VALIDATION = HaiDataset(
	VALIDATION_DF_RAW[TIMESTAMP_FIELD], VALIDATION_DF, attacks=VALIDATION_DF_RAW[ATTACK_FIELD]
)
print(HAI_DATASET_VALIDATION[0])


def inference(dataset, model, batch_size):
	dataloader = DataLoader(dataset, batch_size=batch_size)
	ts, dist, att = [], [], []
	with torch.no_grad():
		for batch in dataloader:
			given = batch["given"].cuda()
			answer = batch["answer"].cuda()
			guess = model(given)
			ts.append(np.array(batch["ts"]))
			dist.append(torch.abs(answer - guess).cpu().numpy())
			try:
				att.append(np.array(batch["attack"]))
			except:
				att.append(np.zeros(batch_size))

	return (
		np.concatenate(ts),
		np.concatenate(dist),
		np.concatenate(att),
	)


#%%time
MODEL.eval()
CHECK_TS, CHECK_DIST, CHECK_ATT = inference(HAI_DATASET_VALIDATION, MODEL, BATCH_SIZE)
print(CHECK_DIST.shape)

ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)					# 평균


def check_graph(xs, att, piece=2, THRESHOLD=None):
	l = xs.shape[0]
	chunk = l // piece
	fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
	for i in range(piece):
		L = i * chunk
		R = min(L + chunk, l)
		xticks = range(L, R)
		axs[i].plot(xticks, xs[L:R])
		if len(xs[L:R]) > 0:
			peak = max(xs[L:R])
			axs[i].plot(xticks, att[L:R] * peak * 0.3)
		if THRESHOLD!=None:
			axs[i].axhline(y=THRESHOLD, color='r')
	plt.show()

THRESHOLD03 = 0.5
THRESHOLD12 = 0.053
THRESHOLD71 = 0.18
THRESHOLD73 = 0.1
THRESHOLD75 = 0.025
THRESHOLD76 = 0.07

check_graph(ANOMALY_SCORE, CHECK_ATT, piece=2, THRESHOLD=0.05)


# def three_test(CHECK_TS, VALIDATION_DF_RAW, ATTACK_FIELD, TIMESTAMP_FIELD, ANOMALY_SCORE, THRESHOLD):
# 	def put_labels(distance, threshold):
# 		xs = np.zeros_like(distance)
# 		xs[distance > threshold] = 1
# 		return xs

# 	LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
# 	print(LABELS)
# 	print(LABELS.shape)

# 	ATTACK_LABELS = put_labels(np.array(VALIDATION_DF_RAW[ATTACK_FIELD]), threshold=0.5)
# 	print(ATTACK_LABELS)
# 	print(ATTACK_LABELS.shape)


# 	def fill_blank(check_ts, labels, total_ts):
# 		def ts_generator():
# 			for t in total_ts:
# 				yield dateutil.parser.parse(t)

# 		def label_generator():
# 			for t, label in zip(check_ts, labels):
# 				yield dateutil.parser.parse(t), label

# 		g_ts = ts_generator()
# 		g_label = label_generator()
# 		final_labels = []

# 		try:
# 			current = next(g_ts)
# 			ts_label, label = next(g_label)
# 			while True:
# 				if current > ts_label:
# 					ts_label, label = next(g_label)
# 					continue
# 				elif current < ts_label:
# 					final_labels.append(0)
# 					current = next(g_ts)
# 					continue
# 				final_labels.append(label)
# 				current = next(g_ts)
# 				ts_label, label = next(g_label)
# 		except StopIteration:
# 			return np.array(final_labels, dtype=np.int8)


# 	#%%time
# 	FINAL_LABELS = fill_blank(CHECK_TS, LABELS, np.array(VALIDATION_DF_RAW[TIMESTAMP_FIELD]))
# 	print(FINAL_LABELS.shape)


# 	TaPR = etapr.evaluate_haicon(anomalies=ATTACK_LABELS, predictions=FINAL_LABELS)
# 	print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
# 	print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
# 	print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")

# # 평균
# three_test(CHECK_TS, VALIDATION_DF_RAW, ATTACK_FIELD, TIMESTAMP_FIELD, ANOMALY_SCORE, THRESHOLD1)

# TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)
# print(TEST_DF_RAW)

# TEST_DF = normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
# print(TEST_DF)
# print(boundary_check(TEST_DF))

# HAI_DATASET_TEST = HaiDataset(TEST_DF_RAW[TIMESTAMP_FIELD], TEST_DF, attacks=None)
# print(HAI_DATASET_VALIDATION[0])

#%%time
# MODEL.eval()
# CHECK_TS, CHECK_DIST, CHECK_ATT = inference(HAI_DATASET_TEST, MODEL, BATCH_SIZE)

# ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)
# check_graph(ANOMALY_SCORE, CHECK_ATT, piece=3, THRESHOLD=THRESHOLD)

# LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
# print(LABELS)
# print(LABELS.shape)

# submission = pd.read_csv('./sample_submission.csv')
# submission.index = submission['timestamp']
# submission.loc[CHECK_TS,'attack'] = LABELS
# print(submission)

# submission.to_csv('baseline.csv', index=False)