#!/usr/bin/env python

import tensorflow as tf
import tensorflow.keras as keras
#import tensorflow_addons as tfa


def Model(num_classes=2):
	input0 = tf.keras.layers.Input(shape=(None, 4), name='mirSeq')
	input1 = tf.keras.layers.Input(shape=(None, 4), name='mrnaTarget')

	mirConv = tf.keras.layers.Conv1D(128, 2, activation='relu')(input0) # orig_filter: 320
	targetConv = tf.keras.layers.Conv1D(128, 2, activation='relu')(input1) # orig_filter: 320

	mirDropout = tf.keras.layers.Dropout(0.2)(mirConv)
	targetDropout = tf.keras.layers.Dropout(0.2)(targetConv)

	mir_maxPooling = tf.keras.layers.MaxPooling1D()(mirDropout)
	target_maxPooling = tf.keras.layers.MaxPooling1D()(targetDropout)

	mir_dropout = tf.keras.layers.Dropout(0.2)(mir_maxPooling)
	target_dropout = tf.keras.layers.Dropout(0.2)(target_maxPooling)
	
	inputConcat = tf.keras.layers.Concatenate(axis=-1)([mir_dropout, target_dropout])

	# Bidirectional LSTM
	biLstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu'))(inputConcat)

	# Two Dense Layers
	x = tf.keras.layers.Dropout(0.2)(biLstm)
	x = tf.keras.layers.Dense(16, activation='relu')(x)
	x = tf.keras.layers.Dropout(0.2)(x)
	x = tf.keras.layers.Dense(2, activation='softmax', name='output')(x)

	#optimizer = tfa.optimizers.LAMB(learning_rate=5e-4)
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

	model = tf.keras.Model(inputs=[input0, input1], outputs=x)
	model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

	return model


import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class DataGenerator(keras.utils.Sequence):
	def __init__(self, dataFrameIn, list_IDs, labels, batch_size=32, n_classes=2, shuffle=True):
		self.dataFrameIn = dataFrameIn
		self.list_IDs = list_IDs
		self.labels = labels
		self.batch_size = batch_size
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()
	
	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.ceil(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indices of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)				

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples'
		thisBatchSize = len(list_IDs_temp)
		longestSeqLen = max(self.dataFrameIn['miRNA_seq'].str.len().max(), self.dataFrameIn['mRNA_seq_extended'].str.len().max())
		mirSeqEncoded = np.zeros((thisBatchSize, longestSeqLen, 4))
		mrnaTargetEncoded = np.zeros((thisBatchSize, longestSeqLen, 4))
		y = np.empty((thisBatchSize), dtype=int) # labels
		#token = dict(zip('ACGU', np.arange(4))) 

		# Vertical: residue position Horizontal: [A, C, G, U]
		for idx, ID in enumerate(list_IDs_temp):
			mirSeqLen = len(self.dataFrameIn.loc[ID, 'miRNA_seq'])
			mrnaTargetLen = len(self.dataFrameIn.loc[ID, 'mRNA_seq_extended'])

			"""
			mirSeq_integer_encoded = np.array([token[nt] for nt in np.array(list(self.dataFrameIn.loc[ID, 'miRNA_seq']))])
			mrnaTarget_integer_encoded = np.array([token[nt] for nt in np.array(list(self.dataFrameIn.loc[ID, 'mRNA_seq_extended']))])

			#label_encoder = LabelEncoder()
			onehot_encoder = OneHotEncoder(sparse=False)
			
			#mir_integer_encoded = label_encoder.fit_transform(np.array(mirSeq))
			mirSeq = mirSeq_integer_encoded.reshape(mirSeqLen, 1)
			mir_onehot_encoded = onehot_encoder.fit_transform(mirSeq)

			if list(set(np.arange(4)) - set(mirSeq_integer_encoded)):
				missing_nt = list(set(np.arange(4)) - set(mirSeq_integer_encoded))
				for nt_token in missing_nt:
					mir_onehot_encoded = np.insert(mir_onehot_encoded, nt_token, 0, axis=1)

			#mrna_integer_encoded = label_encoder.fit_transform(np.array(mrnaTarget))
			mrnaTarget = mrnaTarget_integer_encoded.reshape(mrnaTargetLen, 1)
			mrna_onehot_encoded = onehot_encoder.fit_transform(mrnaTarget)

			if list(set(np.arange(4)) - set(mrnaTarget_integer_encoded)):
				missing_nt = list(set(np.arange(4)) - set(mrnaTarget_integer_encoded))
				for nt_token in missing_nt:
					mrna_onehot_encoded = np.insert(mrna_onehot_encoded, nt_token, 0, axis=1)

			#miRNA_seq = mir_onehot_encoded.transpose()
			#mRNA_target = mrna_onehot_encoded.transpose()
			"""
			nucleotide_to_token = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'L': 4,
				'a': 0, 'c': 1, 'g': 2, 'u': 3, 't': 3, 'l': 4}

			mir_onehot_encoded = np.zeros((mirSeqLen, 5))
			mrna_onehot_encoded = np.zeros((mrnaTargetLen, 5))

			mir_column_encoding = [nucleotide_to_token[n] for n in list(self.dataFrameIn.loc[ID, 'miRNA_seq'])]
			mrna_column_encoding = [nucleotide_to_token[n] for n in list(self.dataFrameIn.loc[ID, 'mRNA_seq_extended'])]
			
			mir_row = np.arange(mirSeqLen)
			mrna_row = np.arange(mrnaTargetLen)

			mir_onehot_encoded[mir_row, mir_column_encoding] = 1
			mrna_onehot_encoded[mrna_row, mrna_column_encoding] = 1

			mir_onehot_encoded = mir_onehot_encoded[:,:-1]
			mrna_onehot_encoded = mrna_onehot_encoded[:,:-1]
			

			mirSeqEncoded[idx,:mirSeqLen,:4] = mir_onehot_encoded #miRNA_seq
			mrnaTargetEncoded[idx,:mrnaTargetLen,:4] = mrna_onehot_encoded #mRNA_target

			y[idx] = self.labels[ID]
		
		X = {'mirSeq': mirSeqEncoded.astype(np.float32), 'mrnaTarget': mrnaTargetEncoded.astype(np.float32)}
		#print(mirSeqEncoded.shape, mrnaTargetEncoded.shape, y.shape)
		#return X, y
		return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
			

batch_size=32
#numEpochs = 100

import tensorflow as tf
import pandas as pd

# Cross Validation Training
num_split = 10
"""
for split in range(num_split):
	train_split = pd.read_csv(f'train_test_full/set_{split}_train.csv', sep='\t')
	y_train = train_split['classLabel'].to_numpy()
	train_generator = DataGenerator(train_split, np.arange(len(train_split)), y_train, shuffle=True)
	y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)

	model = Model()
	#print(model.summary())
	modelWeightsName = f'train_test_full/weights/split{split}/weights_train_split_{split}'
	callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=modelWeightsName,save_weights_only=True,save_best_only=True,monitor='loss',verbose=0)]
	model.fit(train_generator, epochs=50, callbacks=callbacks)
"""        

# Cross Validation Testing
#"""
for split in range(num_split):
	test_split = pd.read_csv(f'train_test_full/set_{split}_test.csv', sep='\t')
	y_test = test_split['classLabel'].to_numpy()
	test_generator = DataGenerator(test_split, np.arange(len(test_split)), y_test, shuffle=False)
	y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

	model = Model()
	model.load_weights(f'train_test_full/weights/split{split}/weights_train_split_{split}')
	y_test_pred = model.predict(test_generator)
	testPredictions = pd.DataFrame({'classLabel': test_split.classLabel.values, 'NotBindingSite': y_test_pred[:,0].tolist(), 'BindingSite': y_test_pred[:,1].tolist()})
	testPredictions.to_csv(f'train_test_full/weights/split{split}/test_split_{split}_Predictions.txt', sep='\t', index=False)
#"""




























	
