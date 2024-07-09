#!/usr/bin/env python

import pandas as pd


def binarizeTranscript(transcript):
	#print(type(transcript))
	len_transcript = len(transcript)
	nucleotide_to_binary = {'U':'0,0,0,1', 'G':'0,0,1,0', 'A':'0,1,0,0', 'C':'1,0,0,0', 'T': '0,0,0,1', 'L':'0,0,0,0',
				'u':'0,0,0,1', 'g':'0,0,1,0', 'a':'0,1,0,0', 'c':'1,0,0,0', 't': '0,0,0,1', 'l':'0,0,0,0',
				'uracil':'0,0,0,1', 'guanine':'0,0,1,0', 'adenine':'0,1,0,0', 'cytosine':'1,0,0,0', 'thymine': '0,0,0,1'
				}

	binarized_transcript = ''
	for nucleotide in transcript:
		#print(nucleotide)
		binarized_transcript = binarized_transcript+nucleotide_to_binary[nucleotide]+','

	return binarized_transcript[:-1]


def binarizedToTranscript(binarizedTranscript):
	len_binarizedTranscript = len(binarizedTranscript)
	binary_to_nucleotide = {'0,0,0,1':'U', '0,0,1,0':'G', '0,1,0,0':'A', '1,0,0,0':'C', '0,0,0,0':'L'}
			
	# Split binarizedTranscript into groups of 7 characters
	binarizedNucleotides = []
	for idx in range(len(binarizedTranscript)):
		if idx%8 == 0 and idx != 0:
			binarizedNucleotides.append(binarizedTranscript[idx-8:idx-1])
	binarizedNucleotides.append(binarizedTranscript[len(binarizedTranscript)-8+1:])

	transcript_sequence = ''
	for binarizedNucleotide in binarizedNucleotides:
		#print(nucleotide)
		transcript_sequence = transcript_sequence+binary_to_nucleotide[binarizedNucleotide]

	return transcript_sequence


def collectCrossValSets(filename, outdir):
	# Format filename: [InteractionBinary, mRNA, miRNA]
	set_interactions = pd.read_csv(f'../../miRAW/miraw_data/PLOSComb/Data/CrossVal/RelaxedPW60_14K_35K/{filename}', header=None)
	print(len(set_interactions))

	classLabel = set_interactions.iloc[:, 0]
	binary = set_interactions.iloc[:, 1:281].astype(str)
	binary = binary.apply(lambda x: ','.join(x), axis=1)	
	#print(binary)

	set_interactions = set_interactions.iloc[:, 281:]
	set_interactions = pd.concat([binary, set_interactions, classLabel], ignore_index=True, axis=1)
	set_interactions.rename(columns={0: 'InteractionBinary', 1: 'mRNA', 2: 'miRNA', 3: 'classLabel'}, inplace=True)
	#print(set_interactions)

	allTrainingSites = pd.read_csv('../../miRAW/miraw_data/PLOSComb/Data/ValidTargetSites/allTrainingSites.txt', sep='\t')
	#print(allTrainingSites)

	# Collect targets for custom model
	#"""
	miRNA = []
	mRNA = []
	classLabel = []
	for idx, interaction in set_interactions.iterrows():
		bindingSite = allTrainingSites.loc[(allTrainingSites['miRNA'] == interaction['miRNA']) & (allTrainingSites['EnsemblId'] == interaction['mRNA'])]
		#print(bindingSite)
		if len(bindingSite) < 1:
			#print(interaction)
			# miRNA is NaN
			if pd.isna(interaction['miRNA']):
				mRNA_sequence = allTrainingSites['mRNA_Site_Transcript'].loc[allTrainingSites['EnsemblId'] == interaction['mRNA']].drop_duplicates().values[0][::-1]
				# Get miRNA sequence (nucleotide format)
				interactionBinary = interaction['InteractionBinary']
				miRNA_binary = interactionBinary[:-40*8]
				miRNA_sequence = binarizedToTranscript(miRNA_binary)
				#print(len(interactionBinary), interactionBinary)
				print(miRNA_sequence, mRNA_sequence)
			if pd.isna(interaction['mRNA']):
				miRNA_sequence = allTrainingSites['mature_miRNA_Transcript'].loc[allTrainingSites['miRNA'] == interaction['miRNA']].drop_duplicates().values[0]
				# Get mRNA sequence (nucleotide format)
				interactionBinary = interaction['InteractionBinary']
				mRNA_binary = interactionBinary[(-40*8)+1:]
				mRNA_sequence = binarizedToTranscript(mRNA_binary).replace('U', 'T') # already reversed
				#mRNA_sequence = allTrainingSites.loc[(allTrainingSites['mature_miRNA_Transcript'] == miRNA_sequence.replace('L', '')) & (allTrainingSites['EnsemblId'] == interaction['mRNA'])]
				print(mRNA_sequence[::-1], miRNA_sequence)
			
			miRNA.append(miRNA_sequence)
			mRNA.append(mRNA_sequence)
			classLabel.append(interaction['classLabel'])
			#continue
		elif len(bindingSite) > 1:
			for mRNA_transcript in bindingSite['mRNA_Site_Transcript']:
				#len_transcript_binary = len(mRNA_transcript)*4*3
				#print(type(mRNA_transcript))
				binarized_transcript = binarizeTranscript(mRNA_transcript[::-1])
				interaction_cropped = interaction['InteractionBinary'][-len(binarized_transcript):]	
				#print('match?', interaction_cropped == binarized_transcript)
				if interaction_cropped == binarized_transcript:
					miRNA.append(bindingSite['mature_miRNA_Transcript'].values[0])
					mRNA.append(mRNA_transcript[::-1])
					classLabel.append(interaction['classLabel'])
					break
				#print('interaction', interaction_cropped)
				#print('binarized', binarized_transcript)
		
			#break
		else:
			#print(bindingSite['mRNA_Site_Transcript'].item())
			#binarized_transcript = binarizeTranscript(bindingSite['mRNA_Site_Transcript'].item()[::-1])
			#interaction_cropped = interaction['InteractionBinary'][-len(binarized_transcript):]
			#print('match?', interaction_cropped == binarized_transcript)
			miRNA.append(bindingSite['mature_miRNA_Transcript'].values[0])
			mRNA.append(bindingSite['mRNA_Site_Transcript'].values[0][::-1])	
			classLabel.append(interaction['classLabel'])
			#print(classLabel)
			

	#"""
	#"""
	print(len(miRNA), len(mRNA))
	dataset = pd.DataFrame({'miRNA_seq': miRNA, 'mRNA_seq_extended': mRNA, 'classLabel': classLabel})	
	dataset.to_csv(outdir, sep='\t', index=False)
	#"""

#"""
num_splits = 10
for split in range(num_splits):
	collectCrossValSets(f'set_{split}_train.csv', f'train_test_full/set_{split}_train.csv')
	collectCrossValSets(f'set_{split}_test.csv', f'train_test_full/set_{split}_test.csv')
#"""
#collectCrossValSets('set_0_train.csv', 'set_0_train.csv')




