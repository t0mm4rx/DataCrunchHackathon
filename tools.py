import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def clean_data(df):
	for c in df.columns:
		if (not "enum" in c):
			# Removing inf
			df[c] = df[c].replace([np.inf, -np.inf], np.nan)
			# Fill missing values with mean value
			df[c] = df[c].fillna(df[c].mean())
		else:
			df[c] = df[c].fillna(0)
		# Remove outliers greater than 99%
		a = 0.00002
		min = df[c].quantile(a)
		max = df[c].quantile(1 - a)
		df[c] = df[c].clip(min, max)
		# Standarize data
		if (not "enum" in c and not "target" in c and not "bool" in c):
			df[c] = df[c] - df[c].mean()
			df[c] /= df[c].std()
		# One hot encoding
		if ("enum" in c):
			df = pd.concat([df, pd.get_dummies(df[c])], axis=1)
			df.drop(columns=[c], inplace=True)
	return df

def get_pca_features(features, dim=15):
	pca = PCA(n_components=dim)
	pc = pca.fit_transform(features)
	print("PCA keeps {:.6f}% of variance".format(pca.explained_variance_ratio_.sum() * 100))
	return (pc)

def get_features(df):
	features = df.drop(columns=['target_r', 'target_g', 'target_b', 'Feature_4', 'Feature_6'])
	return (features)