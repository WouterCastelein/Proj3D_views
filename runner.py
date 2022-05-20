import os
import sys
import ae
from glob import glob
import numpy as np
import pandas as pd
from sklearn import manifold, decomposition, random_projection

import constants
import tapkee
import umap
import MulticoreTSNE

projection_names = {
                    'A': ['SPE', 'LPP', 'L-LTSA', 'NPE', 'L-MDS', 'DM'],
                    'B': ['LLE', 'H-LLE', 'M-LLE', 'LTSA', 'ISO', 'MDS', 'N-MDS', 'LE'],
                    'C': ['AE', 'TSNE', 'UMAP', 'FA', 'F-ICA', 'NMF', 'T-SVD'],
                    'D': ['PCA', 'I-PCA', 'S-PCA', 'K-PCA-P', 'K-PCA-R', 'K-PCA-S', 'G-RP', 'S-RP'],
                    'S': ['TSNE', 'UMAP', 'AE', 'PCA', 'MDS']
                    # In our subset of projection techniques we aimed to include well-known techniques as well as
                    # well-performing techniques, which resulted in the following selection:
                    #: TSNE: Well-performing, Popular, nonlinear and focuses on local neighboordhood optimization (rather good clustering than perfect projection)
                    #: UMAP: Well-performing, nonlinear, local and considered to be one of the best projection techniques by espadoto 2019 survey
                    #: AE: Well-performing, Nonlinear, global technique,
                    #: PCA: best known technique, linear, global
                    #: MDS: another nonlinear global technique, from the well known family of multidimensional scaling techniques (http://eprints.cs.univie.ac.at/3992/4/sedlmair2013infovis.pdf)

                }

def get_projections(n_components):
    projections = {
                    'DM': tapkee.DiffusionMaps(n_components=n_components),
                    'SPE': tapkee.StochasticProximityEmbedding(n_components=n_components),
                    'LPP': tapkee.LocalityPreservingProjections(n_components=n_components),
                    'L-LTSA': tapkee.LinearLocalTangentSpaceAlignment(n_components=n_components),
                    'NPE': tapkee.NeighborhoodPreservingEmbedding(n_components=n_components),
                    'L-MDS': tapkee.LandmarkMDS(n_components=n_components),
                    'AE': ae.AutoencoderProjection(n_components=n_components),
                    'LLE': manifold.LocallyLinearEmbedding(n_components=n_components, eigen_solver='dense', method='standard', n_jobs=4, random_state=420),
                    'H-LLE': manifold.LocallyLinearEmbedding(n_components=n_components, eigen_solver='dense', method='hessian', n_neighbors=10, n_jobs=4, random_state=420),
                    'M-LLE': manifold.LocallyLinearEmbedding(n_components=n_components, eigen_solver='dense', method='modified', n_jobs=4, random_state=420),
                    'LTSA': manifold.LocallyLinearEmbedding(n_components=n_components, eigen_solver='dense', method='ltsa', n_jobs=4, random_state=420),
                    'ISO': manifold.Isomap(n_components=n_components, eigen_solver='dense', n_jobs=4),
                    'MDS': manifold.MDS(n_components=n_components, metric=True, n_jobs=4, random_state=420),
                    'N-MDS': manifold.MDS(n_components=n_components, metric=False, n_jobs=4, random_state=420),
                    'LE': manifold.SpectralEmbedding(n_components=n_components, n_jobs=4, random_state=420),
                    'TSNE': MulticoreTSNE.MulticoreTSNE(n_components=n_components, perplexity=30, n_jobs=4, random_state=420),
                    'UMAP': umap.UMAP(n_components=n_components, n_neighbors=25, random_state=420),
                    'FA': decomposition.FactorAnalysis(n_components=n_components, random_state=420),
                    'F-ICA': decomposition.FastICA(n_components=n_components, random_state=420),
                    'NMF': decomposition.NMF(n_components=n_components),
                    'T-SVD': decomposition.TruncatedSVD(n_components=n_components, random_state=420),
                    'PCA': decomposition.PCA(n_components=n_components, random_state=420),
                    'I-PCA': decomposition.IncrementalPCA(n_components=n_components),
                    'S-PCA': decomposition.SparsePCA(n_components=n_components, random_state=420, n_jobs=4),
                    'K-PCA-P': decomposition.KernelPCA(n_components=n_components, random_state=420, n_jobs=4, kernel='poly'),
                    'K-PCA-R': decomposition.KernelPCA(n_components=n_components, random_state=420, n_jobs=4, kernel='rbf'),
                    'K-PCA-S': decomposition.KernelPCA(n_components=n_components, random_state=420, n_jobs=4, kernel='sigmoid'),
                    'G-RP': random_projection.GaussianRandomProjection(n_components=n_components, random_state=420),
                    'S-RP': random_projection.SparseRandomProjection(n_components=n_components, random_state=420),
    }

    return projections

if __name__ == '__main__':
        dataset_name = sys.argv[1]
        selection = sys.argv[2]

        input_file = glob('data/{0}/*-src.csv'.format(dataset_name))[0]

        df = pd.read_csv(input_file, sep=';', header=0)
        X_high = df.to_numpy()

        selected_projections = projection_names[selection]

        header = ['x', 'y', 'z']

        for n_components in [2, 3]:
            projections = get_projections(n_components)

            for proj_name in selected_projections:
                p = projections[proj_name]
                output_file = os.path.join(constants.output_dir, '{0}-{1}-{2}d.csv'.format(dataset_name, proj_name, n_components))

                print('dim: {0}, proj: {1}'.format(n_components, proj_name))
                print('output_file: {0}'.format(output_file))

                X_low = p.fit_transform(X_high)

                #Scale to fit exactly in the range (0, 1)
                X_low -= np.min(X_low)
                X_low /= np.max(X_low)
                df_low = pd.DataFrame(X_low)
                df_low.columns = header[:n_components]

                df_low.to_csv(output_file, index=None, sep=';')
