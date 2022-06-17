metrics = ['trustworthiness', 'continuity', 'shepard_correlation', 'normalized_stress']
samples = 1000
hover_to_view = True #Toggle for switching views by either hovering over bars, or clicking on bars


metrics_dir = 'metrics'
output_dir = 'projections'
user_modes = ['free', 'eval_full', 'eval_half']
user_mode = 'eval_full'

ordinal_datasets = ['Wine', 'Concrete', 'Software',]
categorical_datasets = ['Reuters', 'WisconsinBreastCancer']

evaluation_set = [('WisconsinBreastCancer', 'PCA'), ('Wine', 'TSNE'), ('Wine', 'PCA'), ('Concrete', 'TSNE'), ('Reuters', 'AE'), ('Reuters', 'TSNE'), ('Software', 'TSNE')]
required_view_count = 3
output_file = 'evaluationdata.pkl'

debug_mode = False