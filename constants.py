metrics = ['trustworthiness', 'continuity', 'shepard_correlation', 'normalized_stress']
samples = 1000
hover_to_view = True #Toggle for switching views by either hovering over bars, or clicking on bars


metrics_dir = 'metrics'
output_dir = 'projections'
analysis_dir = 'analysis'
use0_1bounds = False
user_modes = ['free', 'eval_full', 'eval_half', 'image', 'evalimage']
user_mode = 'free'

ordinal_datasets = ['Wine', 'Concrete', 'Software',]
categorical_datasets = ['AirQuality', 'Reuters', 'WisconsinBreastCancer']

evaluation_set = [('WisconsinBreastCancer', 'PCA'), ('Wine', 'TSNE'), ('Wine', 'PCA'), ('Concrete', 'TSNE'), ('Reuters', 'AE'), ('Reuters', 'TSNE'), ('Software', 'TSNE')]
required_view_count = 3
output_file = 'evaluationdata/evaluationdata.pkl'

debug_mode = False