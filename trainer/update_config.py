import yaml

def update_config(file_path):
    # Define the new config data
    new_config = {
        'number': '23456789',
        'symbol': "",
        'lang_char': 'abcdefghjkmnpqrstvwxyz',
        'experiment_name': 'en_filtered',
        'train_data': 'all_data/anotherOCR',
        'valid_data': 'all_data/anotherOCR/eval',
        'manualSeed': 1111,
        'workers': 4,
        'batch_size': 32,
        'num_iter': 30000,
        'valInterval': 1000,
        'saved_model': 'saved_models/model.pth',  # 'saved_models/en_filtered/iter_300000.pth',
        'FT': False,
        'optim': 'adam',
        'lr': 1.0,
        'beta1': 0.9,
        'rho': 0.95,
        'eps': 0.00000001,
        'grad_clip': 5,
        'select_data': 'train',
        'batch_ratio': '1',
        'total_data_usage_ratio': 1.0,
        'batch_max_length': 32,
        'imgH': 40,
        'imgW': 16,
        'rgb': False,
        'contrast_adjust': False,
        'sensitive': False,
        'PAD': True,
        'contrast_adjust': 0.0,
        'data_filtering_off': True,
        'Transformation': 'None',
        'FeatureExtraction': 'VGG',
        'SequenceModeling': 'BiLSTM',
        'Prediction': 'CTC',
        'num_fiducial': 20,
        'input_channel': 1,
        'output_channel': 512,
        'hidden_size': 512,
        'decode': 'greedy',
        'new_prediction': False,
        'freeze_FeatureFxtraction': False,
        'freeze_SequenceModeling': False
    }

    # Open the YAML file and write the new content
    with open(file_path, 'w') as file:
        yaml.dump(new_config, file, default_flow_style=False)

# Specify the path to the YAML file
file_path = 'config_files/en_filtered_config.yaml'

# Update the configuration
update_config(file_path)

print(f"Configuration has been updated in {file_path}")
