from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.data.dataset import Dataset
from src.data.preprocessing import create_df
from src.utils.config_loader import load_config
from src.utils.seed import seed_everything

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')

seed_everything(config['general']['seed'])

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

# Data loading
train_df = create_df(base_dir / 'data/norne/ud/nob/no_bokmaal-ud-train.conllu')
train_dataset = Dataset(train_df, tokenizer, 25)
train_loader = DataLoader(train_dataset)
