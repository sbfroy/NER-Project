from pathlib import Path
from src.data.preprocessing import create_df
from src.utils.config_loader import load_config

base_dir = Path(__file__).parent.parent

config = load_config(base_dir / 'model_params.yaml')


train_df = create_df(base_dir / 'data/norne/ud/nob/no_bokmaal-ud-train.conllu')



print(train_df.head())