from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import torch
from umap import UMAP
from matplotlib import pyplot as plt

def check_dataset(dataset_name: str, column: str, split: str, max_length: int, model_id: str, device: str, output_path: str):
  device = torch.device(device)

  model = AutoModel.from_pretrained(model_id).to(device)
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model.eval()

  def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

  def process_row(row):
    batch_dict = tokenizer(row[column], max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**batch_dict)

    row['embedding'] = average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).squeeze(0).detach().cpu().numpy()
    return row

  dataset = load_dataset(dataset_name, split=split).filter(lambda row : row[column] != None)
  dataset = dataset.map(process_row)

  umap = UMAP(n_neighbors=15, n_components=2, metric='cosine')
  umap_data = umap.fit_transform(dataset['embedding'])

  plt.scatter(umap_data[:, 0], umap_data[:, 1], c='#7D7C7C')
  plt.title(f'Projection of {dataset_name}')
  plt.savefig(output_path)

if __name__ == '__main__':
  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('dataset', type=str)
  parser.add_argument('column', type=str)
  parser.add_argument('--split', type=str, default='train')
  parser.add_argument('--max-length', type=int, default=256)
  parser.add_argument('--device', type=str, default='cpu')
  parser.add_argument('--model', type=str, default='thenlper/gte-base')
  parser.add_argument('--output', type=str, default='output.png')

  args = parser.parse_args()
  check_dataset(args.dataset, args.column, args.split, args.max_length, args.model, args.device, args.output)

# Example usage:
# 
# CPU:
# python data_check.py textminr/topic-labeling label --max-length 128
# 
# GPU:
# python data_check.py textminr/topic-labeling label --max-length 128 --device cuda