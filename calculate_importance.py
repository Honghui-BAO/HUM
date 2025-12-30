import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pickle
from dataloader import get_dataloader
from param import parse_args
from utils import print_rank0
from hum import HUM

class ImportanceScorer:
    def __init__(self, model):
        self.model = model
        self.layer_scores = {"attn": [], "mlp": []}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        # Access the underlying Qwen2Model layers
        # In HUM, self.model.llm is the PeftModel or AutoModel
        llm = self.model.llm
        if hasattr(llm, "base_model"):
            layers = llm.base_model.model.model.layers
        else:
            layers = llm.model.layers

        for i, layer in enumerate(layers):
            # Hook for Attention
            self.hooks.append(layer.self_attn.register_forward_hook(
                self._generate_hook(i, "attn")
            ))
            # Hook for MLP
            self.hooks.append(layer.mlp.register_forward_hook(
                self._generate_hook(i, "mlp")
            ))

    def _generate_hook(self, layer_idx, part_type):
        def hook(module, input, output):
            # input[0] is the input tensor x
            # output[0] is the output of the sub-layer (the residual F(x))
            # Qwen2 attention and mlp return (hidden_states, ...) 
            # If output is a tuple, take the first element
            res = output[0] if isinstance(output, tuple) else output
            x = input[0]
            
            # Importance = 1 - CosineSimilarity(x, x + res)
            # Higher score means the layer changed the embedding significantly.
            # We want to identify layers where F(x) is small relative to x.
            with torch.no_grad():
                cosine_sim = F.cosine_similarity(x, x + res, dim=-1).mean()
                score = 1 - cosine_sim
                
                # Store score
                if part_type == "attn":
                    self.current_batch_attn[layer_idx] = score.item()
                else:
                    self.current_batch_mlp[layer_idx] = score.item()
        return hook

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def score(self, dataloader, num_batches=100):
        self.model.eval()
        num_layers = len(self.layer_scores["attn"]) if self.layer_scores["attn"] else 28 # Default for Qwen2.5-1.5B
        
        all_attn_scores = []
        all_mlp_scores = []

        device = next(self.model.parameters()).device

        for i, batch in enumerate(tqdm(dataloader, desc="Scoring layers", total=num_batches)):
            if i >= num_batches:
                break
            
            # Initialize batch scores
            self.current_batch_attn = {}
            self.current_batch_mlp = {}

            # Transfer to device
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            # Forward pass (this will trigger hooks)
            with torch.no_grad():
                # We only need the embeddings for scoring
                self.model.get_embedding(batch['sequence_input_ids'], batch['sequence_attention_mask'])

            all_attn_scores.append([self.current_batch_attn[l] for l in sorted(self.current_batch_attn.keys())])
            all_mlp_scores.append([self.current_batch_mlp[l] for l in sorted(self.current_batch_mlp.keys())])

        # Average over batches
        avg_attn = np.mean(all_attn_scores, axis=0)
        avg_mlp = np.mean(all_mlp_scores, axis=0)

        return avg_attn, avg_mlp

def plot_scores(avg_attn, avg_mlp, save_path="layer_importance.png"):
    layers = np.arange(len(avg_attn))
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, avg_attn, label="Attention Importance", marker='o', linestyle='-', color='blue')
    plt.plot(layers, avg_mlp, label="MLP Importance", marker='s', linestyle='--', color='red')
    
    plt.title("Layer Importance Scoring (EffiR)")
    plt.xlabel("Layer Index")
    plt.ylabel("Importance Score (1 - Cosine Similarity)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Highlight potentially redundant layers (low score)
    plt.axhline(y=np.percentile(avg_mlp, 25), color='gray', linestyle=':', label='MLP 25th percentile')
    
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def main():
    args = parse_args()
    args.distributed = False # Run on single GPU for scoring
    
    # Fix dataset name if needed
    from utils import amazon_dataset2fullname
    args.dataset = amazon_dataset2fullname[args.dataset] if args.dataset in amazon_dataset2fullname.keys() else args.dataset
    args.dataset = args.dataset + f'-{args.ratio}-{args.user_k}-{args.item_k}'
    
    # Initialize rank/gpu for standalone execution
    args.rank = 0
    args.gpu = 0
    args.num_gpus = 1

    # Allow overriding data path if provided via CLI or env
    valid_data_path = os.environ.get("VALID_DATA_PATH", f"{args.data_path}/{args.dataset}/valid_data.pkl")
    print(f"Using validation data from: {valid_data_path}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.root_path + args.backbone, padding_side="left")
    special_tokens_dict = {'additional_special_tokens': ['<|emb|>', '<|thought|>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    token_id_dict = {t: tokenizer.convert_tokens_to_ids(t) for t in special_tokens_dict['additional_special_tokens']}
    
    text = 'Compress the following description about the user or item into the last token:'
    tokenized_text = tokenizer(text, padding=False, add_special_tokens=False)['input_ids']
    
    # Load Dataloader (Validation set)
    # The dataloader needs tokenized_text and tokenized_domain
    # In run_hum.py, they both use tokenized_text.
    _, val_loader, _ = get_dataloader(args, tokenizer, tokenized_text, {d: tokenized_text for d in [args.dataset]}, token_id_dict)
    
    # Load Model
    model = HUM(args, tokenizer)
    if args.load:
        print(f"Loading weights from {args.load}")
        weights = torch.load(args.load, map_location='cpu')
        model.load_state_dict(weights, strict=False)
    
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scoring
    scorer = ImportanceScorer(model)
    avg_attn, avg_mlp = scorer.score(val_loader, num_batches=128)
    scorer.clear_hooks()
    
    # Plotting
    plot_scores(avg_attn, avg_mlp, save_path=f"layer_importance_{args.dataset}.png")
    
    # Save raw scores
    scores = {"attn": avg_attn.tolist(), "mlp": avg_mlp.tolist()}
    import json
    with open(f"layer_importance_{args.dataset}.json", "w") as f:
        json.dump(scores, f, indent=4)
    print("Scores saved to JSON.")

if __name__ == "__main__":
    main()
