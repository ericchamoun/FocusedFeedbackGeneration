import torch
from tqdm import tqdm

class Prediction():
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader
        
    def predict(self):
        self.model.eval()
        all_logits = []
        all_softmax_logits = []
        for idx, batch in tqdm(enumerate(self.loader), total=len(self.loader)):
            for k, v in batch.items():
                batch[k] = v.to("cpu")

            with torch.no_grad():
                outputs = self.model(**batch)
                _, logits = outputs[:2]
                all_logits+=logits[:, 1].tolist()
                all_softmax_logits+=torch.softmax(logits, dim=1)[:, 1].tolist()

        return all_logits,  all_softmax_logits