from torch.utils.data import Dataset
import torch
class TranslationDataset(Dataset):
    def __init__(self, it_tokens, en_tokens):
        super().__init__()
        """
        Inizializza i dataset nella classe
        """
        self.it_tokens = it_tokens
        self.en_tokens = en_tokens
        
        self.max_len = max(max(len(en), len(it)) for en, it in zip(en_tokens, it_tokens))

    def __len__(self):
        """
        Resituisce la grandezza del dataset
        """
        return len(self.it_tokens)

    def __getitem__(self, index):
        """
        Resistuisce un singole tensore di dimensione 1D
        """
        it_data = self.it_tokens[index] + [0] * (self.max_len - len(self.it_tokens[index]))
        en_data = self.en_tokens[index] + [0] * (self.max_len - len(self.en_tokens[index]))

        return torch.LongTensor(it_data), torch.LongTensor(en_data)