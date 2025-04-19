from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, ita_dataset, nl_dataset):
        super().__init__()
        """
        Inizializza i dataset nella classe
        """
        self.ita_dataset = ita_dataset
        self.nl_dataset = nl_dataset

    def __len__(self):
        """
        Resituisce la grandezza del dataset
        """
        return len(self.ita_dataset)

    def __getitem__(self, index):
        """
        Resistuisce un singole tensore di dimensione 1D
        """
        ita_data = self.ita_dataset[index]
        nl_data = self.nl_dataset[index]

        return ita_data, nl_data