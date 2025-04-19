from pandas import DataFrame

class DataProcessor:
    def __init__(self, dataset, train_ration):
        self.dataset = dataset
        self.len_dataset = len(dataset)

        self.train_ration = train_ration

    def split_train(self, split_train) -> DataFrame:
        """
        Prende la percentuale di dataset rispetto a split_train
        """
        train_dataset_ita = self.dataset['ita'][:split_train]
        train_dataset_nl = self.dataset['nl'][:split_train]

        train_dataset = DataFrame({
            'ita' : train_dataset_ita,
            'nl' : train_dataset_nl
        })

        return train_dataset
    
    def split_val(self, split_train)-> DataFrame:
        """
        Prende la percentuale di dataset da split_val
        """
        
        tst_dataset_ita = self.dataset['ita'][split_train:].reset_index(drop = True)
        tst_dataset_nl = self.dataset['nl'][split_train:].reset_index(drop = True)

        tst_dataset = DataFrame({
            'ita' : tst_dataset_ita,
            'nl' : tst_dataset_nl
        })

        return tst_dataset
    
    def split_dataset(self)-> DataFrame:
        """
        Formalizza le percentuali del dataset per:
        - Training
        - Validation
        E raccoglie i rispettivi dataset
        """

        split_train = int(self.train_ration * self.len_dataset)

        train_dataset = self.split_train(split_train)
        val_dataset = self.split_val(split_train)

        return train_dataset, val_dataset




