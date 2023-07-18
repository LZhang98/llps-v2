import torch
from torch.utils.data import Dataset
import os
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None) -> None:
        self.labels = pd.read_csv(annotations_file)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.labels.iloc[index, 0])
        embedding = torch.load(data_path)
        label = self.labels.iloc[index, 1]
        if label == 2:
            label = 1
        return embedding, label

class SingleFileDataset(Dataset):
    def __init__(self, datafile, threshold=-1) -> None:
        self.data = pd.read_csv(datafile)
        if threshold > 0:
            self.data = self.data.loc[self.data['Sequence'].str.len() < threshold]
        self.sequences = self.data['Sequence'].reset_index(drop=True)
        self.categories = self.data['Source'].reset_index(drop=True)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        cat = self.categories[index]
        if cat in ['DrLLPS','PhaSePro','LLPSDB']:
            l = 1
        else:
            l = 0
        return seq, l

class SingleFileTestDataset(Dataset):
    def __init__(self, datafile, threshold=-1) -> None:
        self.data = pd.read_csv(datafile)
        if threshold > 0:
            self.data = self.data[self.data['sequences'].str.len() <= threshold]
        self.sequences = self.data['sequences']
        self.labels = self.data['labels']

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        label = self.labels[index]
        return seq, label

class ToyboxDataset(Dataset):
    def __init__(self) -> None:
        self.labels = [0, 1]
        self.sequences = [
            'AGFQPQSQGMSLNDFQKQQKQAAPKPKKTLKLVSSSGIKLANATKKVGTKPAESDKKEEEKSAETKEPTKEPTKVEEPVKKEEKPVQTEEKTEEKSELPKVEDLKISESTHNTNNANVTSADALIKEQEEEVDDEVVNDMFGGKDHVSLIFMGHVDAGKSTMGGNLLYLTGSVDKRTIEKYEREAKDAGRQGWYLSWVMDTNKEERNDGKTIEVGKAYFETEKRRYTILDAPGHKMYVSEMIGGASQADVGVLVISARKGEYETGFERGGQTREHALLAKTQGVNKMVVVVNKMDDPTVNWSKERYDQCVSNVSNFLRAIGYNIKTDVVFMPVSGYSGANLKDHVDPKECPWYTGPTLLEYLDTMNHVDRHINAPFMLPIAAKMKDLGTIVEGKIESGHIKKGQSTLLMPNKTAVEIQNIYNETENEVDMAMCGEQVKLRIKGVEEEDISPGFVLTSPKNPIKSVTKFVAQIAIVELKSIIAAGFSCVMHVHTAIEEVHIVKLLHKLEKGTNRKSKKPPAFAKKGMKVIAVLETEAPVCVETYQDYPQLGRFTLRDQGTTIAIGKIVKIAE',
            'MAAKKDYYAILGVPRNATQEEIKRAYKRLARQYHPDVNKSPEAEEKFKEINEAYAVLSDPEKRRIYDTYGTTEAPPPPPPGGYDFSGFDVEDFSEFFQELFGPGLFGGFGRRSRKGRDLRAELPLTLEEAFHGGERVVEVAGRRVSVRIPPGVREGSVIRVPGMGGQGNPPGDLLLVVRLLPHPVFRLEGQDLYATLDVPAPIAVVGGKVRAMTLEGPVEVAVPPRTQAGRKLRLKGKGFPGPAGRGDLYLEVRITIPERLTPEEEALWKKLAEAYYARA'
        ]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        seq = self.sequences[index]
        l = self.labels[index]
        return seq, l

class ProteomeDataset(Dataset):
    def __init__(self, datafile, seq_col_index, id_col_index, threshold=-1):
        df = pd.read_csv(datafile)
        if threshold > 0:
            df = df.loc[df[seq_col_index].str.len() <= threshold]
        self.data = df.reset_index(drop=True)
        self.seqs = self.data[seq_col_index]
        self.labels = self.data[id_col_index]

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, index):
        seq = self.seqs[index]
        label = self.labels[index]
        return seq, label

if __name__ == '__main__':
    dataset = SingleFileTestDataset('llps-v2/data/test_set_1_pos.csv', 1500)

    print(len(dataset))
    for i in range(len(dataset)):
        print(dataset[i])