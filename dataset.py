import torch
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, datafile, threshold=None) -> None:
        self.data = pd.read_csv(datafile)
        self.data = self.data[len(self.data['Sequence']) <= threshold]
        self.sequences = self.data['Sequence']
        self.categories = self.data['Category']

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        cat = self.categories[index]
        if cat == 'LLPS+':
            l = 1
        else:
            l = 0
        return seq, l

class SingleFileTestDataset(Dataset):
    def __init__(self, datafile, threshold=2000) -> None:
        self.data = pd.read_csv(datafile)
        self.sequences = self.data['sequences']
        self.labels = self.data['labels']
        self.threshold = threshold

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        label = self.labels[index]
        if len(seq) > self.threshold:
            seq = seq[0:self.threshold]
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

if __name__ == '__main__':
    dataset = SingleFileDataset('llps-v2/data/toy_dataset/ten_balanced.csv')

    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    for i, data in enumerate(dataloader):
        print(data)