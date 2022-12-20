import torch
from dense import AdaptiveClassifier
from encoder import Encoder
from esm_pretrained import ESM

class Model (torch.nn.Module):
    def __init__(self, device, num_layers, model_dim, num_heads, ff_dim, dropout=0.5) -> None:
        super().__init__()
        self.esm = ESM(embed_dim=model_dim)
        self.device = device
        self.encoder = Encoder(num_layers=num_layers, model_dim=model_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
        self.classifier = AdaptiveClassifier(model_dim=model_dim)

        self.encoder.to(self.device)
        self.classifier.to(self.device)

    def forward(self, x):
        x = self.esm.get_representation(x)
        x = x.to(self.device)
        x = self.encoder(x)
        return self.classifier(x)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    my_model = Model(device, num_layers=1, model_dim=320, num_heads=4, ff_dim=320, dropout=0.3)
    print(my_model)
    test = [(0, 'AGFQPQSQGMSLNDFQKQQKQAAPKPKKTLKLVSSSGIKLANATKKVGTKPAESDKKEEEKSAETKEPTKEPTKVEEPVKKEEKPVQTEEKTEEKSELPKVEDLKISESTHNTNNANVTSADALIKEQEEEVDDEVVNDMFGGKDHVSLIFMGHVDAGKSTMGGNLLYLTGSVDKRTIEKYEREAKDAGRQGWYLSWVMDTNKEERNDGKTIEVGKAYFETEKRRYTILDAPGHKMYVSEMIGGASQADVGVLVISARKGEYETGFERGGQTREHALLAKTQGVNKMVVVVNKMDDPTVNWSKERYDQCVSNVSNFLRAIGYNIKTDVVFMPVSGYSGANLKDHVDPKECPWYTGPTLLEYLDTMNHVDRHINAPFMLPIAAKMKDLGTIVEGKIESGHIKKGQSTLLMPNKTAVEIQNIYNETENEVDMAMCGEQVKLRIKGVEEEDISPGFVLTSPKNPIKSVTKFVAQIAIVELKSIIAAGFSCVMHVHTAIEEVHIVKLLHKLEKGTNRKSKKPPAFAKKGMKVIAVLETEAPVCVETYQDYPQLGRFTLRDQGTTIAIGKIVKIAE')]
    a = my_model(test)
    print(a)