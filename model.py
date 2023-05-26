import torch
from dense import AdaptiveClassifier
from encoder import Encoder
from esm_pretrained import ESM

class Model (torch.nn.Module):
    def __init__(self, device, num_layers, model_dim, num_heads, ff_dim, dropout=0.5, verbose=False, is_eval=False) -> None:
        super().__init__()
        self.esm = ESM(embed_dim=model_dim)
        self.device = device
        self.encoder = Encoder(num_layers=num_layers, model_dim=model_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
        self.classifier = AdaptiveClassifier(model_dim=model_dim)

        # wrap encoder and classifier in nn.DataParallel and send to GPU
        model = torch.nn.Sequential(
            self.encoder,
            self.classifier
        )
        self.model = torch.nn.DataParallel(model)
        self.model.to(self.device)

        self.verbose = verbose
        self.is_eval = is_eval

        if (not self.is_eval):
            self.esm.model.to(self.device)

    def forward(self, x):
        if self.verbose:
            print(x)
        x = self.esm.convert_batch(x)
        if self.verbose:
            print(x)
        # TODO: make x to device smarter (determine if esm is sent to device or not)
        if self.is_eval:
            x = self.esm.get_representation(x)
            x = x.to(self.device)
        else:
            x = x.to(self.device)
            x = self.esm.get_representation(x)
        if self.verbose:
            print(x.size())
        x = self.model(x)
        if self.verbose:
            print(x.size())
        return x

    # Used for score prediction and model interpretation
    # TODO: input requirement of ESM embeddings is awkward: list of (label, seq) tuples. find way to fix.
    def predict(self, sequences):

        inputs = []
        for i in range(len(sequences)):
            inputs.append(('', sequences[i]))
        
        output = self(inputs)
        
        return output
    
    def training_step(self, batch):
        loss = ...
        return loss
    
    def validation_step(self, batch):
        loss = ...
        acc = ...
        return loss, acc
    
    def validation_epoch_end(self, outputs):
        epoch_loss = ...
        epoch_acc = ...
        return epoch_loss, epoch_acc


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    my_model = Model(device, num_layers=1, model_dim=320, num_heads=4, ff_dim=320, dropout=0.3, verbose=True)
    print(my_model)
    test = [(0, 'AGFQPQSQGMSLNDFQKQQKQAAPKPKKTLKLVSSSGIKLANATKKVGTKPAESDKKEEEKSAETKEPTKEPTKVEEPVKKEEKPVQTEEKTEEKSELPKVEDLKISESTHNTNNANVTSADALIKEQEEEVDDEVVNDMFGGKDHVSLIFMGHVDAGKSTMGGNLLYLTGSVDKRTIEKYEREAKDAGRQGWYLSWVMDTNKEERNDGKTIEVGKAYFETEKRRYTILDAPGHKMYVSEMIGGASQADVGVLVISARKGEYETGFERGGQTREHALLAKTQGVNKMVVVVNKMDDPTVNWSKERYDQCVSNVSNFLRAIGYNIKTDVVFMPVSGYSGANLKDHVDPKECPWYTGPTLLEYLDTMNHVDRHINAPFMLPIAAKMKDLGTIVEGKIESGHIKKGQSTLLMPNKTAVEIQNIYNETENEVDMAMCGEQVKLRIKGVEEEDISPGFVLTSPKNPIKSVTKFVAQIAIVELKSIIAAGFSCVMHVHTAIEEVHIVKLLHKLEKGTNRKSKKPPAFAKKGMKVIAVLETEAPVCVETYQDYPQLGRFTLRDQGTTIAIGKIVKIAE')]
    a = my_model(test)
    print(a)