import torch
from dense import AdaptiveClassifier
from encoder import Encoder, ImageEncoder
from esm_pretrained import ESM

# TODO: decide whether to allow for parallelizability -- should be able to switch between the two
class Model (torch.nn.Module):
    def __init__(self, device, num_layers, model_dim, num_heads, ff_dim, dropout=0.3, verbose=False, is_eval=False) -> None:
        super().__init__()
        self.esm = ESM(embed_dim=model_dim)
        self.device = device
        self.encoder = Encoder(num_layers=num_layers, model_dim=model_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
        self.classifier = AdaptiveClassifier(model_dim=model_dim)

        self.encoder.to(self.device)
        self.classifier.to(self.device)

        # wrap encoder and classifier in nn.DataParallel and send to GPU
        # model = torch.nn.Sequential(
        #     self.encoder,
        #     self.classifier
        # )
        # self.model = torch.nn.DataParallel(model)
        # self.model.to(self.device)

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
        # x = self.model(x)
        x = self.encoder(x)
        x = self.classifier(x)

        if self.verbose:
            print(x.size())
        return x
    
    # for feature extraction
    # model must be loaded on gpu, perhaps?
    def get_encoder_embeddings(self, x):
        x = self.esm.convert_batch(x)
        x = x.to(self.device)
        x = self.esm.get_representation(x)
        x = self.encoder(x)
        return x.detach().cpu()
    
    def extract_all_features(self, x):
        a = self.esm.convert_batch(x)
        a_gpu = a.to(self.device)
        b = self.esm.get_representation(a_gpu)
        c = self.encoder(b)

        features = {
            'tokens': a,
            'esm': b.detach().cpu(),
            'encoder': c.detach().cpu()
        }
        return features

    # get tokens from esm tokenizer
    def get_esm_tokens(self, x):
        return self.esm.convert_batch(x)

    # Used for score prediction and model interpretation
    # TODO: input requirement of ESM embeddings is awkward: list of (label, seq) tuples. find way to fix.
    def predict(self, sequences):

        inputs = []
        for i in range(len(sequences)):
            inputs.append(('', sequences[i]))
        
        # Turn single-node classification into 2-node binary classification
        # q = 1 - p
        output = self(inputs)
        complement = 1 - output
        result = torch.cat([complement, output], axis=1)

        return result
    
    def training_step(self, batch, batch_size, optimizer, loss_function):

        x, y = batch
        inputs = []
        for n in range(len(x)):
            inputs.append((y[n], x[n]))

        if batch_size > 1:
            targets = y.unsqueeze(1).float().to(device)
        else:
            targets = y.float().to(device)
        
        optimizer.zero_grad()
        outputs = self(inputs)

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

class ImageModel (torch.nn.Module):
    def __init__(self, device, num_layers, image_dim, model_dim, num_heads, ff_dim, dropout=0.3, verbose=False, is_eval=False) -> None:
        super().__init__()
        self.esm = ESM(embed_dim=model_dim)
        self.device = device
        self.encoder = ImageEncoder()
        self.classifier = AdaptiveClassifier(model_dim=model_dim)

        

        self.verbose = verbose
        self.is_eval = is_eval

        if (not self.is_eval):
            self.esm.model.to(self.device)

class SimplifiedModel (torch.nn.Module):
    # TODO: implement alternative architecture with MHSA module instead of transformer encoder.
    # option: projection layer for ESM embedding?

    def __init__(self, device, model_dim, num_heads, dropout=0.3, is_eval=False) -> None:
        super().__init__()
        self.esm = ESM(embed_dim=model_dim)
        self.device = device
        self.key = torch.nn.Linear()
        self.mhsa = torch.nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.classifier = AdaptiveClassifier(model_dim=model_dim)

        self.is_eval = is_eval

        if (not self.is_eval):
            self.esm.model.to(self.device)

    def forward(self, x):
        x = self.esm.convert_batch(x)
        # TODO: make x to device smarter (determine if esm is sent to device or not)
        if self.is_eval:
            x = self.esm.get_representation(x)
            x = x.to(self.device)
        else:
            x = x.to(self.device)
            x = self.esm.get_representation(x)
        
        attn_output, attn_weights = self.mhsa(x, x, x)
        x = self.classifier(attn_output)
        return x

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    my_model = SimplifiedModel(device, model_dim=320, num_heads=4, dropout=0.3)
    print(my_model)
    test = [(0, 'AGFQPQSQGMSLNDFQKQQKQAAPKPKKTLKLVSSSGIKLANATKKVGTKPAESDKKEEEKSAETKEPTKEPTKVEEPVKKEEKPVQTEEKTEEKSELPKVEDLKISESTHNTNNANVTSADALIKEQEEEVDDEVVNDMFGGKDHVSLIFMGHVDAGKSTMGGNLLYLTGSVDKRTIEKYEREAKDAGRQGWYLSWVMDTNKEERNDGKTIEVGKAYFETEKRRYTILDAPGHKMYVSEMIGGASQADVGVLVISARKGEYETGFERGGQTREHALLAKTQGVNKMVVVVNKMDDPTVNWSKERYDQCVSNVSNFLRAIGYNIKTDVVFMPVSGYSGANLKDHVDPKECPWYTGPTLLEYLDTMNHVDRHINAPFMLPIAAKMKDLGTIVEGKIESGHIKKGQSTLLMPNKTAVEIQNIYNETENEVDMAMCGEQVKLRIKGVEEEDISPGFVLTSPKNPIKSVTKFVAQIAIVELKSIIAAGFSCVMHVHTAIEEVHIVKLLHKLEKGTNRKSKKPPAFAKKGMKVIAVLETEAPVCVETYQDYPQLGRFTLRDQGTTIAIGKIVKIAE')]
    a = my_model(test)
    print(a)