import torch
from .dense import AdaptiveClassifier
from .encoder import Encoder, ImageEncoder
from .esm_pretrained import ESM

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
    
    def extract_features(self, x):
        a = self.esm.convert_batch(x)
        a_gpu = a.to(self.device)
        b = self.esm.get_representation(a_gpu)
        encoder_features = self.encoder.extract_features(b)
        c = encoder_features[-1]
        classifier_features = self.classifier.extract_features(c)
        
        for i in range(len(encoder_features)):
            encoder_features[i] = encoder_features[i].detach().cpu()
        
        for i in range(len(classifier_features)):
            classifier_features[i] = classifier_features[i].detach().cpu()

        features = {
            'tokens': a,
            'esm': b,
            'encoder': encoder_features,
            'classifier': classifier_features[0:-1],
            'output': classifier_features[-1]
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
        self.mhsa = torch.nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.classifier = AdaptiveClassifier(model_dim=model_dim)

        self.is_eval = is_eval

        self.mhsa.to(self.device)
        self.classifier.to(self.device)

        if (not self.is_eval):
            self.esm.model.to(self.device)

    def forward(self, x, extract_attn=False):
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

        if extract_attn == False:
            return self.classifier(attn_output)
        else:
            return (x, attn_output, attn_weights)

    def forward_on_lst_input(self, x, extract_attn=False):
        print('executing forward_on_lst_input')

        inputs = []
        for i in range(len(x)):
            inputs.append(('', x[i]))
    
        for n in inputs:
            print(n)
        
        result = self(inputs, extract_attn)
        return result

    def forward_on_single_seq(self, x, extract_attn=False):
        print('executing forward_on_single_seq')

        input = []
        input.append(('', x))

        print(input)

        result = self(input, extract_attn)
        print(result)
        return result
        

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    my_model = SimplifiedModel(device, model_dim=320, num_heads=4, dropout=0.3)
    print(my_model)
    test = [(0, 'AGFQPQSQGMSLNDFQKQQKQAAPKPKKTLKLVSSSGIKLANATKKVGTKPAESDKKEEEKSAETKEPTKEPTKVEEPVKKEEKPVQTEEKTEEKSELPKVEDLKISESTHNTNNANVTSADALIKEQEEEVDDEVVNDMFGGKDHVSLIFMGHVDAGKSTMGGNLLYLTGSVDKRTIEKYEREAKDAGRQGWYLSWVMDTNKEERNDGKTIEVGKAYFETEKRRYTILDAPGHKMYVSEMIGGASQADVGVLVISARKGEYETGFERGGQTREHALLAKTQGVNKMVVVVNKMDDPTVNWSKERYDQCVSNVSNFLRAIGYNIKTDVVFMPVSGYSGANLKDHVDPKECPWYTGPTLLEYLDTMNHVDRHINAPFMLPIAAKMKDLGTIVEGKIESGHIKKGQSTLLMPNKTAVEIQNIYNETENEVDMAMCGEQVKLRIKGVEEEDISPGFVLTSPKNPIKSVTKFVAQIAIVELKSIIAAGFSCVMHVHTAIEEVHIVKLLHKLEKGTNRKSKKPPAFAKKGMKVIAVLETEAPVCVETYQDYPQLGRFTLRDQGTTIAIGKIVKIAE')]
    a = my_model(test)
    print(a)