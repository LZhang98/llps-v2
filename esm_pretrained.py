import torch
import esm.pretrained

class ESM():

    def __init__(self, embed_dim=320, eval=False):
        
        model_dict = {
            320: esm.pretrained.esm2_t6_8M_UR50D(),
            # 480: esm.pretrained.esm2_t12_35M_UR50D(),
            # 640: esm.pretrained.esm2_t30_150M_UR50D(),
            # 1280: esm.pretrained.esm2_t33_650M_UR50D()
        }
    
        self.model, self.alphabet = model_dict[embed_dim]
        self.batch_converter = self.alphabet.get_batch_converter()

    def get_representation(self, x):

        with torch.no_grad():
            results = self.model(x, repr_layers=[6], return_contacts=True)

        token_representations = results['representations'][6]

        return token_representations
    
    def convert_batch(self, x):
        batch_labels, batch_strs, batch_tokens = self.batch_converter(x)
        return batch_tokens

if __name__ == '__main__':
    data = [
        ("pos", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("neg", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE")
    ]

    data2 = ('MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG',
            'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE')
    print('1')
    print(data)
    print('2')
    print(len(data[0][1]), len(data[1][1]))
    # my_esm = ESM(320)
    # print(my_esm)
    # a = my_esm.get_representation(data)

    print('===esm2_t6_8M_UR50D===')
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()  # disables dropout for deterministic results
    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    token_representations = results['representations'][6]
    print(token_representations)
    print(token_representations.size())
