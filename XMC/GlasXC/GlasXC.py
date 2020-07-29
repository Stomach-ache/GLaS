import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm


def _get_sequential_module(config, doNormalize=False):
    seq_mdl = nn.Sequential()
    for i, layer in enumerate(config):
        if 'kwargs' in layer.keys():
            if doNormalize is True and layer['name'] == 'Linear':
                seq_mdl.add_module('{}_{}'.format(layer['name'], i),
                                weightNorm(getattr(nn, layer['name'])(**layer['kwargs']), name='weight'))
            else:
                seq_mdl.add_module('{}_{}'.format(layer['name'], i),
                                getattr(nn, layer['name'])(**layer['kwargs']))
        else:
            seq_mdl.add_module('{}_{}'.format(layer['name'], i),
                               getattr(nn, layer['name'])())
    return seq_mdl


class GlasXC(nn.Module):
    """
    Module for performing extreme classifcation with Autoencoders and Regressors.

    Args:
        input_encoder_config : List of dictionaries with layer types and configurations for the
                               input encoder.
        input_decoder_config : List of dictionaries with layer types and configurations for the
                               input decoder.
        output_encoder_config : List of dictionaries with layer types and configurations for the
                                output encoder.
        output_decoder_config : List of dictionaries with layer types and configurations for the
                                output decoder.
        regressor_config : List of dictionaries with layer types and configurations for the
                           regressor.

        An example of a configuration would be:
               [{'name': 'Linear',
                 'kwargs': {'in_features': 10, 'out_features': 10, 'bias': False}},
                {'name': 'ReLU',
                 'kwargs': {'inplace': True}},
                {'name': 'Linear',
                 'kwargs: {'in_features': 10, 'out_features': 10}},
               ]
        `kwargs` not specified will be considered to be the default options in the constructor
        functions of the sub-modules. For instance, in the above configuration: the last layer
        has no mention of `bias` for the `Linear` layer. This is `True` by default.

    The forward pass would return a three-tuple:
        - The first item is the input reconstructed using the input encoder
        - The second item is the output reconstructed using the output encoder
        - The third item is the decoded version of the regression output

    In addition, there are separate methods for encoding/decoding inputs/outputs
    and performing regression.
    """
    def __init__(self, input_encoder_config, input_decoder_config,
                 output_encoder_config, output_decoder_config, regressor_config):
        super(GlasXC, self).__init__()

        # Construct input encoder
        self.input_encoder = _get_sequential_module(input_encoder_config)

        # Construct input decoder
        self.input_decoder = _get_sequential_module(input_decoder_config)

        # Construct output encoder
        self.output_encoder = _get_sequential_module(output_encoder_config)

        # Construct output encoder
        self.output_decoder = _get_sequential_module(output_decoder_config, True)

        # Construct regressor
        self.regressor = _get_sequential_module(regressor_config)

    def forward(self, x, y):
        """
        Function to perform the forward pass.

        Args:
            x : This is a tensor of inputs (batched)
            y : This is a tensor of multi-hot outputs (batched)

        Returns:
            As described above - 4-tuple
        """

        ret_tup = (None, None,
                    #self.decode_input(self.encode_input(x)),
                   #self.decode_output(self.encode_output(y)),
                   self.decode_output(self.regressor_input(self.encode_input(x))),
                   #self.decode_output(self.regressor(self.encode_input(x))),
                   #self.predict(x),
                   #self.decode_output_weight(self.encode_output(y)))
                   self.decode_output_weight())

        #self.output_decoder[0].weight = F.normalize(self.output_decoder[0].weight)

        return ret_tup

    def regressor_input(self, x):
        #self.regressor[0].weight_g = Parameter(F.normalize(self.regressor[0].weight_g))
        #return F.normalize(self.regressor(x))
        return self.regressor(x)
        #return F.normalize(x, p=2, dim=1)
        #return x

    def encode_input(self, x):
        """
        Function to return the encoding of input using the input encoder

        Args:
            x : This is a tensor of inputs (batched)
            Batched encoding of inputs
        """
        return self.input_encoder(x)

    def decode_input(self, enc_x):
        """
        Function to return the decoding of encoded input using the input decoder

        Args:
            x : This is a tensor of input encodings (batched)

        Returns:
            Reconstruction of inputs
        """
        return self.input_decoder(enc_x)

    def encode_output(self, y):
        """
        Function to return the encoding of multi-hot output using the output encoder

        Args:
            x : This is a tensor of outputs (batched)

        Returns:
            Batched encoding of outputs
        """
        return self.output_encoder(y)

    def decode_output(self, enc_y):
        """
        Function to return the decoding of encoded output using the output decoder

        Args:
            x : This is a tensor of output encodings (batched)

        Returns:
            Reconstruction of multi-hot encoding of outputs
        """
        """
        for name, module in self.output_decoder.named_modules():
            print("name : ",name)
            print("module : ", module)
            print("Size of the decoder weight matrix is : ",module[0].weight.size())
            #decode_weight_matrix = module[0].weight.size()

        #print("Size of the decoder weight matrix is : ",self.output_decoder.Linear)
        print(self.output_decoder.parameters())
        """

        #print (self.output_decoder[0].weight == self.output_encoder[0].weight_v)
        #self.output_decoder[0].weight = F.normalize(self.output_decoder[0].weight)
        #with torch.no_grad():
        #    enc_y = F.normalize(enc_y)
        #    self.output_decoder[0].weight = F.normalize(self.output_decoder[0].weight)
        #self.output_decoder[0].weight_g = Parameter(F.normalize(self.output_decoder[0].weight_g))
        #print (self.output_decoder[0].weight)
        #print (torch.norm(self.output_decoder[0].weight, p=2, dim=1))

        #print (torch.norm(enc_y, p=2, dim=1))
        #print (torch.norm(self.output_decoder[0].weight, p=2, dim=1))
        #print ("Size of weight is: ", weight.size())
        #return self.output_decoder(F.normalize(enc_y))
        #return F.linear(enc_y, self.output_decoder[0].weight)
        return self.output_decoder(enc_y)


    def decode_output_weight(self, enc_y=None):
        """
        Function to return the decoding of encoded output using the output decoder
        This will work only of the decoder has one layer

        Args:
            x : This is a tensor of output encodings (batched)

        Returns:
            Weight matrix of the decode layer of dimensions L x d
        """
        """
        for name, module in self.output_decoder.named_modules():
            print("name : ",name)
            print("module : ",type(module[0]))
            #print("Size of the decoder weight matrix is : ",module[0].weight.size())
            decode_weight_matrix = module[0].weight.size()
        """

        for layer in self.output_decoder.modules():
            #print("Layer type  : ",type(layer))
            if isinstance(layer, nn.Linear):
                #print(layer.weight)
                #print("Layer weight size : ",layer.weight.size())
                #layer.weight = Parameter(F.normalize(layer.weight, p=2, dim=1))
                decode_weight_matrix = layer.weight

        #print (torch.norm(decode_weight_matrix, p=2, dim=1))
        #print (decode_weight_matrix)
        return decode_weight_matrix.t()



    def predict(self, x):
        """
        Function to provide predictions for a batch of datapoints.

        Args:
            x : This is a tensor of batch inputs
        """
        #x = self.regressor(self.encode_input(x))
        #for layer in self.output_decoder.modules():
        #    if isinstance(layer, nn.Linear):
        #        weight = Parameter(F.normalize(layer.weight, dim=1))

        #print ("Size of weight is: ", weight.size())
        #return F.linear(F.normalize(x), weight)
        #print (x)
        #print (self.output_decoder[0].weight)
        return self.decode_output(self.regressor_input(self.encode_input(x)))
        #return self.decode_output(self.regressor(self.encode_input(x)))

    def get_embedding(self, x):
        return self.regressor_input(self.encode_input(x))
