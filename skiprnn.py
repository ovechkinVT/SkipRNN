from torch import jit
import torch
from torch import nn

class SkipGRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        """Simple SkipGRU cell.
        Args:
            input_size (int): size of input space of GRU
            hidden_size (int): size of hidden space of GRU
        """
        super().__init__()
        self.inner_gru = nn.GRUCell(input_size, hidden_size)

    @jit.script_method
    def forward(self, input, hidden, mix):
        """
        Args:
            input (torch.FloatTensor): input vector for the current step
            hidden (torch.FloatTensor): hidden vector for the previous step
            mix (torch.FloatTensor): mixture vector which defines proportion of mix between previous step and current.
        Returns:
            hidden (torch.FloatTensor): hidden vetor for the current step
        """
        # type: (Tensor, Tensor, Tensor) -> Tensor

        output = self.inner_gru(input, hidden)
        output = output * mix + hidden * (1 - mix)

        return output

from typing import List, Tuple

class SkipGRULayer(jit.ScriptModule):
    __constants__ = ['hidden_size']
    def __init__(self, input_size, hidden_size):
        """
        Layer consisted of SkipGRU cells.
        Args:
            input_size (int): size of input space of GRU
            hidden_size (int): size of hidden space of GRU
        """
        super().__init__()
        self.cell = SkipGRUCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    @jit.script_method
    def forward(self, input, mix):
        """
        Forward pass on SkipGRU layer.
        Args:
            input (torch.FloatTensor): inputs for the SkipGRU layer of form [BxTxN]
            mix (torch.FloatTensor): mixture coefficients for thehidden steps of form [BxTx1]
        Returns:
            h (torch.FloatTensor): history of outputs by steps. Size [BxTxH]
            o (torch.FloatTensor): last hidden step of size [BxH]
        """
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        hidden = torch.zeros(input.shape[0], self.hidden_size,
                             device=self.cell.inner_gru.weight_ih.device)
        inputs = input.unbind(1)
        mixtures = mix.unbind(1)
        outputs = jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            hidden = self.cell(inputs[i], hidden, mixtures[i])
            outputs += [hidden]
        return torch.stack(outputs, 1), hidden

from .binarizers import HintonBinarizer, ConcreteBinarizer

class SkipGRU(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, layer=None, do_copy_weights=False):
        """
        SkipRNN layer with trivial binarizer (All betas are directly parametrized.)
        Args:
            input_size (int): size of input space for SkipGRU
            hidden_size (int): size of hidden space for SkipGRU
            layer (nn.GRU): original GRU layers to copy sizes and make a link to weights
        """
        super().__init__()
        if layer is not None:
            self.layer = SkipGRULayer(layer.input_size, layer.hidden_size)
            self.impute_weights(layer, do_copy_weights)
        elif (input_size is not None) and (hidden_size is not None):
            self.layer = SkipGRULayer(input_size, hidden_size)
        else:
            raise Exception('Either layer or input_size & hidden_size are required')

    def impute_weights(self, donor_layer, copy=False):
        """
        Get weights from instance of nn.GRU.
        Args:
            donor_layer (nn.GRU): pretrained layer to get weights from.
            copy (bool): if True, will make copy of weights instead of linkage.
        """
        assert isinstance(donor_layer, nn.GRU), 'Wrong type of donor layer. GRU required!'

        if copy:
            self.layer.cell.inner_gru.weight_ih.data = donor_layer.weight_ih_l0.clone()
            self.layer.cell.inner_gru.weight_hh.data = donor_layer.weight_hh_l0.clone()
            self.layer.cell.inner_gru.bias_ih.data = donor_layer.bias_ih_l0.clone()
            self.layer.cell.inner_gru.bias_hh.data = donor_layer.bias_hh_l0.clone()
        else:
            self.layer.cell.inner_gru.weight_ih = donor_layer.weight_ih_l0
            self.layer.cell.inner_gru.weight_hh = donor_layer.weight_hh_l0
            self.layer.cell.inner_gru.bias_ih = donor_layer.bias_ih_l0
            self.layer.cell.inner_gru.bias_hh = donor_layer.bias_hh_l0

    def forward(self, x, b, l=None):
        """
        Make forward SkipGRU pass.
        Args:
            x (torch.FloatTensor): input of size [BxTxN]. NB! sequences should be padded from the end, not from the start position.
            u (torch.FloatTensor): mixture coefficients of size [BxTx1]
            l (torch.LongTensor): lengths of the padded sequences. If not provided, output will contain last elements of sequences.
        Returns:
            h (torch.FloatTensor, nn.utils.rnn.PackedSequence): history of hidden states of size [BxTxH]
            o (torch.FloatTensor): last hidden state of size [BxH]
        """
        h, o = self.layer(x, b)
        if l is not None:
            o = h[torch.arange(h.shape[0]), l-1]
        return h, o
