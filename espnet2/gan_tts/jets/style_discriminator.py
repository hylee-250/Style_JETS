import torch
import torch.nn
import torch.nn.functional as F
from espnet2.gan_tts.jets.saln import ConvNorm, LinearNorm, MultiHeadAttention, get_sinusoid_encoding_table
from espnet2.gan_tts.jets.loss import LSGANLoss

LEAKY_RELU = 0.1


def dot_product_logit(a, b):
    n = a.size(0)
    m = b.size(0)
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = (a*b).sum(dim=2)
    return logits


class MetaDiscriminator(torch.nn.Module):
    ''' Discriminator '''
    def __init__(self, n_speakers,n_mel,hidden_dim,style_dim,style_kernel_size,style_head,encoder_hidden,max_seq_len):
        super(MetaDiscriminator, self).__init__()

        self.style_D = StyleDiscriminator(
            n_speakers, 
            n_mel, 
            style_dim, 
            style_dim,
            style_kernel_size,
            style_head)

        self.phoneme_D = PhonemeDiscriminator(
            n_mel, 
            encoder_hidden, 
            max_seq_len)


    def forward(self, mels, srcs, ws, sids, mask):
        # mels = mels.masked_fill(mask.unsqueeze(-1), 0)
            
        t_val = self.phoneme_D(mels, srcs, mask)
        s_val, ce_loss = self.style_D(mels, ws, sids, mask)

        return t_val, s_val, ce_loss
    
    def get_criterion(self):
        return LSGANLoss()



class StyleDiscriminator(torch.nn.Module):
    ''' Style Discriminator '''
    def __init__(self, n_speakers, input_dim, hidden_dim, style_dim, kernel_size, n_head):
        super(StyleDiscriminator, self).__init__()

        self.style_prototypes = torch.nn.Embedding(n_speakers, style_dim)

        self.spectral = torch.nn.Sequential(
            LinearNorm(input_dim, hidden_dim, spectral_norm=True),
            torch.nn.LeakyReLU(LEAKY_RELU),
            LinearNorm(hidden_dim, hidden_dim, spectral_norm=True),
            torch.nn.LeakyReLU(LEAKY_RELU),
        )

        self.temporal = torch.nn.ModuleList([torch.nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim, kernel_size, spectral_norm=True),
            torch.nn.LeakyReLU(LEAKY_RELU)) for _ in range(2)])

        self.slf_attn = MultiHeadAttention(n_head, hidden_dim, hidden_dim//n_head, hidden_dim//n_head, spectral_norm=True) 

        self.fc = LinearNorm(hidden_dim, hidden_dim, spectral_norm=True)

        self.V = LinearNorm(style_dim, hidden_dim, spectral_norm=True)

        self.w = torch.nn.Parameter(torch.ones(1))
        self.b = torch.nn.Parameter(torch.zeros(1))

    def temporal_avg_pool(self, xs, mask):
        xs = xs.masked_fill(mask.unsqueeze(-1), 0)
        len_ = (~mask).sum(dim=1).unsqueeze(1)
        xs = torch.sum(xs, dim=1)
        xs = torch.div(xs, len_)
        return xs

    def forward(self, mels, ws, sids, mask):
        max_len = mels.shape[1]

        # Update style prototypes
        if ws is not None:
            style_prototypes = self.style_prototypes.weight.clone()
            logit = dot_product_logit(ws, style_prototypes) 
            cls_loss = F.cross_entropy(logit, sids)
        else: 
            cls_loss = None

        # Style discriminator
        x = self.spectral(mels)

        for conv in self.temporal:
            residual = x
            x = x.transpose(1,2)
            x = conv(x)
            x = x.transpose(1,2)
            x = residual + x

        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        x, _ = self.slf_attn(x, slf_attn_mask)

        x = self.fc(x)
        h = self.temporal_avg_pool(x, mask)

        ps = self.style_prototypes(sids)
        s_val = self.w * torch.sum(self.V(ps)*h, dim=1) + self.b

        return s_val, cls_loss


class PhonemeDiscriminator(torch.nn.Module):
    ''' Phoneme Discriminator '''
    def __init__(self, input_dim, hidden_dim, max_seq_len):
        super(PhonemeDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.mel_prenet = torch.nn.Sequential(
            LinearNorm(input_dim, hidden_dim, spectral_norm=True),
            torch.nn.LeakyReLU(LEAKY_RELU),
            LinearNorm(hidden_dim, hidden_dim, spectral_norm=True),
            torch.nn.LeakyReLU(LEAKY_RELU),
        )

        n_position = max_seq_len + 1
        self.position_enc = torch.nn.Parameter(
                get_sinusoid_encoding_table(n_position, hidden_dim).unsqueeze(0),
                requires_grad = False)
        
        self.fcs = torch.nn.Sequential(
            LinearNorm(hidden_dim*2, hidden_dim*2, spectral_norm=True),
            torch.nn.LeakyReLU(LEAKY_RELU),
            LinearNorm(hidden_dim*2, hidden_dim*2, spectral_norm=True),
            torch.nn.LeakyReLU(LEAKY_RELU),
            LinearNorm(hidden_dim*2, hidden_dim*2, spectral_norm=True),
            torch.nn.LeakyReLU(LEAKY_RELU),
            LinearNorm(hidden_dim*2, 1, spectral_norm=True)
        )


    def forward(self, mels, srcs, mask):
        batch_size, max_len = mels.shape[0], mels.shape[1]

        mels = self.mel_prenet(mels)
        # print('-------mels shape:',mels.shape)
        # print('-------srcs shape:',srcs.shape)

        if srcs.shape[1] > self.max_seq_len:
            position_embed = get_sinusoid_encoding_table(srcs.shape[1], self.hidden_dim)[:srcs.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(srcs.device)
            srcs = srcs + position_embed
        else:
            position_embed = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
            # print('-------position embed:',position_embed)
            padded_srcs = F.pad(srcs,(0,0,0,position_embed.shape[1]-srcs.shape[1]),'constant',0)
            srcs = padded_srcs + position_embed
        # print('-------padded srcs shape:',padded_srcs.shape)
        # srcs = srcs + position_embed
        

        xs = torch.cat((mels, srcs), dim=-1)
        xs = self.fcs(xs)
        t_val = xs.squeeze(-1)
        mel_len = (~mask).sum(-1)

        # temporal avg pooling
        t_val = t_val.masked_fill(mask, 0.)
        t_val = torch.div(torch.sum(t_val, dim=1), mel_len)
        return t_val
