from coat import *
from daformer import *
from helper import *

class Net(nn.Module):
    
    def __init__(self,
                 encoder=coat_lite_medium,
                 decoder=daformer_conv3x3,
                 encoder_cfg={},
                 decoder_cfg={},
                 ):
        
        super(Net, self).__init__()
        decoder_dim = decoder_cfg.get('decoder_dim', 320)

        self.encoder = encoder

        self.rgb = RGB()

        encoder_dim = self.encoder.embed_dims
        # [64, 128, 320, 512]

        self.decoder = decoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )
        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
            nn.Upsample(scale_factor = 4, mode='bilinear', align_corners=False),
        )

    def forward(self, batch):

        x = self.rgb(batch)

        B, C, H, W = x.shape
        encoder = self.encoder(x)

        last, decoder = self.decoder(encoder)
        logit = self.logit(last)

        return logit
    
def init_model():
    encoder = coat_lite_medium()
    net = Net(encoder=encoder).cuda()
    return net

class coat_model(nn.Module):
    def __init__(self):
        super(coat_model, self).__init__()
        self.model = init_model()
        
    def forward(self, images):
        out= self.model(images)
        return out
