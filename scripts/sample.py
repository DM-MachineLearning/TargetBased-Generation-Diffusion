
import torch, argparse
from src.models.denoiser import Denoiser
ap=argparse.ArgumentParser(); ap.add_argument('--ckpt'); ap.add_argument('--out')
args=ap.parse_args()
na=16; ka=8
model=Denoiser(ka,128)
model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
model.eval()
x=torch.randn(1,na,3); a=torch.randint(0,ka,(1,na))
with torch.no_grad():
    eps,logits=model(x,a)
    x=x-eps; a=logits.argmax(-1)
torch.save(dict(x=x,a=a),args.out)
print('saved',args.out)
