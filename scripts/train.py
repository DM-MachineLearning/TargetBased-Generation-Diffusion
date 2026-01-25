
import yaml, torch, os
from src.utils.seed import set_seed
from src.data.unified import get_dataset
from src.diffusion.schedules import linear_beta
from src.diffusion.continuous import ContinuousDiffusion
from src.diffusion.categorical import QCat
from src.models.denoiser import Denoiser
import argparse

ap=argparse.ArgumentParser(); ap.add_argument('--config')
cfg=yaml.safe_load(open(ap.parse_args().config))

set_seed(cfg['seed'])
device=cfg['device']

ds, collate_fn = get_dataset(cfg)
dl=torch.utils.data.DataLoader(ds,batch_size=cfg['train']['batch_size'],shuffle=True,collate_fn=collate_fn)

beta=linear_beta(cfg['diffusion']['T'],cfg['diffusion']['beta_start'],cfg['diffusion']['beta_end'],device)
cd=ContinuousDiffusion(cfg['diffusion']['T'],beta)
qc=QCat(cfg['model']['atom_types'],cfg['diffusion']['T'],beta)

model=Denoiser(cfg['model']['atom_types'],cfg['model']['hidden']).to(device)
opt=torch.optim.Adam(model.parameters(),lr=cfg['train']['lr'])

os.makedirs('runs/'+cfg['run_name'],exist_ok=True)

for ep in range(cfg['train']['epochs']):
    for x,a,m in dl:
        x,a,m = x.to(device), a.to(device), m.to(device)
        B=x.size(0)
        t=torch.randint(0,cfg['diffusion']['T'],(B,),device=device)
        noise=torch.randn_like(x)
        xt=cd.q_sample(x,t,noise,m)
        at=qc.q_sample(a,t[0].item())
        eps_hat,logits=model(xt,at)
        loss=((eps_hat-noise)**2*(m[:,:,None])).mean() +              torch.nn.functional.cross_entropy(logits.view(-1,logits.size(-1)),a.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    print('epoch',ep,'loss',float(loss))

torch.save(model.state_dict(),f"runs/{cfg['run_name']}/ckpt.pt")
