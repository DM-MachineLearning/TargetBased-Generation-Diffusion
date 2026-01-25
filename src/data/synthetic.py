
import torch
class Synthetic(torch.utils.data.Dataset):
    def __init__(self,n,na,np,ka):
        self.n=n; self.na=na; self.np=np; self.ka=ka
    def __len__(self): return self.n
    def __getitem__(self,i):
        x=torch.randn(self.na,3)
        a=torch.randint(0,self.ka,(self.na,))
        mask=torch.ones(self.na); mask[:self.na//3]=0
        return dict(x=x,a=a,mask=mask)
