import torch

original_logit = torch.load('4.44.2_exaone_original_logit.pt')
original_token = torch.load('4.44.2_exaone_original_token.pt')

ported_logit = torch.load('4.31.0_exaone_ported_logit.pt')
ported_token = torch.load('4.31.0_exaone_ported_token.pt')


for ol, ot, pl, pt in zip(original_logit, original_token, ported_logit, ported_token):
    assert ot == pt
    torch.allclose(ol, pl)