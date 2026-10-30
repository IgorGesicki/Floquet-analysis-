import torch 

A = torch.tensor([0,1,2], dtype=torch.float32)
Q = torch.tensor([0.5,1.0], dtype=torch.float32)

AA, QQ = torch.meshgrid(A,Q, indexing='ij')

Omega_sq= AA-2*QQ

print(f'PyTorch version is {torch.__version__}')
print(f'AA shape is {AA.size()}')
print(f'QQ shape is {QQ.size()}')
print(f'Omega^2 shape is {Omega_sq.size()}')
print(Omega_sq)


