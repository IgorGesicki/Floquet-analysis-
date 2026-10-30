import torch

z = torch.tensor([[0.3],[1.0],[2.0]], dtype=torch.float64, requires_grad=True)

X = torch.sin(z)

X_z = torch.autograd.grad(X,z, grad_outputs=torch.ones_like(z),create_graph=True)[0]
X_zz = torch.autograd.grad(X_z, z, grad_outputs=torch.ones_like(z), create_graph=True)[0]

# print(X_z)
# print(X_zz)

Max_err = torch.max(torch.abs((X_z-torch.cos(z))))

ressid = torch.max(torch.abs(X_zz+X))

Is_correct = torch.allclose(X, -X_zz)

print(f'Max. error is: {Max_err}\n')
print(f'Residuum X\'\'-X: {ressid}\n')
print(f'Test allclose: {Is_correct}')