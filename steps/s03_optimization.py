import torch
import matplotlib.pyplot as plt


z = torch.linspace(0, torch.pi,80)
y = 1.7*torch.cos(z)
L = []
N = []


a=torch.nn.Parameter(torch.tensor(0.2))

def MSE(a):
    return torch.sum((a*torch.cos(z)-y)**2)/z.size()[0]

optimizer = torch.optim.Adam([a], lr=0.09)

for epoch in range(0,150):
    optimizer.zero_grad()
    loss = MSE(a)
    loss.backward()
    L.append(loss.item())
    N.append(epoch)
    if (epoch%10==0):
        print(loss)
    optimizer.step()
    
    


        
print(f'a after training is: {a.item()}')
plt.plot(N,L)
plt.show()