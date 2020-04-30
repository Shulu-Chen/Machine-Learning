import torch

##############################
## Specify the matrix X
## Dimensions: X (3x3)
##############################
X = torch.Tensor([[0,0,1],[1,1,1],[4,2,1]])
y = torch.Tensor([[0],[1],[1]])
print(X)
print(y)

# Solution
##############################
## Use one of the ways to compute the result
##############################
l =torch.matmul(torch.transpose(X, 0, 1),X)
r =torch.matmul(torch.transpose(X, 0, 1),y)
res1 = torch.gesv(r,l)
print(res1[0])