import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





class fbsde():
    def __init__(self, x_0, b, sigma, f, g, lower, upper, T, dim_x, dim_y, dim_d):
        self.x_0 = x_0
        self.b = b
        self.sigma = sigma
        self.f = f
        self.g = g
        self.lower = lower #lower
        self.upper=upper #upper
        self.T = T
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_d = dim_d




class Model(nn.Module):
    def __init__(self, equation, dim_h):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(equation.dim_x + 1, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_h)
        self.linear3 = nn.Linear(dim_h, dim_h)
        self.linear4 = nn.Linear(dim_h, equation.dim_y + equation.dim_y * equation.dim_d)
        self.bn1 = nn.BatchNorm1d(dim_h)
        self.bn2 = nn.BatchNorm1d(dim_h)
        self.bn3 = nn.BatchNorm1d(dim_h)


        self.equation = equation

    def forward(self, N, n, x):

        def normalize(x):
            xmax = x.max(dim=0).values
            xmin = x.min(dim=0).values
            return (x-xmin)/(xmax-xmin)

        def standardize(x):
            mean = torch.mean(x,dim=0)
            sd = torch.std(x,dim=0)
            return (x-mean)/(sd + 0.0001)

        def phi(x):
            x = torch.tanh(self.linear1(x))
            x = torch.tanh(self.linear2(x))
            x = torch.tanh(self.linear3(x))
            return self.linear4(x) #[bs,(dy*dd)] -> [bs,dy,dd]



        delta_t = self.equation.T / N

        x_nor = standardize(x)
        #x_nor = x
        '''
        if n!=0:
            #x_nor = normalize(x)
            x_nor = standardize(x)
            '''

        inpt = torch.cat((x_nor, torch.ones(x.size()[0], 1, device=device) * delta_t * n), 1)
        yz = phi(inpt)
        y = yz[:,:self.equation.dim_y].clone()
        z = yz[:,self.equation.dim_y:self.equation.dim_y + self.equation.dim_y * self.equation.dim_d].reshape(-1,self.equation.dim_y, self.equation.dim_d).clone()
        return y,z



class BSDEsolver():
    def __init__(self, equation, dim_h, model,lr,coeff):
        self.model = model
        self.equation = equation
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr*coeff)
        self.dim_h = dim_h

    def loss(self, x, n, y_prev, y, z, w, N):
        #if n == N-2:
        #    dist = (y - self.equation.g(x)).norm(2,dim=1)
        #else:
        delta_t = self.equation.T / N
        estimate = y - self.equation.f(delta_t*n, x , y, z)*delta_t + torch.matmul(z, w).reshape(-1, self.equation.dim_y)

        dist = (y_prev - estimate).norm(2,dim=1)**2
        return torch.mean(dist)



    def gen_forward(self, batch_size, N,n):
        delta_t = self.equation.T / N
        x = self.equation.x_0 + torch.zeros(batch_size, self.equation.dim_x, requires_grad=True, device=device).reshape(
            -1, self.equation.dim_x)  # [bs,dx]
        if n==0:
            w = torch.randn(batch_size, self.equation.dim_d, 1, device=device)*np.sqrt(delta_t)
            x_next = x + (self.equation.b(delta_t*0, x)) * delta_t + torch.matmul(self.equation.sigma(delta_t*0, x),
                                                                                  w).reshape(-1, self.equation.dim_x)
            #x = torch.exp(x)
            #x_next = torch.exp(x_next)
        else:
            for i in range(n):
                w = torch.randn(batch_size, self.equation.dim_x, 1, device=device)*np.sqrt(delta_t)
                x = x + (self.equation.b(delta_t * (i), x)) * delta_t + torch.matmul(self.equation.sigma(delta_t * (i), x),w).reshape(-1, self.equation.dim_x)
            w = torch.randn(batch_size, self.equation.dim_d, 1, device=device)*np.sqrt(delta_t)
            x_next = x + (self.equation.b(delta_t * (n), x)) * delta_t + torch.matmul(self.equation.sigma(delta_t * (n), x),w).reshape(-1, self.equation.dim_x)
            #x = torch.exp(x)
            #x_next = torch.exp(x_next)
        return x, w, x_next

    def train(self, batch_size, N, n, itr, path, multiplyer):
        loss_n = []
        delta_t = self.equation.T / N

        if n != N-2:
            mod2 = Model(self.equation, self.dim_h).to(device)
            mod2.load_state_dict(torch.load(path + "state_dict_" + str(n + 1)), strict=False)
            mod2.eval()

        if n >= N-3:
            itr_actual = multiplyer*itr
        else:
            itr_actual = itr

        for i in range(itr_actual):
            if i % 200 == 0:
                print("itr=" + str(i))
            flag = True
            while flag:
                x, w, x_next = self.gen_forward(batch_size,N, n)
                flag = torch.isnan(x_next).any()




            y,z = self.model(N, n, x)

            if n == N-2:
                y_prev = self.equation.g(x_next).to(device)
            else:

                y_prev, z_prev = mod2(N,n+1,x_next)


            if 0==0:
                #y_prev = torch.maximum(y_prev, self.equation.l(x_next))
                y_prev = torch.minimum(torch.maximum(y_prev, self.equation.lower(delta_t*n, x_next)), self.equation.upper(delta_t*n, x_next))



            loss = self.loss(x,n,y_prev,y,z,w,N)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            loss_n.append(float(loss))

            #if i%(itr_actual-1) == 0:
                #print("time_"+str(n)+ "iter_"+str(i))
                #for par_group in self.optimizer.param_groups:
                    #print(par_group["lr"])


        return loss_n, y

class BSDEiter():
    def __init__(self, equation, dim_h):
        self.equation = equation
        self.dim_h = dim_h


    def train_whole(self, batch_size, N, path, itr, multiplyer):
        loss_data = []


        for n in range(N-2,-1,-1):
            lr = 0.001
            coeff = 1
            if n==N-2:
                coeff = 1
            print("time "+ str(n))
            mod = Model(self.equation, self.dim_h).to(device)
            bsde_solver = BSDEsolver(self.equation, self.dim_h, mod, lr, coeff)
            if n != N-2:
                #break
                bsde_solver.model.load_state_dict(torch.load(path+"state_dict_" + str(n+1)), strict=False)
                bsde_solver.optimizer.load_state_dict(torch.load(path + "state_dict_opt_" + str(n + 1)))

            loss_n, y = bsde_solver.train(batch_size, N, n, itr, path, multiplyer)
            loss_data.append(loss_n)
            torch.save(bsde_solver.model.state_dict(),path+"state_dict_" + str(n))
            torch.save(bsde_solver.optimizer.state_dict(), path + "state_dict_opt_" + str(n))

        return loss_data, y














class Result():
    def __init__(self,model, equation):
        self.model = model
        self.equation = equation

    def gen_b_motion(self, batch_size, N):
        delta_t = self.equation.T / N
        W = torch.randn(batch_size, self.equation.dim_d, N, device=device) * np.sqrt(delta_t)

        return W



    def gen_x(self, batch_size, N, W):
        delta_t = self.equation.T / N
        #x = self.equation.x_0 + torch.zeros(batch_size, N * self.equation.dim_x, device=device).reshape(-1,self.equation.dim_x, N) #[bs,dx,N]
        x = torch.zeros(batch_size, N, self.equation.dim_x, device=device).reshape(-1, self.equation.dim_x,N)  # [bs,dx,N]
        x[:, :, 0] = self.equation.x_0.view(1, self.equation.dim_x).expand(x.size(0), -1)

        for i in range(N-1):
            w = W[:, :, i].reshape(-1, self.equation.dim_d, 1)
            x[:,:,i+1] = x[:,:,i] + self.equation.b(delta_t * i, x[:,:,i]) * delta_t + torch.matmul(self.equation.sigma(delta_t * i, x[:,:,i]),w).reshape(-1, self.equation.dim_x)
        #return torch.exp(x)
        return x


    def predict(self,N,batch_size,x, path):
        delta_t = self.equation.T / N
        ys = torch.zeros(batch_size, self.equation.dim_y, N)
        zs = torch.zeros(batch_size, self.equation.dim_y, self.equation.dim_d, N)



        for n in range(N-1):
            self.model.load_state_dict(torch.load(path + "state_dict_" + str(n),map_location=torch.device('cpu')), strict=False)
            y,z = self.model(N, n, x[:,:,n])
            if 0==0:
                #y = torch.maximum(y,self.equation.l(x[:,:,n]))
                y = torch.minimum(torch.maximum(y, self.equation.lower(delta_t*n, x[:, :, n])), self.equation.upper(delta_t*n, x[:, :, n]))


            ys[:,:,n] = y
            zs[:,:,:,n] = z
        ys[:,:,N-1] = self.equation.g(x[:,:,N-1])
        return ys, zs

    def regenerate(self, N, x, W, y, z):
        delta_t = self.equation.T/N
        y_g = y
        for n in range(N-1):
            w = W[:, :, n].reshape(-1, self.equation.dim_d, 1)
            y_g[:,:,n+1] = y[:,:,n] - self.equation.f(delta_t*n, x[:,:,n] ,y[:,:,n], z[:,:,:,n])*delta_t + torch.matmul(z[:,:,:,n], w).reshape(-1, self.equation.dim_y)

        return y_g

    def L2(self,true,est,N):
        dt = self.equation.T/N
        diff = torch.mean(torch.sum(torch.linalg.norm((true-est)**2,dim=1)*dt, dim=-1),dim=0)
        l2_true = torch.mean(torch.sum(torch.linalg.norm((true)**2,dim=1)*dt, dim=-1),dim=0)
        return float(torch.sqrt(diff/l2_true))
