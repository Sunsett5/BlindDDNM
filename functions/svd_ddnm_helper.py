import torch 

@ torch.no_grad()
def cal_x0(xt, t, at, at_next, et, y_0, A_funcs, sigma_y, eta, learned=False, classes=None):
    
    """ if self.sigma_y == 0:
        if self.cls_fn == None:
            et = self.model(xt, t)
        else:
            et = self.model(xt, t, classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0,0,0,0] * self.cls_fn(x,t,classes)
    
        if et.size(1) == 6:
            et = et[:, :3]
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_t = x0_t.clip(-1, 1)
        x0_t = x0_t + self.A_funcs.A_pinv(y_0 - self.A_funcs.A(x0_t)).view(y_0.shape[0], 3, x0_t.shape[2], x0_t.shape[3])
        add_up = self.eta * (1-at_next).sqrt() * torch.randn_like(x0_t) + (1-self.eta**2)**0.5 * (1-at_next).sqrt() * et
    else: """
    x = torch.randn_like(xt)
    singulars = A_funcs.singulars()
    Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
    Sigma[:singulars.shape[0]] = singulars
    Inv_Sigma = 1 / Sigma
    Inv_Sigma[Sigma==0] = 0
    U_t_y = A_funcs.Ut(y_0)
    Sigma = Sigma.view([1, x.shape[1], x.shape[2], x.shape[3]]).repeat(x.shape[0], 1, 1, 1)
    Inv_Sigma = Inv_Sigma.view([1, x.shape[1], x.shape[2], x.shape[3]]).repeat(x.shape[0], 1, 1, 1)
    
    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
    x0_t = x0_t.clip(-1, 1)

    V_t_et = A_funcs.Vt(et).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
    V_t_x0_t = A_funcs.Vt(x0_t).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
    
    lambda_t = torch.ones_like(V_t_x0_t)
    sigma_t = (1 - at_next[0,0,0,0]) ** 0.5
    change_idx = 1.0 * (sigma_t < at_next[0,0,0,0].sqrt()*sigma_y*Inv_Sigma)
    lambda_t = (1-change_idx) * lambda_t + change_idx * Sigma * sigma_t * (1-eta**2)**0.5/at_next[0,0,0,0].sqrt()/sigma_y
    random_noise = torch.randn_like(V_t_x0_t)
    epsilon_tmp = torch.zeros_like(V_t_x0_t)
    change_idx = 1.0 * (sigma_t >= at_next[0,0,0,0].sqrt()*sigma_y*Inv_Sigma)
    epsilon_tmp = (1-change_idx) * epsilon_tmp + change_idx * (sigma_t**2-at_next[0,0,0,0]*sigma_y**2*Inv_Sigma**2) * random_noise
    change_idx = 1.0 * (sigma_t < at_next[0,0,0,0].sqrt()*sigma_y*Inv_Sigma)
    epsilon_tmp = (1-change_idx) * epsilon_tmp + change_idx * eta * sigma_t * random_noise
    change_idx = 1.0 * (Sigma==0)
    epsilon_tmp = (1-change_idx) * epsilon_tmp + change_idx * (sigma_t * (1-eta**2)**0.5 * V_t_et + sigma_t * eta * random_noise)

    x0_t = x0_t - A_funcs.V(
        (lambda_t * A_funcs.Vt(
            A_funcs.A_pinv(
                A_funcs.A(x0_t) - y_0
                ).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
                ).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
                ).view(x.shape[0], -1)
                ).view([x.shape[0], x.shape[1], x.shape[2], x.shape[3]])
    

    # if learned:
    #     add_up = eta * (1-at_next).sqrt() * torch.randn_like(x0_t) + (1-eta**2)**0.5 * (1-at_next).sqrt() * et
    # else:

    add_up = A_funcs.V(epsilon_tmp.view([epsilon_tmp.shape[0], -1])).view(x.shape)
    return x0_t, add_up

@torch.no_grad()
def map_back(x0_t, y_0, add_up, at_next, at):
    xt_next = at_next.sqrt() * x0_t + add_up
    return xt_next