import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

    
class A_functions:
    """
    A class replacing the SVD of a matrix A, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()
    
    def A(self, vec):
        """
        Multiplies the input vector by A
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, :singulars.shape[0]])
    
    def At(self, vec):
        """
        Multiplies the input vector by A transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[0]]))
    
    def A_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of A
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        
        factors = 1. / singulars
        factors[singulars == 0] = 0.
        
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def A_pinv_eta(self, vec, eta):
        """
        Multiplies the input vector by the pseudo inverse of A with factor eta
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        factors = singulars / (singulars*singulars+eta)
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        raise NotImplementedError()

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        raise NotImplementedError()
        
class DeblurringArbitral2D(A_functions):

    def __init__(self, kernel, channels, img_dim, device, conv_shape='same'):
        self.img_dim   = img_dim
        self.channels  = channels
        self.conv_shape = conv_shape
        _nextpow2 = lambda x : int(np.power(2, np.ceil(np.log2(x))))
        self.fft2_size = _nextpow2(img_dim + kernel.shape[0] - 1) # next pow 2
        self.kernel_size = (kernel.shape[-2], kernel.shape[-1])
        self.kernel = kernel
        self.init_kernel = kernel.detach().clone()
        self.update_kernel(kernel)
        self.device = device

        if conv_shape == 'same':
            self.out_img_dim = img_dim
        elif conv_shape == 'full':
            # TODO: rectangular kernel size
            self.out_img_dim = img_dim + (self.kernel_size[0] - 1)
        elif conv_shape == "same_interp":
            self.out_img_dim = img_dim
    
    def A(self, vec):
        """
        Multiplies the input vector by A
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        ret = self.U(singulars * temp[:, :singulars.shape[1]])

        return ret

    def At(self, vec):
        """
        Multiplies the input vector by A transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, :singulars.shape[1]]))
    
    def A_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of A
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        
        factors = 1. / singulars
        factors[singulars == 0] = 0.
        
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def A_pinv_eta(self, vec, eta):
        """
        Multiplies the input vector by the pseudo inverse of A with factor eta
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        factors = singulars / (singulars*singulars+eta)
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))

    def V(self, vec):
        
        vec = vec.reshape(vec.shape[0], self.channels, -1)
        vec = vec / self._singular_phases[:, None, :]
        vec = vec.reshape(vec.shape[0], self.channels, -1)

        vec = self._batch_inv_perm(vec, self._perm)

        vec_ifft = torch.fft.ifft2(vec.reshape(vec.shape[0], self.channels, self.fft2_size, self.fft2_size),\
            norm="ortho").real


        out = vec_ifft[:, :, :self.img_dim, :self.img_dim].reshape(vec.shape[0], -1)

        return out

    def Vt(self, vec):
        
        vec_fft = torch.fft.fft2(vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim), (self.fft2_size, self.fft2_size), norm="ortho")

        vec_fft = self._batch_perm(vec_fft.reshape(vec.shape[0], self.channels, -1), self._perm)
        vec_fft = vec_fft * self._singular_phases[:, None, :]

        return vec_fft.reshape(vec.shape[0], -1)

    def U(self, vec):

        vec = vec.reshape(vec.shape[0], self.channels, -1)
        vec = self._batch_inv_perm(vec, self._perm)


        vec_ifft = torch.fft.ifft2(vec.reshape(vec.shape[0], self.channels, self.fft2_size, self.fft2_size),\
            norm="ortho").real

        if self.conv_shape == 'same':        
            out = vec_ifft[:, :, self.kernel_size[0]//2:int(self.kernel_size[0]//2+self.img_dim), \
                self.kernel_size[1]//2:int(self.kernel_size[1]//2+self.img_dim)]
        elif self.conv_shape == 'full':
            out = vec_ifft[:, :, :self.out_img_dim, :self.out_img_dim]
        else: # elif self.conv_shape == "same_interp":
            out = vec_ifft[:, :, self.kernel_size[0]//2:int(self.kernel_size[0]//2+self.img_dim), \
                self.kernel_size[1]//2:int(self.kernel_size[1]//2+self.img_dim)]

        return out

    def Ut(self, vec):
        
        _ks0 = self.kernel_size[0]
        _ks1 = self.kernel_size[1]
        _Nf  =  self.fft2_size

        if self.conv_shape == 'same':
            exec_zeropad = torch.nn.ZeroPad2d((_ks0//2, _Nf-_ks0//2-self.img_dim,\
                _ks1//2, _Nf-_ks1//2-self.img_dim))
            
            vec = exec_zeropad(vec.reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim))
        elif self.conv_shape == 'full':
            vec = vec.reshape(vec.shape[0], self.channels, self.out_img_dim, self.out_img_dim)
            exec_zeropad = torch.nn.ZeroPad2d((0, _Nf-self.out_img_dim, 0, _Nf-self.out_img_dim))
            vec = exec_zeropad(vec)

        elif self.conv_shape == "same_interp":
            pass

        vec_fft = torch.fft.fft2(vec, (self.fft2_size, self.fft2_size), norm="ortho")

        vec_fft = self._batch_perm(vec_fft.reshape(vec.shape[0], self.channels, -1), self._perm)

        return vec_fft.reshape(vec.shape[0], -1)

    def singulars(self):
        
        bsz = self._singulars.shape[0]
        return self._singulars.repeat(1, 3).reshape(bsz, -1)

    def add_zeros(self, vec):
        tmp = torch.zeros(vec.shape[0], self.channels, self.fft2_size**2, device=vec.device, dtype=vec.dtype)
        reshaped = vec.clone().reshape(vec.shape[0], self.channels, -1)
        tmp[:, :, :reshaped.shape[2]] = reshaped

        return tmp.reshape(vec.shape[0], -1)
    
    def update_kernel(self, kernel):
        """
        Update the internal kernel and associated variables using the provided kernel tensor.

        Args:
            kernel (torch.Tensor): The kernel tensor for the update. It should have the same shape of self.kernel

        Returns:
            None
        """

        self.kernel = kernel
        self.k_fft = torch.fft.fft2(kernel, (self.fft2_size, self.fft2_size), norm="ortho")

        bsz = kernel.shape[0]
        _eps_singulars = 1e-10 * torch.randn_like(self.k_fft)
        self._singular_phases = ((self.k_fft + _eps_singulars) / torch.abs(self.k_fft + _eps_singulars)).reshape(bsz, -1)
        self._singulars = torch.abs(self.k_fft * self.fft2_size).reshape(bsz, -1)
        ZERO = 0.05
        self._singulars[self._singulars < ZERO] = 0.0
        self._singulars, self._perm = self._singulars.sort(descending=True)
        self._singular_phases = self._batch_perm(self._singular_phases.reshape(bsz, -1), self._perm)
    
    def _batch_perm(self, tensor, perm):

        bsz = tensor.shape[0]
        for i_bsz in range(bsz):
            if tensor.dim() == 2:
                tensor[i_bsz, :] = tensor[i_bsz, perm[i_bsz]]
            elif tensor.dim() == 3:
                tensor[i_bsz, :, :] = tensor[i_bsz, :, perm[i_bsz]]

        return tensor

    def _batch_inv_perm(self, tensor, perm):

        bsz = tensor.shape[0]
        for i_bsz in range(bsz):
            if tensor.dim() == 2:
                tensor[i_bsz, perm[i_bsz]] = tensor[i_bsz, :].clone()
            elif tensor.dim() == 3:
                tensor[i_bsz, :, perm[i_bsz]] = tensor[i_bsz, :, :].clone()

        return tensor

    def update_A_optim(self, y, x, n_iter=1000, lr = 1e-3, reg_A_gamma=0.0, reg_A_type="norm"):
        """
            update the kernel parameter so that it satisfies y = H(k)x
                using Adam optimizer
            The optimizer minimizes the following objective:
            if reg_A_type == "norm":
                ||y - H(x)||_2 + reg_A_gamma * ||k||_1
            else reg_A_type == "diff_norm":
                ||y - H(x)||_2 + reg_A_gamma * ||k - k_init||_2
            
            Args:
                y : (#batch, #channel, self.out_img_dim, self.out_img_dim) blurry image
                x : (#batch, #channel, self.img_dim, self.img_dim) (estimated) clean image
                n_iter : (int) the optimizer iterations
                lr : (float) leraning rate for the optimizer
                reg_A_gamma : weight of regularization
                reg_A_type : ["norm", "diff_norm"]  l1 norm or norm of difference between kernel and init_kernel        
        """

        with torch.set_grad_enabled(True):
            for i_batch in range(x.shape[0]):
                self.kernel.requires_grad_()

                params = [{'params': self.kernel}]
                optimizer = optim.Adam(params, lr=lr)

                x = x.to(y.device)

                for i in range(n_iter):
                    
                    optimizer.zero_grad()
                    
                    y_est = self.A_fftconv(x[i_batch][None, :, :, :], self.kernel[i_batch][None, :, :])

                    if reg_A_type == "norm":
                        loss = torch.norm(y[i_batch] - y_est) + reg_A_gamma * torch.sum(torch.abs(self.kernel[i_batch])) # L1 Regularization
                    else: # elif reg_A_type == "diff_norm":
                        loss = torch.norm(y[i_batch] - y_est) + reg_A_gamma * torch.norm(self.kernel[i_batch] - self.init_kernel[i_batch])

                    if i % 100 == 0:
                        print(f"loss : {loss.item()}")
                    
                    loss.backward()
                    optimizer.step()

            self.kernel.requires_grad_(False)
            self.kernel = self.kernel / self.kernel.sum(dim=(-2, -1), keepdim=True)
            self.update_kernel(self.kernel)
            return None

    def update_A_langevin(self, y, x, n_iter=1000, lr = 1e-3, reg_A_gamma=0.0, reg_A_type="norm"):
        """
        Update the kernel parameter using Langevin dynamics.

        Args:
            y (torch.Tensor): Blurry image tensor of shape (#batch, #channel, self.out_img_dim, self.out_img_dim).
            x (torch.Tensor): (Estimated) clean image tensor of shape (#batch, #channel, self.img_dim, self.img_dim).
            n_iter (int): Number of optimizer iterations.
            lr (float): Learning rate for the optimizer.
            reg_A_gamma (float): Weight of regularization.
            reg_A_type (str): Type of regularization. Can be either "norm" or "diff_norm".

        Returns:
            None
        """

        with torch.set_grad_enabled(True):
            self.kernel.requires_grad_()

            x = x.to(y.device)

            for i in range(n_iter):
                                    
                y_est = self.A_fftconv(x, self.kernel)

                if reg_A_type == "norm":
                    loss = torch.norm(y - y_est)**2 / (2*0.04**2) + reg_A_gamma * torch.sum(torch.abs(self.kernel)) # L1 Regularization
                else: # elif reg_A_type == "diff_norm":
                    loss = torch.norm(y - y_est)**2 + reg_A_gamma * torch.norm(self.kernel - self.init_kernel)
                
                loss.backward()
                with torch.no_grad():
                    self.kernel.add_(self.kernel.grad, alpha=-lr/2)
                    self.kernel.add_(torch.randn_like(self.kernel), alpha=np.sqrt(lr))
                self.kernel.grad.zero_()

            self.kernel.requires_grad_(False)
            self.kernel = self.kernel / self.kernel.sum(dim=(-2, -1), keepdim=True)
            self.update_kernel(self.kernel)
            return None

    def reset_Hupdate(self):

        self.update_kernel(self.init_kernel.detach().clone())

    def A_fftconv(self, x, kernel):

        x_fft = torch.fft.fft2(x.reshape(x.shape[0], self.channels, self.img_dim, self.img_dim), (self.fft2_size, self.fft2_size), norm="ortho")

        k_fft = torch.fft.fft2(kernel, (self.fft2_size, self.fft2_size), norm="ortho")[:, None, :, :]

        y_fft = k_fft * x_fft

        y_fftconv = torch.fft.ifft2(y_fft, norm="ortho").real * self.fft2_size

        if self.conv_shape == "same" or self.conv_shape == "same_interp":
            shifts = (self.kernel_size[0]//2, self.kernel_size[1]//2)
            y_fftconv_clip = y_fftconv[:, :, shifts[0]:int(shifts[0]+self.img_dim), \
                shifts[1]:int(shifts[1]+self.img_dim)]
        else: # self.conv_type == "full"
            y_fftconv_clip = y_fftconv[:, :, :self.img_dim+(self.kernel_size[0]-1), :self.img_dim+(self.kernel_size[1]-1)]
        
        return y_fftconv_clip

    def interp_y_0(self, y_0, x_0, sigma_0):

        x_fft = torch.fft.fft2(x_0.reshape(x_0.shape[0], self.channels, self.img_dim, self.img_dim), (self.fft2_size, self.fft2_size), norm="ortho")

        k_fft = torch.fft.fft2(self.kernel, (self.fft2_size, self.fft2_size), norm="ortho")[:, None, :, :]

        y_fft = k_fft * x_fft
        y_fftconv = torch.fft.ifft2(y_fft, norm="ortho").real * self.fft2_size

        shifts = (self.kernel_size[0]//2, self.kernel_size[1]//2)

        y_fftconv += sigma_0 * torch.randn_like(y_fftconv)
        y_fftconv[:, :, shifts[0]:int(shifts[0]+self.img_dim), \
                shifts[1]:int(shifts[1]+self.img_dim)] = y_0
        
        return y_fftconv
    
    @staticmethod
    def get_blur_kernel_batch(batch_size, kernel_type, device):
        """
        Generates a batch of blur kernels of the specified type.

        Args:
            batch_size (int): The number of blur kernels to generate.
            kernel_type (str): The type of blur kernel to generate. Can be one of "gauss", "motionblur", "from_png", or "uniform".
            device (torch.device): The device on which to generate the blur kernels.

        Returns:
            torch.Tensor: A batch of blur kernels of shape (batch_size, kernel_size, kernel_size).
        """

        if kernel_type == "gauss":
            sigma = 5
            pdf = lambda x : torch.exp(torch.Tensor([-0.5 * (x / sigma)]))
            kernel_size = 9 # must be odd
            kernel = torch.zeros((kernel_size, kernel_size)).to(device)
            for i in range(-(kernel_size//2), kernel_size//2+1):
                for j in range(-(kernel_size//2), kernel_size//2+1):
                    kernel[i+kernel_size//2, j+kernel_size//2] = pdf(torch.sqrt(torch.Tensor([i**2+j**2])))
                # zeropad_fun = torch.nn.ZeroPad2d((10, 10, 10, 10))
                # kernel = zeropad_fun(kernel)
            kernel = kernel / kernel.sum()
            kernel_batch = kernel.repeat(batch_size, 1, 1)

        elif kernel_type == "motionblur":
            kernel_size = 64
            kernel_batch = torch.zeros(batch_size, kernel_size, kernel_size, device=device)
            for i_batch in range(batch_size):
                kernel = (Kernel(size=(kernel_size, kernel_size), intensity=0.50).kernelMatrix)
                kernel = torch.from_numpy(kernel).clone().to(device)
                kernel = kernel / kernel.sum()
                kernel_batch[i_batch] = kernel
        else: # config.deblur.kernel_type == "uniform":
            kernel_size = 31
            kernel = torch.ones((kernel_size, kernel_size)).to(device)
            kernel = kernel / kernel.sum()
            kernel_batch = kernel.repeat(batch_size, 1, 1)
            if kernel_type != "uniform":
                print("please specify the kernel type from [gauss, mnist, uniform, motionblur]. uniform kernel is used.")

        return kernel_batch

    @staticmethod
    def corrupt_kernel_batch(kernel_batch, kernel_corruption, kernel_corruption_coef=None):
        """
        Adds corruption to a batch of blur kernels.

        Args:
            kernel_batch (torch.Tensor): A batch of blur kernels of shape (batch_size, kernel_size, kernel_size).
            kernel_corruption (str): The type of corruption to add. Can be one of "additive", "multiplicative", "random_init", or "gauss_init".
            kernel_corruption_coef (float, optional): The coefficient of corruption. Defaults to None.

        Returns:
            torch.Tensor: A batch of corrupted blur kernels of shape (batch_size, kernel_size, kernel_size).
        """
        if kernel_corruption == "additive":                
            kernel_uncert_batch = kernel_batch + torch.abs(kernel_corruption_coef * torch.randn_like(kernel_batch))

        elif kernel_corruption == "multiplicative":
            kernel_uncert_batch = kernel_batch + kernel_corruption_coef * torch.randn_like(kernel_batch) * kernel_batch

        elif kernel_corruption == "random_init":
            kernel_uncert_batch = torch.rand_like(kernel_batch)

        elif kernel_corruption == "gauss_init":

            _batch_size = kernel_batch.shape[0]
            _kernel_size = kernel_batch.shape[1]

            sigma = 5
            pdf = lambda x : torch.exp(torch.Tensor([-0.5 * (x / sigma)]))
            kernel = torch.zeros((_kernel_size, _kernel_size)).to(kernel_batch.device)
            for i in range(-_kernel_size//2+1, _kernel_size//2+1):
                for j in range(-_kernel_size//2+1, _kernel_size//2+1):
                    kernel[i+_kernel_size//2-1, j+_kernel_size//2-1] = pdf(torch.sqrt(torch.Tensor([i**2+j**2])))
            kernel = kernel / kernel.sum()
            kernel_uncert_batch = kernel.repeat(_batch_size, 1, 1)

        else:
            kernel_uncert_batch = kernel_batch
        
        # Normalization
        kernel_uncert_batch = kernel_uncert_batch / kernel_uncert_batch.sum(dim=(-2, -1), keepdim=True)
        return kernel_uncert_batch

# block-wise CS
class CS(A_functions):
    def __init__(self, channels, img_dim, ratio, device): #ratio = 2 or 4
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // 32
        self.ratio = 32
        A = torch.randn(32**2, 32**2).to(device)
        _, _, self.V_small = torch.svd(A, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)
        self.singulars_small = torch.ones(int(32 * 32 * ratio), device=device)
        self.cs_size = self.singulars_small.size(0)

    def V(self, vec):
        #reorder the vector back into patches (because singulars are ordered descendingly)

        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.size(0), self.channels * self.y_dim ** 2, self.ratio ** 2, device=vec.device)
        patches[:, :, :self.cs_size] = temp[:, :self.channels * self.y_dim ** 2 * self.cs_size].contiguous().reshape(
            vec.size(0), -1, self.cs_size)
        patches[:, :, self.cs_size:] = temp[:, self.channels * self.y_dim ** 2 * self.cs_size:].contiguous().reshape(
            vec.size(0), self.channels * self.y_dim ** 2, -1)

        #multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        #extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, :self.channels * self.y_dim ** 2 * self.cs_size] = patches[:, :, :, :self.cs_size].contiguous().reshape(
            vec.shape[0], -1)
        recon[:, self.channels * self.y_dim ** 2 * self.cs_size:] = patches[:, :, :, self.cs_size:].contiguous().reshape(
            vec.shape[0], -1)
        return recon

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp


def color2gray(x):
    x = x[:, 0:1, :, :] * 0.3333 + x[:, 1:2, :, :] * 0.3334 + x[:, 2:, :, :] * 0.3333
    return x


def gray2color(x):
    base = 0.3333 ** 2 + 0.3334 ** 2 + 0.3333 ** 2
    return torch.stack((x * 0.3333 / base, x * 0.3334 / base, x * 0.3333 / base), 1)
    
    
#a memory inefficient implementation for any general degradation A
class GeneralA(A_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2: vshape = vshape * v.shape[2]
        if len(v.shape) > 3: vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape,
                        1)).view(v.shape[0], M.shape[0])

    def __init__(self, A):
        self._U, self._singulars, self._V = torch.svd(A, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0
        print(len([x.item() for x in self._singulars if x == 0]))

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, :self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out

#Walsh-Hadamard Compressive Sensing
class WalshHadamardCS(A_functions):
    def fwht(self, vec): #the Fast Walsh Hadamard Transform is the same as its inverse
        a = vec.reshape(vec.shape[0], self.channels, self.img_dim**2)
        h = 1
        while h < self.img_dim**2:
            a = a.reshape(vec.shape[0], self.channels, -1, h * 2)
            b = a.clone()
            a[:, :, :, :h] = b[:, :, :, :h] + b[:, :, :, h:2*h]
            a[:, :, :, h:2*h] = b[:, :, :, :h] - b[:, :, :, h:2*h]
            h *= 2
        a = a.reshape(vec.shape[0], self.channels, self.img_dim**2) / self.img_dim
        return a

    def __init__(self, channels, img_dim, ratio, perm, device):
        self.channels = channels
        self.img_dim = img_dim
        self.ratio = ratio
        self.perm = perm
        self._singulars = torch.ones(channels * img_dim**2 // ratio, device=device)

    def V(self, vec):
        temp = torch.zeros(vec.shape[0], self.channels, self.img_dim**2, device=vec.device)
        temp[:, :, self.perm] = vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        return self.fwht(temp).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return self.fwht(vec.clone())[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        out[:, :self.channels * self.img_dim**2 // self.ratio] = vec.clone().reshape(vec.shape[0], -1)
        return out
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        temp_vec = self.fwht(vec.clone())[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

        singulars = self._singulars
        lambda_t = torch.ones(self.channels * self.img_dim ** 2, device=vec.device)
        temp = torch.zeros(self.channels * self.img_dim ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                    singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y)

        lambda_t = lambda_t.reshape(1, -1)
        temp_vec = temp_vec * lambda_t

        temp_out = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp_out[:, :, self.perm] = temp_vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        return self.fwht(temp_out).reshape(vec.shape[0], -1)
        
    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        temp_vec = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim ** 2)[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)
        temp_eps = epsilon.clone().reshape(
            vec.shape[0], self.channels, self.img_dim ** 2)[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

        d1_t = torch.ones(self.channels * self.img_dim ** 2, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.channels * self.img_dim ** 2, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5
        
        singulars = self._singulars
        temp = torch.zeros(self.channels * self.img_dim ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_y ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, -1)
        d2_t = d2_t.reshape(1, -1)
        
        temp_vec = temp_vec * d1_t
        temp_eps = temp_eps * d2_t

        temp_out_vec = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp_out_vec[:, :, self.perm] = temp_vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        temp_out_vec = self.fwht(temp_out_vec).reshape(vec.shape[0], -1)

        temp_out_eps = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp_out_eps[:, :, self.perm] = temp_eps.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        temp_out_eps = self.fwht(temp_out_eps).reshape(vec.shape[0], -1)
        
        return temp_out_vec + temp_out_eps
    
    
#Inpainting
class Inpainting(A_functions):
    def __init__(self, channels, img_dim, missing_indices, device):
        self.channels = channels
        self.img_dim = img_dim
        self._singulars = torch.ones(channels * img_dim**2 - missing_indices.shape[0]).to(device)
        self.missing_indices = missing_indices
        self.kept_indices = torch.Tensor([i for i in range(channels * img_dim**2) if i not in missing_indices]).to(device).long()

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, self.kept_indices] = temp[:, :self.kept_indices.shape[0]]
        out[:, self.missing_indices] = temp[:, self.kept_indices.shape[0]:]
        return out.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):

        temp = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        out[:, :self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0]:] = temp[:, self.missing_indices]

        singulars = self._singulars
        lambda_t = torch.ones(temp.size(1), device=vec.device)
        temp_singulars = torch.zeros(temp.size(1), device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                    singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y)

        lambda_t = lambda_t.reshape(1, -1)
        out = out * lambda_t

        result = torch.zeros_like(temp)
        result[:, self.kept_indices] = out[:, :self.kept_indices.shape[0]]
        result[:, self.missing_indices] = out[:, self.kept_indices.shape[0]:]
        return result.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        temp_vec = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out_vec = torch.zeros_like(temp_vec)
        out_vec[:, :self.kept_indices.shape[0]] = temp_vec[:, self.kept_indices]
        out_vec[:, self.kept_indices.shape[0]:] = temp_vec[:, self.missing_indices]

        temp_eps = epsilon.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1).reshape(vec.shape[0], -1)
        out_eps = torch.zeros_like(temp_eps)
        out_eps[:, :self.kept_indices.shape[0]] = temp_eps[:, self.kept_indices]
        out_eps[:, self.kept_indices.shape[0]:] = temp_eps[:, self.missing_indices]

        singulars = self._singulars
        d1_t = torch.ones(temp_vec.size(1), device=vec.device) * sigma_t * eta
        d2_t = torch.ones(temp_vec.size(1), device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5

        temp_singulars = torch.zeros(temp_vec.size(1), device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_y ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, -1)
        d2_t = d2_t.reshape(1, -1)
        out_vec = out_vec * d1_t
        out_eps = out_eps * d2_t

        result_vec = torch.zeros_like(temp_vec)
        result_vec[:, self.kept_indices] = out_vec[:, :self.kept_indices.shape[0]]
        result_vec[:, self.missing_indices] = out_vec[:, self.kept_indices.shape[0]:]
        result_vec = result_vec.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

        result_eps = torch.zeros_like(temp_eps)
        result_eps[:, self.kept_indices] = out_eps[:, :self.kept_indices.shape[0]]
        result_eps[:, self.missing_indices] = out_eps[:, self.kept_indices.shape[0]:]
        result_eps = result_eps.reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)
        
        return result_vec + result_eps

#Denoising
class Denoising(A_functions):
    def __init__(self, channels, img_dim, device):
        self._singulars = torch.ones(channels * img_dim**2, device=device)

    def V(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        if sigma_t < a * sigma_y:
            factor = (sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y).item()
            return vec * factor
        else:
            return vec
    
    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        if sigma_t >= a * sigma_y:
            factor = torch.sqrt(sigma_t ** 2 - a ** 2 * sigma_y ** 2).item()
            return vec * factor
        else:
            return vec * sigma_t * eta 

#Super Resolution
class SuperResolution(A_functions):
    def __init__(self, channels, img_dim, ratio, device): #ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        A = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(A, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        #reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(vec.shape[0], self.channels, self.y_dim**2, self.ratio**2, device=vec.device)
        patches[:, :, :, 0] = temp[:, :self.channels * self.y_dim**2].view(vec.shape[0], self.channels, -1)
        for idx in range(self.ratio**2-1):
            patches[:, :, :, idx+1] = temp[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1].view(vec.shape[0], self.channels, -1)
        #multiply each patch by the small V
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #repatch the patches into an image
        patches_orig = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        return recon

    def Vt(self, vec):
        #extract flattened patches
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #multiply each by the small V transposed
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        #reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        recon[:, :self.channels * self.y_dim**2] = patches[:, :, :, 0].view(vec.shape[0], self.channels * self.y_dim**2)
        for idx in range(self.ratio**2-1):
            recon[:, (self.channels*self.y_dim**2+idx)::self.ratio**2-1] = patches[:, :, :, idx+1].view(vec.shape[0], self.channels * self.y_dim**2)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        singulars = self.singulars_small
        
        patches = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches = patches.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        patches = torch.matmul(self.Vt_small, patches.reshape(-1, self.ratio**2, 1)).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        
        lambda_t = torch.ones(self.ratio ** 2, device=vec.device)
        
        temp = torch.zeros(self.ratio ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.
        
        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y)
            
        lambda_t = lambda_t.reshape(1, 1, 1, -1)
#         print("lambda_t:", lambda_t)
#         print("V:", self.V_small)
#         print(lambda_t.size(), self.V_small.size())
#         print("Sigma_t:", torch.matmul(torch.matmul(self.V_small, torch.diag(lambda_t.reshape(-1))), self.Vt_small))
        patches = patches * lambda_t
        
        
        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1))
        
        patches = patches.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches = patches.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
        return patches

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        singulars = self.singulars_small
        
        patches_vec = vec.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches_vec = patches_vec.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches_vec = patches_vec.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        patches_eps = epsilon.clone().reshape(vec.shape[0], self.channels, self.img_dim, self.img_dim)
        patches_eps = patches_eps.unfold(2, self.ratio, self.ratio).unfold(3, self.ratio, self.ratio)
        patches_eps = patches_eps.contiguous().reshape(vec.shape[0], self.channels, -1, self.ratio ** 2)
        
        d1_t = torch.ones(self.ratio ** 2, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.ratio ** 2, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5
        
        temp = torch.zeros(self.ratio ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.
        
        if a != 0 and sigma_y != 0:
            
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index  * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)
             
            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(change_index * (sigma_t ** 2 - a ** 2 * sigma_y ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)
            
            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5
        
        d1_t = d1_t.reshape(1, 1, 1, -1)
        d2_t = d2_t.reshape(1, 1, 1, -1)
        patches_vec = patches_vec * d1_t
        patches_eps = patches_eps * d2_t
        
        patches_vec = torch.matmul(self.V_small, patches_vec.reshape(-1, self.ratio**2, 1))
        
        patches_vec = patches_vec.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches_vec = patches_vec.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_vec = patches_vec.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
        patches_eps = torch.matmul(self.V_small, patches_eps.reshape(-1, self.ratio**2, 1))
        
        patches_eps = patches_eps.reshape(vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio)
        patches_eps = patches_eps.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_eps = patches_eps.reshape(vec.shape[0], self.channels * self.img_dim ** 2)
        
        return patches_vec + patches_eps
    

#Colorization
class Colorization(A_functions):
    def __init__(self, img_dim, device):
        self.channels = 3
        self.img_dim = img_dim
        #Do the SVD for the per-pixel matrix
        A = torch.Tensor([[0.3333, 0.3334, 0.3333]]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(A, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        #get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1) #shape: B, WA, C'
        #multiply each needle by the small V
        needles = torch.matmul(self.V_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels) #shape: B, WA, C
        #permute back to vector representation
        recon = needles.permute(0, 2, 1) #shape: B, C, WA
        return recon.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        #get the needles
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1) #shape: B, WA, C
        #multiply each needle by the small V transposed
        needles = torch.matmul(self.Vt_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels) #shape: B, WA, C'
        #reorder the vector so that the first entry of each needle is at the top
        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec): #U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.img_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], self.channels * self.img_dim**2), device=vec.device)
        temp[:, :self.img_dim**2] = reshaped
        return temp
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        needles = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)

        needles = torch.matmul(self.Vt_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels)

        singulars = self.singulars_small
        lambda_t = torch.ones(self.channels, device=vec.device)
        temp = torch.zeros(self.channels, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                        singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y)

        lambda_t = lambda_t.reshape(1, 1, self.channels)
        needles = needles * lambda_t

        needles = torch.matmul(self.V_small, needles.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels)

        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        singulars = self.singulars_small

        needles_vec = vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)
        needles_epsilon = epsilon.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)

        d1_t = torch.ones(self.channels, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.channels, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5

        temp = torch.zeros(self.channels, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_y ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, 1, self.channels)
        d2_t = d2_t.reshape(1, 1, self.channels)

        needles_vec = needles_vec * d1_t
        needles_epsilon = needles_epsilon * d2_t

        needles_vec = torch.matmul(self.V_small, needles_vec.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1, self.channels)
        recon_vec = needles_vec.permute(0, 2, 1).reshape(vec.shape[0], -1)

        needles_epsilon = torch.matmul(self.V_small, needles_epsilon.reshape(-1, self.channels, 1)).reshape(vec.shape[0], -1,self.channels)
        recon_epsilon = needles_epsilon.permute(0, 2, 1).reshape(vec.shape[0], -1)
        
        return recon_vec + recon_epsilon

#Walsh-Aadamard Compressive Sensing
class WalshAadamardCS(A_functions):
    def fwht(self, vec): #the Fast Walsh Aadamard Transform is the same as its inverse
        a = vec.reshape(vec.shape[0], self.channels, self.img_dim**2)
        h = 1
        while h < self.img_dim**2:
            a = a.reshape(vec.shape[0], self.channels, -1, h * 2)
            b = a.clone()
            a[:, :, :, :h] = b[:, :, :, :h] + b[:, :, :, h:2*h]
            a[:, :, :, h:2*h] = b[:, :, :, :h] - b[:, :, :, h:2*h]
            h *= 2
        a = a.reshape(vec.shape[0], self.channels, self.img_dim**2) / self.img_dim
        return a

    def __init__(self, channels, img_dim, ratio, perm, device):
        self.channels = channels
        self.img_dim = img_dim
        self.ratio = ratio
        self.perm = perm
        self._singulars = torch.ones(channels * img_dim**2 // ratio, device=device)

    def V(self, vec):
        temp = torch.zeros(vec.shape[0], self.channels, self.img_dim**2, device=vec.device)
        temp[:, :, self.perm] = vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        return self.fwht(temp).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return self.fwht(vec.clone())[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self.channels * self.img_dim**2, device=vec.device)
        out[:, :self.channels * self.img_dim**2 // self.ratio] = vec.clone().reshape(vec.shape[0], -1)
        return out
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        temp_vec = self.fwht(vec.clone())[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

        singulars = self._singulars
        lambda_t = torch.ones(self.channels * self.img_dim ** 2, device=vec.device)
        temp = torch.zeros(self.channels * self.img_dim ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                    singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y)

        lambda_t = lambda_t.reshape(1, -1)
        temp_vec = temp_vec * lambda_t

        temp_out = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp_out[:, :, self.perm] = temp_vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        return self.fwht(temp_out).reshape(vec.shape[0], -1)
        
    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        temp_vec = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim ** 2)[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)
        temp_eps = epsilon.clone().reshape(
            vec.shape[0], self.channels, self.img_dim ** 2)[:, :, self.perm].permute(0, 2, 1).reshape(vec.shape[0], -1)

        d1_t = torch.ones(self.channels * self.img_dim ** 2, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.channels * self.img_dim ** 2, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5
        
        singulars = self._singulars
        temp = torch.zeros(self.channels * self.img_dim ** 2, device=vec.device)
        temp[:singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_y ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, -1)
        d2_t = d2_t.reshape(1, -1)
        
        temp_vec = temp_vec * d1_t
        temp_eps = temp_eps * d2_t

        temp_out_vec = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp_out_vec[:, :, self.perm] = temp_vec.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        temp_out_vec = self.fwht(temp_out_vec).reshape(vec.shape[0], -1)

        temp_out_eps = torch.zeros(vec.shape[0], self.channels, self.img_dim ** 2, device=vec.device)
        temp_out_eps[:, :, self.perm] = temp_eps.clone().reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1)
        temp_out_eps = self.fwht(temp_out_eps).reshape(vec.shape[0], -1)
        
        return temp_out_vec + temp_out_eps

#Convolution-based super-resolution
class SRConv(A_functions):
    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim,
                        dim)).reshape(v.shape[0], self.channels, M.shape[0], dim)

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim,
                        dim), M).reshape(v.shape[0], self.channels, dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, stride = 1):
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        #build 1D conv matrix
        A_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride//2, img_dim + stride//2, stride):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                j_effective = j
                #reflective padding
                if j_effective < 0: j_effective = -j_effective-1
                if j_effective >= img_dim: j_effective = (img_dim - 1) - (j_effective - img_dim)
                #matrix building
                A_small[i // stride, j_effective] += kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(A_small, some=False)
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small.reshape(small_dim, 1), self.singulars_small.reshape(1, small_dim)).reshape(small_dim**2)
        #permutation for matching the singular values. See P_1 in Appendix D.5.
        self._perm = torch.Tensor([self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim)] + \
                                  [self.img_dim * i + j for i in range(self.small_dim) for j in range(self.small_dim, self.img_dim)]).to(device).long()

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, :self._perm.shape[0], :]
        temp[:, self._perm.shape[0]:, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)[:, self._perm.shape[0]:, :]
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp[:, :, :self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.small_dim**2, self.channels, device=vec.device)
        temp[:, :self.small_dim**2, :] = vec.clone().reshape(vec.shape[0], self.small_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1), self.small_dim).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone(), self.small_dim)
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(vec.shape[0], self.channels, -1)
        #permute the entries
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros((vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device)
        temp[:, :reshaped.shape[1]] = reshaped
        return temp

#Deblurring
class Deblurring(A_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device, ZERO = 3e-2):
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix
        A_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0]//2, i + kernel.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                A_small[i, j] = kernel[j - i + kernel.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(A_small, some=False)
        #ZERO = 3e-2
        self.singulars_small_orig = self.singulars_small.clone()
        self.singulars_small[self.singulars_small < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars_orig = torch.matmul(self.singulars_small_orig.reshape(img_dim, 1), self.singulars_small_orig.reshape(1, img_dim)).reshape(img_dim**2)
        self._singulars = torch.matmul(self.singulars_small.reshape(img_dim, 1), self.singulars_small.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)
        self._singulars_orig = self._singulars_orig[self._perm]

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)
    
    def A_pinv(self, vec):
        temp = self.Ut(vec)
        singulars = self._singulars.repeat(1, 3).reshape(-1)
        
        factors = 1. / singulars
        factors[singulars == 0] = 0.
        
        temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] * factors
        return self.V(self.add_zeros(temp))
    
    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        temp_vec = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp_vec = self.img_by_mat(temp_vec, self.V_small).reshape(vec.shape[0], self.channels, -1)
        temp_vec = temp_vec[:, :, self._perm].permute(0, 2, 1)

        singulars = self._singulars_orig
        lambda_t = torch.ones(self.img_dim ** 2, device=vec.device)
        temp_singulars = torch.zeros(self.img_dim ** 2, device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                    singulars * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y)

        lambda_t = lambda_t.reshape(1, -1, 1)
        temp_vec = temp_vec * lambda_t

        temp = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp[:, self._perm, :] = temp_vec.clone().reshape(vec.shape[0], self.img_dim ** 2, self.channels)
        temp = temp.permute(0, 2, 1)
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        temp_vec = vec.clone().reshape(vec.shape[0], self.channels, -1)
        temp_vec = temp_vec[:, :, self._perm].permute(0, 2, 1)

        temp_eps = epsilon.clone().reshape(vec.shape[0], self.channels, -1)
        temp_eps = temp_eps[:, :, self._perm].permute(0, 2, 1)

        singulars = self._singulars_orig
        d1_t = torch.ones(self.img_dim ** 2, device=vec.device) * sigma_t * eta
        d2_t = torch.ones(self.img_dim ** 2, device=vec.device) * sigma_t * (1 - eta ** 2) ** 0.5

        temp_singulars = torch.zeros(self.img_dim ** 2, device=vec.device)
        temp_singulars[:singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1. / singulars
        inverse_singulars[singulars == 0] = 0.

        if a != 0 and sigma_y != 0:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t ** 2 - a ** 2 * sigma_y ** 2 * inverse_singulars ** 2))
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0) + change_index * sigma_t * (1 - eta ** 2) ** 0.5

        d1_t = d1_t.reshape(1, -1, 1)
        d2_t = d2_t.reshape(1, -1, 1)

        temp_vec = temp_vec * d1_t
        temp_eps = temp_eps * d2_t

        temp_vec_new = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp_vec_new[:, self._perm, :] = temp_vec
        out_vec = self.mat_by_img(self.V_small, temp_vec_new.permute(0, 2, 1))
        out_vec = self.img_by_mat(out_vec, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)

        temp_eps_new = torch.zeros(vec.shape[0], self.img_dim ** 2, self.channels, device=vec.device)
        temp_eps_new[:, self._perm, :] = temp_eps
        out_eps = self.mat_by_img(self.V_small, temp_eps_new.permute(0, 2, 1))
        out_eps = self.img_by_mat(out_eps, self.V_small.transpose(0, 1)).reshape(vec.shape[0], -1)

        return out_vec + out_eps

#Anisotropic Deblurring
class Deblurring2D(A_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel1, kernel2, channels, img_dim, device):
        self.img_dim = img_dim
        self.channels = channels
        A_small1 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel1.shape[0]//2, i + kernel1.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                A_small1[i, j] = kernel1[j - i + kernel1.shape[0]//2]
        A_small2 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel2.shape[0]//2, i + kernel2.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                A_small2[i, j] = kernel2[j - i + kernel2.shape[0]//2]
        self.U_small1, self.singulars_small1, self.V_small1 = torch.svd(A_small1, some=False)
        self.U_small2, self.singulars_small2, self.V_small2 = torch.svd(A_small2, some=False)
        ZERO = 3e-2
        self.singulars_small1[self.singulars_small1 < ZERO] = 0
        self.singulars_small2[self.singulars_small2 < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small1.reshape(img_dim, 1), self.singulars_small2.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small1, temp)
        out = self.img_by_mat(out, self.V_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small2).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small1, temp)
        out = self.img_by_mat(out, self.U_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small2).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)