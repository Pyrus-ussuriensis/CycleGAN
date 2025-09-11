import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
import random

class CycleGAN(L.LightningModule):
    def __init__(self, G:nn.Module, F:nn.Module, Dx:nn.Module, Dy:nn.Module, 
                  lambda_cyc=10, lambda_id=5, n_epochs=100, n_epochs_decay=100, lr=2e-4):
        super(CycleGAN, self).__init__()
        self.G= G #Generator1
        self.F= F #Generator2
        self.Dx= Dx #Discriminator1
        self.Dy= Dy #Discriminator2
        self.lambda_cyc = float(lambda_cyc) # lambda1 for Lcyc_GF
        self.lambda_id = float(lambda_id) # lambda2 for Lid
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.lr = lr
        self.save_hyperparameters(ignore=["G", "F", "Dx", "Dy"])

        #self.example_input_array = torch.Tensor(1, 3, 256, 256)

        self.automatic_optimization = False

        self.fakes_m = []
        self.fakes_p = []
    
    @torch.no_grad()
    def query(self, fakes, fake):
        out = fake
        if len(fakes) < 50:
            fakes.append(fake)
        elif random.random() < 0.5:
            i = random.randint(0, len(fakes)-1)
            out = fakes[i]
            fakes[i] = fake
        return out

    def training_step(self, batch, batch_idx):
        m, p = batch

        optG, optD = self.optimizers()

    #if optimizer_idx == 0:
        fake_m = self.G(p)
        fake_p = self.F(m)

        out_Gy = self.Dy(fake_m)
        out_Gx = self.Dx(fake_p)
        #LGAN_GDXY = (out_Gy-1)^2 + (out_y-1)^2 + out_Gy^2
        #LGAN_FDXY = (out_Gx-1)^2 + (out_x-1)^2 + out_Gx^2

        LGAN_G = F.mse_loss(out_Gy, torch.ones_like(out_Gy))
        LGAN_F = F.mse_loss(out_Gx, torch.ones_like(out_Gx))
        LGAN_GF = LGAN_F + LGAN_G

        Lcyc_GF = F.l1_loss(self.F(self.G(p)), p) + F.l1_loss(self.G(self.F(m)), m)
        Lid = F.l1_loss(self.G(m), m) + F.l1_loss(self.F(p), p)
        Total_Loss0 = LGAN_GF + self.lambda_cyc*Lcyc_GF + self.lambda_id*Lid
        self.toggle_optimizer(optG);optG.zero_grad(); self.manual_backward(Total_Loss0); optG.step(); self.untoggle_optimizer(optG)

        self.log_dict({"g/Total_Loss":Total_Loss0,"g/LGAN_GF":LGAN_GF, "g/Lcyc_GF":Lcyc_GF, "g/Lid":Lid}, prog_bar=True, on_step=True, on_epoch=True)
        #return Total_Loss0
    
    #if optimizer_idx == 1:
        with torch.no_grad():
            fake_m = self.G(p)
            fake_p = self.F(m)
        fake_m = self.query(self.fakes_m, fake_m).detach()
        fake_p = self.query(self.fakes_p, fake_p).detach()

        out_Gy = self.Dy(fake_m)
        out_Gx = self.Dx(fake_p)
        out_y = self.Dy(m)
        out_x = self.Dx(p)
        #LGAN_GDXY = (out_Gy-1)^2 + (out_y-1)^2 + out_Gy^2
        #LGAN_FDXY = (out_Gx-1)^2 + (out_x-1)^2 + out_Gx^2
        LGAN_DX = F.mse_loss(out_y, torch.ones_like(out_y)) + F.mse_loss(out_Gy, torch.zeros_like(out_Gy))
        LGAN_DY = F.mse_loss(out_x, torch.ones_like(out_x)) + F.mse_loss(out_Gx, torch.zeros_like(out_Gx))
        LGAN_DXY = LGAN_DX + LGAN_DY

        Total_Loss1 = LGAN_DXY/2
        self.toggle_optimizer(optD); optD.zero_grad(); self.manual_backward(Total_Loss1); optD.step(); self.untoggle_optimizer(optD)

        self.log_dict({"d/Total_Loss":Total_Loss1,"d/LGAN_DX":LGAN_DX, "d/LGAN_DY":LGAN_DY}, prog_bar=True, on_step=True, on_epoch=True)


        #return Total_Loss1
    def on_train_epoch_end(self):
        schG, schD = self.lr_schedulers()
        schG.step(); schD.step()


    def _shared_eval_step(self, batch, batch_idx, state):
        m, p = batch
        fake_m = self.G(p)
        fake_p = self.F(m)

        out_Gy = self.Dy(fake_m)
        out_Gx = self.Dx(fake_p)
        out_y = self.Dy(m)
        out_x = self.Dx(p)
        #LGAN_GDXY = (out_Gy-1)^2 + (out_y-1)^2 + out_Gy^2
        #LGAN_FDXY = (out_Gx-1)^2 + (out_x-1)^2 + out_Gx^2
        LGAN_G = F.mse_loss(out_Gy, torch.ones_like(out_Gy))
        LGAN_F = F.mse_loss(out_Gx, torch.ones_like(out_Gx))
        LGAN_GF = LGAN_F + LGAN_G
        LGAN_DX = F.mse_loss(out_y, torch.ones_like(out_y)) + F.mse_loss(out_Gy, torch.zeros_like(out_Gy))
        LGAN_DY = F.mse_loss(out_x, torch.ones_like(out_x)) + F.mse_loss(out_Gx, torch.zeros_like(out_Gx))
        LGAN_DXY = LGAN_DX + LGAN_DY


        Lcyc_GF = F.l1_loss(self.F(self.G(p)), p) + F.l1_loss(self.G(self.F(m)), m)
        Lid = F.l1_loss(self.G(m), m) + F.l1_loss(self.F(p), p)
        Total_Loss = LGAN_GF + LGAN_DXY + self.lambda_cyc*Lcyc_GF + self.lambda_id*Lid
        self.log_dict({f"{state}_Total_Loss":Total_Loss, f"{state}_LGAN_GF":LGAN_GF, f"{state}_LGAN_DXY":LGAN_DXY, 
                        f"{state}_Lcyc_GF":Lcyc_GF, f"{state}_Lid":Lid}, prog_bar=True, on_epoch=True)
        return Total_Loss


    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch=batch, batch_idx=batch_idx, state="val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch=batch, batch_idx=batch_idx, state="test")


    def configure_optimizers(self):
        opt_G = torch.optim.Adam(list(self.G.parameters())+list(self.F.parameters()), lr=self.lr, betas=(0.5, 0.999)) 
        opt_D = torch.optim.Adam(list(self.Dx.parameters())+list(self.Dy.parameters()), lr=self.lr, betas=(0.5, 0.999)) 

        def lambda_rule(epoch):
            if epoch < self.n_epochs: 
                return 1.0
            t = (epoch - self.n_epochs) / float(max(1, self.n_epochs_decay))
            return max(0.0, 1.0 - t)

        sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_rule)
        sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lambda_rule)

        return ([opt_G, opt_D],
                [{"scheduler": sch_G, "interval": "epoch"},
                 {"scheduler": sch_D, "interval": "epoch"}])



