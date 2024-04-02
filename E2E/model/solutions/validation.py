import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pytorch_msssim import SSIM
import gc


class validation:
    def __init__(self, DL, neptunerun, model, batch_size=8, optimizer=(), device=None, *args):
        self.dl = DL
        self.run = neptunerun
        self.model = model
        self.optimizer = optimizer if isinstance(optimizer, (tuple, list)) else (optimizer,)
        metrics = (torch.nn.L1Loss(), torch.nn.MSELoss(), SSIM(1.0, channel=1))
        self.metrics = {m.__class__.__name__: m for m in metrics}
        self.epoch = 0
        self.device = next(model.parameters()).device if device is None else device
        self.init(*args)

    def init(self, *args):
        pass

    def get_prediction(self, output):
        return output

    def prepare_input(self, data):
        return data

    def log_intermediate_values(self, accumulators, output, gt, mask, *other):
        pass

    def log_sample(self, output, gt, netinput, mask, other):
        p, gt = self.mask((self.get_prediction(output)[0].detach(), gt[0].detach()), mask[0], np.nan)
        self.log_image(p.cpu().numpy(), gt.cpu().numpy())

    def log_image(self, prediction, gt):
        assert prediction.shape == gt.shape
        fig, axs = plt.subplots(3, len(prediction), tight_layout=True, figsize=(len(prediction) * 2.5, 6), dpi=100)

        def plot(ax, data, cutpct=1, sym=False, **kwargs):
            vmin, vmax = np.nanpercentile(data, cutpct), np.nanpercentile(data, 100 - cutpct)
            if sym:
                vmax = max(np.abs(vmax), np.abs(vmin))
                vmin = min(-np.abs(vmin), -np.abs(vmax))
                kwargs["cmap"] = kwargs.get("cmap", "coolwarm")
            c = ax.matshow(data, vmin=vmin, vmax=vmax, **kwargs)
            plt.colorbar(c, ax=ax)

        for ax, p, g in zip(axs.T, prediction, gt):
            plot(ax[0], p, cmap="viridis", cutpct=5)
            plot(ax[1], g, cmap="viridis", cutpct=5)
            plot(ax[2], p - g, cmap="coolwarm", sym=True)
        self.run["validation/sample"].log(fig, self.epoch)
        plt.close()

    def log_parameters(self):
        for name, weight in self.model.named_parameters():
            try:
                self.run[f"weights/norm/{name}"].log(weight.data.detach().norm().cpu(), self.epoch)
                self.run[f"weights/mean/{name}"].log(weight.data.detach().mean().cpu(), self.epoch)
                if weight.data.numel() > 1:
                    self.run[f"weights/var/{name}"].log(weight.data.detach().var().cpu(), self.epoch)
                    self.run[f"weights/gradvar/{name}"].log(weight.grad.data.detach().var().cpu(), self.epoch)
                self.run[f"weights/gradnorm/{name}"].log(weight.grad.data.detach().norm().cpu(), self.epoch)
            except Exception as e:
                pass
        if self.optimizer is not None:
            if isinstance(self.optimizer, (tuple, list)):
                for j, opt in enumerate(self.optimizer):
                    for i, param_group in enumerate(opt.param_groups):
                        self.run[f"lr_{j}_{i}"].log(param_group["lr"], self.epoch)
            else:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    self.run[f"lr_{i}"].log(param_group["lr"], self.epoch)

    def log_parameters_extra(self):
        pass

    def mask(self, tensors, mask, val=0.0):
        if mask is None:
            return tensors
        ret = []
        for t in tensors:
            masked = t.clone()
            masked[~mask.expand_as(masked)] = val
            ret.append(masked)
        return ret

    def __call__(self, traininglosses=(), epoch=None):
        torch.cuda.synchronize()
        gc.collect()

        if epoch is not None:
            newepoch = epoch
        else:
            newepoch = self.epoch + 1

        for iloss, losses in enumerate(traininglosses):
            prop = self.run[f"training/loss_{iloss}"]
            for i, v in enumerate(losses):
                stamp = self.epoch + (newepoch - self.epoch) * (i + 1) / len(losses)
                prop.log(float(v), stamp)

        self.epoch = newepoch

        self.log_parameters()
        self.log_parameters_extra()

        with torch.no_grad():
            self.model.eval()
            accumulators = defaultdict(float)
            ret = 0.0

            for nbatch, data in enumerate(self.dl):
                netinput, gt, mask, *other = self.prepare_input(data)
                output = self.model(*netinput)
                prediction = self.get_prediction(output)
                if torch.is_complex(gt):
                    gt = torch.view_as_real(gt)
                    prediction = torch.view_as_real(prediction)
                mask_prediction, mask_gt = self.mask((prediction, gt), mask)
                for metricname, metric in self.metrics.items():

                    for channel in range(gt.shape[1]):
                        loss = metric(mask_prediction[:, channel : channel + 1], mask_gt[:, channel : channel + 1]).item()
                        accumulators[f"{metricname}/{channel}"] += loss
                        ret += loss
                try:
                    self.log_intermediate_values(accumulators, output, gt, netinput, mask, *other)
                except Exception as e:
                    print("Error in logging intermediate values. continueing.", e)
            self.log_sample(output, gt, netinput, mask, other)

            for key, accum in accumulators.items():
                value = accum / (nbatch + 1)
                self.run[f"validation/{key}"].log(value, self.epoch)
                ret += value

        torch.cuda.synchronize()
        gc.collect()
        return ret


class validationE2E(validation):
    def get_prediction(self, output):
        return output[0][-1]

    def prepare_input(self, data):
        with torch.no_grad():
            x, mask, noise, *enc_args = (i.to(self.model.device) for i in data)
            enc = self.model.getEncodingOperator(*enc_args)
            y = self.model.q(x)
            k = enc.A(y)
            k += enc.mask(noise)
            netinput = (k, enc)
            gt = x
            mask = mask
            other = None
        return netinput, gt, mask, other

    def log_parameters_extra(self):
        for name, weight in self.model.named_parameters():
            if "lambda" in name:
                try:
                    self.run[f"weights/softplus_{name}"].log(self.model.net.softplus(weight.data).detach().cpu(), self.epoch)
                except:
                    pass

    def log_intermediate_values(self, accumulators, output, gt, netinput, mask, *other):
        def mse_loss(x0, x1):
            if torch.is_complex(x0) or torch.is_complex(x1):
                if torch.is_complex(x0) and torch.is_complex(x1):
                    return 2 * F.mse_loss(torch.view_as_real(x0), torch.view_as_real(x1))
                else:
                    raise TypeError(f"either both or neither should be complex not {torch.is_complex(x0)} and {torch.is_complex(x1)}")
            else:
                return F.mse_loss(x0, x1)

        xtrue = gt
        ytrue = self.model.q(gt) + 0j
        x, y, xreg, yreg, *yreg2 = output
        k, enc = netinput

        x = self.mask(x, mask)
        y = self.mask(y, mask)
        yreg = self.mask(yreg, mask)
        xtrue, ytrue = self.mask([xtrue, ytrue], mask)

        for i in range(len(y)):
            val = mse_loss(enc.A(y[i]), k)
            key = f"Ay_k/{i}"
            accumulators[f"intermediate/{key}"] += val.detach().cpu()

            for c in range(y[i].shape[1]):
                val = mse_loss(y[i][:, c], ytrue[:, c])
                key = f"y_ytrue/channel_{c}/{i}"
                accumulators[f"intermediate/{key}"] += val.detach().cpu()

        for i in range(min(len(x), len(y))):
            val = mse_loss(self.model.q(x[i]).abs(), y[i].abs())
            key = f"qx_y/{i}"
            accumulators[f"intermediate/{key}"] += val.detach().cpu()

        for i in range(len(x)):
            for c in range(x[i].shape[1]):
                val = mse_loss(x[i][:, c], xtrue[:, c])
                key = f"x_xtrue/channel_{c}/{i}"
                accumulators[f"intermediate/{key}"] += val.detach().cpu()

        for i in range(len(yreg)):
            val = mse_loss(y[i + 1], yreg[i])
            key = f"y_yreg/{i+1}"
            accumulators[f"intermediate/{key}"] += val.detach().cpu()

            for c in range(yreg[i].shape[1]):
                val = mse_loss(ytrue[:, c], yreg[i][:, c])
                key = f"yreg_ytrue/channel_{c}/{i+1}"
                accumulators[f"intermediate/{key}"] += val.detach().cpu()

        if yreg2:
            yreg2 = yreg2[0]
            for i in range(len(yreg2)):
                val = mse_loss(y[i + 1], yreg2[i])
                key = f"y_yreg2/{i+1}"
                accumulators[f"intermediate/{key}"] += val.detach().cpu()

            for c in range(yreg2[i].shape[1]):
                val = mse_loss(ytrue[:, c], yreg2[i][:, c])
                key = f"yreg_ytrue/channel_{c}/{i+1}"
                accumulators[f"intermediate/{key}"] += val.detach().cpu()

        for i in range(min(len(xreg), len(x))):
            val = mse_loss(x[i], xreg[i])
            key = f"x_xreg/{i}"
            accumulators[f"intermediate/{key}"] += val.detach().cpu()

            for c in range(xreg[i].shape[1]):
                val = mse_loss(xtrue[:, c], xreg[i][:, c])
                key = f"xreg_xtrue/channel_{c}/{i}"
                accumulators[f"intermediate/{key}"] += val.detach().cpu()
