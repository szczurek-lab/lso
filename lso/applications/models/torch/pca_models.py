from typing import Dict
from typing import Optional

from torch import nn
import torch
from pytorch_lightning.utilities import types as pl_u_types

from lso.data import data as lso_data, numpy_data as lso_np_data
from lso.model import model as lso_model
from lso.applications.models.torch import torch_model as lso_tm


class SimplePCATorchModel(lso_tm.PytorchModel):

    def __init__(
            self,
            latent_dim: int,
            input_dim: int,
    ):
        self.latent_dim = latent_dim
        self.input_dim = input_dim

    def get_instance(self, instance_params: Optional[lso_model.InstanceParams] = None):
        model_instance = SimplePCATorchModelInstance(
            latent_dim=self.latent_dim,
            input_dim=self.input_dim,
        )

        if instance_params is None:
            return model_instance

        if isinstance(instance_params, lso_tm.PytorchInstanceParams):
            model_instance.load_state_dict(instance_params.state_dict)
            return model_instance

        raise ValueError(f'BasicPCATorchModel accepts only PytorchInstanceParams, but {instance_params}'
                         f'were provided.')

    @classmethod
    def from_config_dict(cls, config_dict):
        return SimplePCATorchModel(
            latent_dim=config_dict['latent_dim'],
            input_dim=config_dict['input_dim'],
        )

    def get_config_dict(self) -> Dict:
        return {
            'latent_dim': self.latent_dim,
            'input_dim': self.input_dim,
        }


class SimplePCATorchModelInstance(lso_tm.PytorchModelInstance):

    def __init__(self, latent_dim: int, input_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, 2 * self.latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, self.input_dim))

    def forward(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        mu, log_var = torch.chunk(self.encoder(x), 2, dim=1)
        var = torch.exp(log_var)
        enc_dist = torch.distributions.Normal(mu, var)
        z = enc_dist.rsample()
        logits = self.decoder(z)
        x_hat = torch.round(torch.sigmoid(logits))
        return x_hat

    def training_step(self, batch, batch_idx):
        mu, log_var = torch.chunk(self.encoder(batch), 2, dim=1)
        var = torch.exp(log_var)
        enc_dist = torch.distributions.Normal(mu, var)
        z = enc_dist.rsample()
        logits = self.decoder(z)
        x_hat = torch.round(torch.sigmoid(logits))
        loss = nn.BCELoss()(x_hat, batch)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def get_params(self):
        return lso_tm.PytorchInstanceParams(state_dict=self.state_dict())

    def encode(self, data: lso_np_data.NumpyData) -> lso_data.Latent:
        mu, log_var = torch.chunk(self.encoder(torch.tensor(data.x)), 2, dim=1)
        return lso_np_data.NumpyLatent(
            z=mu.detach().numpy(),
            objective=data.objective,
            features=data.features,
        )

    def decode(self, latent: lso_data.Latent) -> lso_np_data.NumpyData:
        return lso_np_data.NumpyData(
            x=self.decoder.forward(torch.tensor(latent.z)).detach().numpy(),
            objective=latent.objective,
            features=latent.features,
        )

    def train_dataloader(self) -> pl_u_types.TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> pl_u_types.EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> pl_u_types.EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> pl_u_types.EVAL_DATALOADERS:
        pass
