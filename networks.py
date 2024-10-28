import torch
from torch import nn
import tinycudann as tcnn
from torch.nn import functional as F


class EncodingMLP(nn.Module):
	'''
	Base component of custom MLP.
	encoding_config follows the documentation of Encoding of tinycudann.
	'''
	def __init__(self, input_dim, output_dim, encoding_config=None, num_layers=4, num_neurons=64, activation='ReLU', mlp_lr=5e-4, grid_lr=5e-2):
		super(EncodingMLP, self).__init__()
		activation_funcs = dict(
			ReLU=nn.ReLU
		)
		assert num_layers >= 2
		assert activation in activation_funcs

		modules = nn.ModuleList()
		MLP_input_dim = input_dim
		# Encoding can be turned off if the config is None. Same as 'Identity' encoding config.
		if encoding_config is None:
			encoding_config = {'otype': 'Identity'}
		encoding = tcnn.Encoding(input_dim, encoding_config, dtype=torch.float32)
		# TODO: record if encoding is grid.
		self.grid_encode = True if encoding_config['otype'] in ['Grid', 'DenseGrid', 'HashGrid'] else False
		modules.append(encoding)
		MLP_input_dim = encoding.n_output_dims

		# input layer
		modules.append(nn.Linear(MLP_input_dim, num_neurons))
		modules.append(activation_funcs[activation]())

		# hidden layers
		for _ in range(num_layers-2):
			modules.append(nn.Linear(num_neurons, num_neurons))
			modules.append(activation_funcs[activation]())

		# output layer
		modules.append(nn.Linear(num_neurons, output_dim))

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.num_layers = num_layers
		self.num_neurons = num_neurons
		self.network = nn.Sequential(*modules)
		self.grid_lr = grid_lr
		self.mlp_lr = mlp_lr

	def forward(self, x):
		# TODO: temp impl: [-1, 1] to [0, 1]
		if self.grid_encode: x = x / 2 + 0.5
		return self.network(x)

	def get_optimizer_list(self):
		optimizer_list = list()
		optimizer_list.append({'params': self.network[0].parameters(), 'lr': self.grid_lr})
		for i in range(1, len(self.network)):
			optimizer_list.append({'params': self.network[i].parameters(), 'lr': self.mlp_lr})
		return optimizer_list

# Residual estimators
class ResidualEstimator(nn.Module):
	def __init__(self, encoding_config, num_layers=4, num_neurons=64, mlp_lr=5e-4, grid_lr=1e-2):
		super(ResidualEstimator, self).__init__()
		self.input_dim = 3
		self.output_dim = 3
		self.residual = EncodingMLP(self.input_dim, self.output_dim, encoding_config, num_layers, num_neurons, mlp_lr=mlp_lr, grid_lr=grid_lr)

	def forward(self, uv, t):
		return F.softplus(self.residual(torch.hstack((uv, t))).to(torch.float32))

	def get_optimizer_list(self):
		return self.residual.get_optimizer_list()

# Texture Networks
class BaseTextureNetwork(nn.Module):
	def __init__(self, input_dim):
		super(BaseTextureNetwork, self).__init__()
		self.input_dim = input_dim
		self.output_dim = 3 # (r, g, b)

	def forward(self, uv):
		return (torch.tanh(self.texture(uv).to(torch.float32)) + 1.0) * 0.5

class EncodingTextureNetwork(BaseTextureNetwork):
	def __init__(self, input_dim, encoding_config, num_layers=4, num_neurons=64, mlp_lr=5e-4, grid_lr=5e-2):
		super(EncodingTextureNetwork, self).__init__(input_dim)
		self.texture = EncodingMLP(self.input_dim, self.output_dim, encoding_config, num_layers, num_neurons, mlp_lr=mlp_lr, grid_lr=grid_lr)

	def get_optimizer_list(self):
		return self.texture.get_optimizer_list()

# Mapping Networks
class BaseMappingNetwork(nn.Module):
	def __init__(self, pretrain, texture, residual):
		super(BaseMappingNetwork, self).__init__()
		self.input_dim = 3 # (x, y, t)
		self.output_dim = 2 # (u, v)
		self.pretrain = pretrain
		self.model_texture = build_network(input_dim=self.output_dim, **texture)
		self.model_residual = build_network(**residual)
		#self.model_residual = None

	def forward(self, xyt, return_residual=False, return_rgb=False):
		res = list()
		uv = self._get_uv(xyt)
		res.append(uv)
		if return_residual:
			if self.model_residual is None:
				residual = torch.ones(len(uv), 3).to(uv)
			else:
				residual = self.model_residual(uv.detach(), xyt[:, [2]])
				#residual = self.model_residual(xyt)
			res.append(residual)

		if return_rgb:
			rgb = self.model_texture(uv)
			res.append(rgb)

		if len(res) == 1:
			return res[0]
		else:
			return tuple(res)

	def _get_uv(self, xyt):
		return torch.tanh(self.model_mapping(xyt).to(torch.float32)) # [-1, 1]

	def get_optimizer_list(self):
		optimizer_list = list()
		optimizer_list.extend(self.model_texture.get_optimizer_list())
		if self.model_residual is not None:
			optimizer_list.extend(self.model_residual.get_optimizer_list())
		return optimizer_list

class EncodingMappingNetwork(BaseMappingNetwork):
	def __init__(self, pretrain, texture, residual, encoding_config, num_layers=4, num_neurons=64, mlp_lr=5e-4, grid_lr=5e-2):
		super(EncodingMappingNetwork, self).__init__(pretrain, texture, residual)
		self.model_mapping = EncodingMLP(self.input_dim, self.output_dim, encoding_config, num_layers, num_neurons, mlp_lr=mlp_lr, grid_lr=grid_lr)

	def get_optimizer_list(self):
		optimizer_list = list()
		optimizer_list.extend(self.model_mapping.get_optimizer_list())
		optimizer_list.extend(super(EncodingMappingNetwork, self).get_optimizer_list())
		return optimizer_list

# Alpha Networks
class BaseAlphaNetwork(nn.Module):
	def __init__(self):
		super(BaseAlphaNetwork, self).__init__()
		self.input_dim = 3 # (x, y, t)
		self.output_dim = 1
		self.eps = 1e-5

	def forward(self, xyt):
		alpha = self.alpha(xyt).to(torch.float32)
		alpha = (torch.tanh(alpha) + 1.0) * 0.5 # normalize to [0, 1]
		alpha = torch.clamp(alpha, self.eps, 1-self.eps)
		return alpha

class EncodingAlphaNetwork(BaseAlphaNetwork):
	def __init__(self, encoding_config, num_layers=4, num_neurons=64, mlp_lr=5e-4, grid_lr=5e-2):
		super(EncodingAlphaNetwork, self).__init__()
		self.alpha = EncodingMLP(
			self.input_dim,
			self.output_dim,
			encoding_config,
			num_layers, num_neurons,
			mlp_lr=mlp_lr, grid_lr=grid_lr)

	def get_optimizer_list(self):
		return self.alpha.get_optimizer_list()

# General Network Building Function
def build_network(model_type: str, device='cpu', **kwargs):
	#print(model_type)
	if model_type == 'None':
		return None
	return globals()[model_type](**kwargs).to(device)
