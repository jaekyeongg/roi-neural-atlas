from easydict import EasyDict
import os

# merge d2 into d1
def _merge(d1: EasyDict, d2: EasyDict):
	res = EasyDict(d1)
	for key, value in d2.items():
		if isinstance(value, EasyDict) and isinstance(res.get(key, None), EasyDict):
			res[key] = _merge(res[key], value)
		else: res[key] = value
	return res

def config_load(filename) -> EasyDict:
	if filename.find('/') != -1:
		__CONFIG_ROOTDIR__ = filename.split('/')[-2]
	else: __CONFIG_ROOTDIR__ = ''
	with open(filename, 'r') as f:
		__CONTENT__ = f.read()
	del filename, f
	exec(__CONTENT__)
	res = EasyDict(**locals())
	res.pop('__CONTENT__')
	res.pop('__CONFIG_ROOTDIR__')
	if res.get('_base_', None) is not None:
		bases = EasyDict()
		for basename in list(res._base_):
			base = config_load(os.path.join(__CONFIG_ROOTDIR__, basename))
			bases = _merge(bases, base)
		res = _merge(res, bases)
	return res

from pprint import pformat

def config_save(config, filename):
	res = ''.join([f'{key} = {pformat(item, indent=4)}\n' for key, item in config.items()])
	with open(filename, 'w') as f:
		f.write(res)
