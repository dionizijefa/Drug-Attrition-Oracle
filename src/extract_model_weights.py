""" Reduce the file size of the model """
from collections import OrderedDict
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1].absolute()

def main():
    production_dir = str(root / 'production')
    for path in Path(production_dir).rglob('*.ckpt'):
        try:
            state_dict = torch.load(
                path,
                map_location=torch.device("cpu"))
            ckpt_dir = path.resolve().parents[0].absolute()
            new_state_dict = OrderedDict()
            for k, v in state_dict['state_dict'].items():
                name = k[6:]  # remove `model'
                new_state_dict[name] = v
            hyperparams = state_dict['hyper_parameters']
            all_dict = OrderedDict()
            all_dict['state_dict'] = new_state_dict
            all_dict['hyper_parameters'] = hyperparams
            torch.save(all_dict, ckpt_dir / 'state_dict.pt')

        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()
