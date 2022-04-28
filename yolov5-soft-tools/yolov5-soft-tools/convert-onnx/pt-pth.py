import torch
import pickle
import argparse
from collections import OrderedDict

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='best')
    args = parser.parse_args()

    modelfile = args.source + '.pt'
    utl_model = torch.load(modelfile, map_location=device)
    utl_param = utl_model['model'].model
    torch.save(utl_param.state_dict(), args.source + '.pth')
    own_state = utl_param.state_dict()
    print(len(own_state))

    numpy_param = OrderedDict()
    for name in own_state:
        numpy_param[name] = own_state[name].data.cpu().numpy()
    print(len(numpy_param))
    with open(args.source + '_numpy_param.pkl', 'wb') as fw:
        pickle.dump(numpy_param, fw)