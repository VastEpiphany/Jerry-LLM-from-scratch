import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim',type=int,default=768)
    parser.add_argument('--hidden_dim',type=int,default=1024)
    parser.add_argument('--multiple_of',type=int,default=4)
    parser.add_argument('--norm_eps',type=float,default=1e-5)
    parser.add_argument('--dropout',type=float,default=0.0)
    args = parser.parse_args()
    return args