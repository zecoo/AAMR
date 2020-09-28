import os
import argparse

svc_arr = ['paymentservice', 'currencyservice', 'cartservice', 'productcatalogservice', 'checkoutservice', 'recommendationservice', 'frontend']

def parse_args():
    """Parse the args."""
    parser = argparse.ArgumentParser(
        description='Root cause analysis for microservices')

    parser.add_argument('--replicas', type=int, required=False,
                        default=2,
                        help='replica num')

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    replica_num = args.replicas

    for svc in svc_arr:
        os.system('kubectl scale deployment -n hipster %s --replicas=%d' %(svc, replica_num))
