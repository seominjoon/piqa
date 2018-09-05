import numpy as np
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', default=10, type=int)
    parser.add_argument('--num_vecs', default=10000, type=int)
    parser.add_argument('--dim', default=1024, type=int)
    args = parser.parse_args()

    query = np.random.randn(args.dim)

    # numpy experiment
    docs = [np.random.randn(args.num_vecs, args.dim) for _ in range(args.num_iters)]

    start_time = time.time()
    for i, doc in enumerate(docs):
        ans = np.argmax(np.matmul(doc, np.expand_dims(query, -1)), 0)
    duration = time.time() - start_time
    speed = args.num_vecs * args.num_iters / duration
    print('numpy: %.3f ms per %d vecs of %dD, or %d vecs/s' % (
    duration * 1000 / args.num_iters, args.num_vecs, args.dim, speed))
