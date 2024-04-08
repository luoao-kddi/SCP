import multiprocessing
import subprocess
import sys

if __name__ == '__main__':
    cmd = sys.argv[2:]
    processes = []
    splits = int(sys.argv[1])
    print('start:', cmd)
    for i in range(splits):
        process = multiprocessing.Process(target=lambda c: subprocess.call(c), args=(cmd + ['--parts', f'{i}/{splits}'],))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    print('finished:', cmd)
