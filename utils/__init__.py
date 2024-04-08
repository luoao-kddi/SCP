import os

def get_psnr(tmp_test_file):
    with open(tmp_test_file) as f:
        c = f.readlines()
    f.close()
    for i in range(len(c)):
        if c[i].startswith('3.'):
            d1 = float(c[i+2].split(' ')[-1])
            try:
                d2 = float(c[i+4].split(' ')[-1])
            except Exception as e:
                d2 = 0.
            break
    os.remove(tmp_test_file)
    return d1, d2