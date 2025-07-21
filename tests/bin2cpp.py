import numpy as np
import os

def bin_array_test(data_loc: str):
    grid = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))
    grid.tofile(os.path.join(data_loc,'full_grid.bin'))
    with open(os.path.join(data_loc ,'shape.txt'), 'w') as f:
        f.write(f"{grid.shape[0]} {grid.shape[1]} {grid.shape[2]}\n")
    print(grid)

def main():
    data_loc = r"H:\bin_array_test"
    bin_array_test(data_loc)

if __name__ == "__main__":
    main()