import time
import matplotlib.pyplot as plt
import sys

def read_positions(): # parses out positions from a file containing 2d coordinates of cities
    coordinates = {}
    with open(sys.argv[1]) as dataFile:
        for line in dataFile:
            line = list(map(int, line.split()))
            key, value = line[0], line[1:]
            coordinates[key] = value
    return coordinates

if __name__ == '__main__':
    coordinates = read_positions()
    #print(coordinates)
    with open(sys.argv[2]) as f:
        for line in f:
            positions = []
            for city in line.split(", "):
                positions.append(int(city))

            x_coords = []
            y_coords = []
            for i in positions:
                x_coords.append(coordinates[i][0])
                y_coords.append(coordinates[i][1])

            plt.plot(x_coords, y_coords, '-o', label='coordinates of cities')
            plt.xlabel("x-coordinate")
            plt.ylabel("y-coordinate")
            plt.legend()
            plt.show()
            time.sleep(1)























