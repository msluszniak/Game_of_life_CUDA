from random import randint

width = height = num_threads = 50
with open("random.txt", "w") as file:
    file.write(f"{num_threads}\n")
    file.write(f"{width} {height}\n")
    num_live = randint(int(0.2 * width * height), int(0.3 * width * height))
    file.write(f"{num_live}\n")
    for _ in range(num_live):
        x, y = randint(0, width-1), randint(0, height-1)
        file.write(f"{x} {y}\n")
