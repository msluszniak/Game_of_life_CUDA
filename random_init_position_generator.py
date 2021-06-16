from random import randint

width = height = num_threads = 50
with open("random.txt", "w") as file:
    file.write(f"{num_threads}\n")
    file.write(f"{width} {height}\n")
    num_live = randint(int(0.2 * 50 * 50), int(0.3 * 50 * 50))
    file.write(f"{num_live}\n")
    for _ in range(num_live):
        x, y = randint(0, 49), randint(0, 49)
        file.write(f"{x} {y}\n")
