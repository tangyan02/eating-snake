from matplotlib import pyplot as plt

f = open("logs/returns_10000.log", "r")
y = f.readline().split(" ")

x = my_array = list(range(1, len(y) + 1))
plt.plot(x, y)
plt.show()