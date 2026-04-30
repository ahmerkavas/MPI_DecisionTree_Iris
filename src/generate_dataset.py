input_file = "dataset/iris.csv"
output_file = "dataset/iris_large.csv"

with open(input_file, "r") as f:
    lines = f.readlines()

header = lines[0]
data = lines[1:]

multiplier = 100  # 150 → 15000 rows

with open(output_file, "w") as f:
    f.write(header)
    for _ in range(multiplier):
        f.writelines(data)

print("iris_large.csv created successfully")