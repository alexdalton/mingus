sum = 0.0
for i in range(512):
    count = 0.0
    stride = 1
    while stride < 1024:
        if ((i + 1) * stride * 2 - 1) < 1024:
            count += 1.0
        stride *= 2
    print i, count
    sum += count

print sum / 512.0
