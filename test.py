import numpy as np

b = np.random.choice(243, 4, False)
b = np.sort(b)

c = np.random.choice(17, 4, False)
c = np.sort(c)
print(b, c)
for i in range(4):
    print(np.random.choice(3) + 3)


def random_romove(frames, pitchs):

    start_frame = np.random.choice(243//5, pitchs, False)
    start_frame = np.sort(start_frame) * 5
    end_frame = start_frame + 25
    for i in range(pitchs - 1):
        if end_frame[i] > start_frame[i+1]:
            end_frame[i] = start_frame[i+1]
    if (end_frame[-1] > 243-1):
        end_frame[-1] = 243-1
    print(start_frame)
    print(end_frame)
    print(np.arange(243)[start_frame[0]:end_frame[0]])
     
    invalid_point_number = np.random.choice(17, np.random.choice(3) + 4, False)
    invalid_point_number = np.sort(c)
    print(invalid_point_number)


random_romove(25, 4)
