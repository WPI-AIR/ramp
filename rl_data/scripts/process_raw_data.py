#!/usr/bin/env python
import os
import rosbag
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## open bag
bag = rosbag.Bag("/home/kai/data/ramp/simulation_env_data/2018-01-10_16:58:39/raw_data/1.bag")
file_name = 'actual_exe_time.csv'



## get directory
cur_dir = os.path.join(os.path.dirname(__file__) + '/')
file_h = open(cur_dir + file_name, "a")

exe_times = np.array([])
A = np.array([])
D = np.array([])
Qk = np.array([])
window_size = 1
print("Processing data......")
for topic, msg, t in bag.read_messages(topics=["/ramp_collection_exe_time", "/ramp_collection_si_act"]):
    topic = topic.decode()
    if topic == "/ramp_collection_exe_time":
        ## process execution time
        exe_times = np.append(exe_times, msg.data)
    elif topic == "/ramp_collection_si_act":
        ## process action
        A = np.append(A, msg.data[0])
        D = np.append(D, msg.data[1])
        Qk = np.append(Qk, msg.data[2])
    else:
        ## never enter here
        assert False

    # file_h.write('{}\n'.format(msg.data))

print("Processing completed!")

## save data
df = pd.DataFrame()
df['actual_exe_time'] = exe_times
df.to_csv(file_name, index=False)

## plot
fig_times = plt.figure(figsize = (15, 10))

times_smoothed = pd.Series(exe_times).rolling(window_size, min_periods = window_size).mean()
plt.subplot(1, 1, 1)
plt.plot(times_smoothed, 'ro')
plt.xlabel("execution")
plt.ylabel("durations/s (smoothed over window size {})".format(window_size))

# plt.subplot(2, 2, 2)
# plt.plot(A)
# plt.xlabel("execution")
# plt.ylabel("A")

# plt.subplot(2, 2, 3)
# plt.plot(D)
# plt.xlabel("execution")
# plt.ylabel("D")

# plt.subplot(2, 2, 4)
# plt.plot(Qk)
# plt.xlabel("execution")
# plt.ylabel("Qk")

## close bag
bag.close()
file_h.close()

plt.show(fig_times)