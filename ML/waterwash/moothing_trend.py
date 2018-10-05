import scipy.signal as signal
import matplotlib.pyplot as plt

# First, design the Buterworth filter
N  = 2    # Filter order
Wn = 0.01 # Cutoff frequency
B, A = signal.butter(N, Wn, output='ba')

# Second, apply the filter
tempf = signal.filtfilt(B,A, temp)

# Make plots
fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.plot(date,temp, 'b-')
plt.plot(date,tempf, 'r-',linewidth=2)
plt.ylabel("Temperature (oC)")
plt.legend(['Original','Filtered'])
plt.title("Temperature from LOBO (Halifax, Canada)")
ax1.axes.get_xaxis().set_visible(False)

ax1 = fig.add_subplot(212)
plt.plot(date,temp-tempf, 'b-')
plt.ylabel("Temperature (oC)")
plt.xlabel("Date")
plt.legend(['Residuals'])
