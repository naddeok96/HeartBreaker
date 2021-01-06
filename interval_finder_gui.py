# Imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from peaks import Peaks
import heartbreaker as hb
from composite_peaks import CompositePeaks
from matplotlib.widgets import Button, RadioButtons

class HeartbeatIntervalFinder(object):
    """
    Cursor for editing heartbeat signals for use of verification
    """
    def __init__(self, files,
                       folder_name = "",
                       dosage = "",
                       file_number = "",
                       interval_number = "",
                       interval_size = 240):

        super(HeartbeatIntervalFinder, self).__init__()
        # Load Data
        self.files           = files
        self.folder_name     = folder_name
        self.dosage          = dosage
        self.file_number     = file_number
        self.interval_number = interval_number
        self.interval_size   = interval_size

        self.time, self.signal, self.seis, _, self.phono, _ = hb.load_file_data( files = files, 
                                                                        folder_name = folder_name, 
                                                                        dosage = dosage,
                                                                        file_number = file_number,
                                                                        interval_number = None,
                                                                        preloaded_signal = False, 
                                                                        save_signal = False)

        self.echo_time = files[folder_name][dosage][file_number]["echo_time"]

        # Clip signal size about echo time
        self.clip_signals()

        # Determine Orginal bound
        self.initialize_bounds()

        # Plot signals
        self.plot_signals()       

    def clip_signals(self):
        max_time = max(self.time)
        min_time = min(self.time)

        if max_time - min_time < self.interval_size:
            interval = range(np.where(self.time == min_time)[0][0], np.where(self.time == max_time)[0][0])

        elif self.echo_time - self.interval_size/2 < min_time:
            interval = range(np.where(self.time == min_time)[0][0], int(np.where(self.time == min_time + self.interval_size)[0][0]))

        elif self.echo_time + self.interval_size/2 > max_time:
            interval = range(int(np.where(self.time == (max_time - self.interval_size))[0][0]), np.where(self.time == max_time)[0][0])

        else:
            interval = range(int(np.where(self.time == (self.echo_time - self.interval_size/2))[0][0]), int(np.where(self.time == (self.echo_time + self.interval_size/2))[0][0]))

        self.interval_near_echo = interval
        self.time   = self.time[interval]
        self.signal = self.signal[interval]
        self.seis   = self.seis[interval]
        self.phono  = self.phono[interval]

    def initialize_bounds(self):
        max_time = max(self.time)
        min_time = min(self.time)

        if max_time - min_time < 20:
            self.bounds = [[np.searchsorted(self.time, min_time), np.searchsorted(self.time, max_time)]]

        elif self.echo_time - (20/2) < min_time:
            self.bounds = [[np.searchsorted(self.time, min_time), np.searchsorted(self.time, min_time + 20)]]

        elif self.echo_time + 20/2 > max_time:
            self.bounds = [[np.searchsorted(self.time, max_time - 20), np.searchsorted(self.time, max_time)]]
            
        else:
            self.bounds = [[np.searchsorted(self.time, self.echo_time - (20/2)), np.searchsorted(self.time, self.echo_time + (20/2))]]

    def plot_signals(self):
        # Create figure
        self.fig, self.ax = plt.subplots()
        
        # Plot ECG, Phono and Seismo
        self.signal_line, = self.ax.plot(self.time, self.signal, linewidth = 1, c = "b")
        
        self.ax.set_xlim(self.time[0] - 0.1*len(self.time), self.time[-1] + 0.1*len(self.time))

        # Echo Line
        signal_max = max(self.signal)
        signal_min = min(self.signal)
        self.echo_line = self.ax.axvline(self.echo_time,
                                             ymin = signal_min - abs(signal_max - signal_min),
                                             ymax = signal_max + abs(signal_max - signal_min))

        # Set endpoints
        self.bound_points = self.ax.scatter([self.time[self.bounds]],
                                            [self.signal[self.bounds]],
                                             c = '#ff7f0e')

        # Initalize axes and data points
        self.x = self.time
        self.y = self.signal

        # Cross hairs
        self.lx = self.ax.axhline(color='k', linewidth=0.2)  # the horiz line
        self.ly = self.ax.axvline(color='k', linewidth=0.2)  # the vert line

        # Add data
        left_shift = 0.45
        start = 0.96
        space = 0.04
        self.folder_text = self.ax.text(0.01, start, transform = self.ax.transAxes,
                                            s = "Folder: " + self.folder_name, fontsize=12, horizontalalignment = 'left')
        self.dosage_text = self.ax.text(0.61 - left_shift, 1.1 - space, transform = self.ax.transAxes,
                    s = "Dosage: " + str(self.dosage), fontsize=12, horizontalalignment = 'left')
        self.file_name_text = self.ax.text(0.01, start - space, transform = self.ax.transAxes,
                    s = "File: " + self.files[self.folder_name][self.dosage][self.file_number]["file_name"], fontsize=12, horizontalalignment = 'left')
        self.interval_text = self.ax.text(0.01, start - 2*space, transform = self.ax.transAxes,
                    s = "File #: " + str(self.interval_number), fontsize=12, horizontalalignment = 'left')

        # Add index buttons
        ax_prev = plt.axes([0.575 - left_shift, 0.9, 0.1, 0.075])
        self.bprev = Button(ax_prev, 'Previous')
        self.bprev.on_clicked(self.prev)
        
        ax_next = plt.axes([0.8 - left_shift, 0.9, 0.1, 0.075])
        self.b_next = Button(ax_next, 'Next')
        self.b_next.on_clicked(self.next)

        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)

        # Add Save Button
        ax_save = plt.axes([0.8, 0.9, 0.1, 0.075])
        self.b_save = Button(ax_save, 'Save')
        self.b_save.on_clicked(self.save)
        
        # Add Line buttons
        self.ax.text(1.015, 0.97, transform = self.ax.transAxes,
                    s = "Snap on to:", fontsize=12, horizontalalignment = 'left')
        # left, bottom, width, height
        ax_switch_signals = plt.axes([0.91, 0.7, 0.075, 0.15])
        self.b_switch_signals = RadioButtons(ax_switch_signals, ('ECG', 'Seismo', 'Phono'))
        for c in self.b_switch_signals.circles:
            c.set_radius(0.05)

        self.b_switch_signals.on_clicked(self.switch_signal)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.off_click)
        
        plt.show()

    def switch_signal(self, label):

        if label == 'ECG':
            self.x = self.time
            self.y = self.signal

            self.signal_line.set_data(self.time, self.signal)
            self.bound_points.set_offsets([self.time[self.bounds], 
                                           self.signal[self.bounds]])

        if label == 'Seismo':
            self.x = self.time
            self.y = self.seis

            self.signal_line.set_data(self.time, self.seis)
            self.bound_points.set_offsets([self.time[self.bounds], 
                                           self.seis[self.bounds]])

        if label == 'Phono':
            self.x = self.time
            self.y = self.phono

            self.signal_line.set_data(self.time, self.phono)
            self.bound_points.set_offsets([self.time[self.bounds], 
                                           self.phono[self.bounds]])

        self.ax.set_xlim(self.time[0] - 0.1*len(self.time), self.time[-1] + 0.1*len(self.time))

    def off_click(self, event):
        self.lx.set_color('k')
        self.ly.set_color('k')

        self.lx.set_linewidth(0.2)
        self.ly.set_linewidth(0.2)

        if event.xdata is not None:
            if self.update_point is not None:

                # Update bounds
                self.bounds[0][np.searchsorted(self.bounds[0], self.update_point)] = int(self.x[min(np.searchsorted(self.x, event.xdata), len(self.x) - 1)])
                self.bounds[0].sort()

                # Update on signal
                self.switch_signal(self.b_switch_signals.value_selected)

                # Update interval in files
                self.files[self.folder_name][self.dosage][self.file_number]["intervals"][self.interval_number] = self.bounds

                
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_click(self, event):
        threshold = 40
        self.update_point = None

        if event.xdata is not None:
            for bound in self.bounds[0]:
                if bound - event.xdata < threshold:
                    self.lx.set_color('#d62728')
                    self.ly.set_color('#d62728')

                    self.lx.set_linewidth(1)
                    self.ly.set_linewidth(1)
                    
                    self.fig.canvas.draw()
                    
                    self.update_point = np.int64(bound)

                    continue
          
    def mouse_move(self, event):
        # If nothing happened do nothing
        if not event.inaxes:
            return

        # Update x data point
        x = event.xdata

        # Lock to closest x coordinate on signal
        indx = min(np.searchsorted(self.x, x), len(self.x) - 1)
        x = self.x[indx]
        y = self.y[indx]

        # Update the crosshairs
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        # Draw everything
        self.ax.figure.canvas.draw()

    def next(self, event):
        self.dosage += 10
        if self.dosage > 40:
            self.dosage = 0
        self.update_plot() 

    def prev(self, event):
        self.dosage -= 10
        if self.dosage < 0:
            self.dosage = 40
        self.update_plot()

    def save(self, event):
        # Get File Name
        save_file_name = "Interval_Dict_" + self.folder_name + "_d" + str(self.dosage) + "_" + self.file_name + "_i" + str(self.interval_number)

        # Save
        with open(save_filename + '.pkl', 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print("Saved")

    def update_plot(self):
        # Display Loading Screen
        self.dosage_text.set_text("Loading")
        self.fig.canvas.draw()

        # Update index
        self.folder_text.set_text("Folder: " + self.folder_name)
        self.dosage_text.set_text("Dosage: " + str(self.dosage))
        self.file_name_text.set_text("File: " + self.files[self.folder_name][self.dosage][self.file_number]["file_name"])
        self.interval_text.set_text("File #: " + str(self.interval_number))

        # Load composite signals
        os.chdir("../..")
        self.time, self.signal, self.seis, _, self.phono, _ = hb.load_file_data( files = self.files, 
                                                                                folder_name = self.folder_name, 
                                                                                dosage = self.dosage,
                                                                                file_number = self.file_number,
                                                                                interval_number = None,
                                                                                preloaded_signal = False, 
                                                                                save_signal = False)

        # Load Echo Time
        self.echo_time = self.files[self.folder_name][self.dosage][self.file_number]["echo_time"]
        self.echo_line.set_xdata(self.echo_time)

        # Clip Signals
        self.clip_signals()

        # Find new orginal bounds
        self.initialize_bounds()

        # Update lines
        self.switch_signal(self.b_switch_signals.value_selected)

        self.fig.canvas.draw()


