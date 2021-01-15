# Imports
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from peaks import Peaks
import heartbreaker as hb
from composite_peaks import CompositePeaks
from matplotlib.widgets import Button, CheckButtons, Slider

class HeartbeatIntervalFinder(object):
    """
    Cursor for editing heartbeat signals for use of verification
    """
    def __init__(self, files,
                       folder_name = "",
                       dosage = 0,
                       file_number = 1,
                       area_around_echo_size = 240,
                       use_intervals = False,
                       preloaded_signal = False,
                       save_signal = False):

        super(HeartbeatIntervalFinder, self).__init__()
        # Load Arguments
        self.files            = files
        self.folder_name      = folder_name
        self.dosage           = dosage
        self.file_number      = file_number
        self.file_name        = files[folder_name][dosage][file_number]["file_name"]

        self.use_intervals            = use_intervals
        self.area_around_echo_size    = area_around_echo_size
        self.preloaded_signal         = preloaded_signal

        # Save signals
        if save_signal:
            self.save_signals()

        # Load Signals and 2D Echo Time
        self.load_signals()

        # Determine Orginal bounds
        self.initialize_bounds()

        # Plot signals
        self.plot_signals()       

    def clip_signals(self):
        max_time = max(self.time)
        min_time = min(self.time)

        if max_time - min_time < self.area_around_echo_size:
            interval = range(np.where(self.time == min_time)[0][0], np.where(self.time == max_time)[0][0])

        elif self.echo_time - self.area_around_echo_size/2 < min_time:
            interval = range(np.where(self.time == min_time)[0][0], int(np.where(self.time == min_time + self.area_around_echo_size)[0][0]))

        elif self.echo_time + self.area_around_echo_size/2 > max_time:
            interval = range(int(np.where(self.time == (max_time - self.area_around_echo_size))[0][0]), np.where(self.time == max_time)[0][0])

        else:
            interval = range(int(np.where(self.time == (self.echo_time - self.area_around_echo_size/2))[0][0]), int(np.where(self.time == (self.echo_time + self.area_around_echo_size/2))[0][0]))

        self.interval_near_echo = interval
        self.time   = self.time[interval]
        self.signal = self.signal[interval]
        self.seis   = self.seis[interval]
        self.phono  = self.phono[interval]

        self.signal = hb.bandpass_filter(time   = self.time, 
                                        signal  = self.signal,
                                        freqmin = 59, 
                                        freqmax = 61)

        self.seis = hb.bandpass_filter(time     = self.time, 
                                        signal  = self.seis,
                                        freqmin = 59, 
                                        freqmax = 61)

        self.phono = hb.bandpass_filter(time    = self.time, 
                                        signal  = self.phono,
                                        freqmin = 59, 
                                        freqmax = 61)

        self.signal = hb.lowpass_filter(time = self.time, 
                                        signal = self.signal,
                                        cutoff_freq = 50)

        self.seis = hb.lowpass_filter(time = self.time, 
                                    signal = self.seis,
                                    cutoff_freq = 50)

    def initialize_bounds(self):
        if self.use_intervals:
            self.lower_bound = self.files[self.folder_name][self.dosage][self.file_number]["intervals"][1][0]
            self.upper_bound = self.files[self.folder_name][self.dosage][self.file_number]["intervals"][1][1]

        else:
            max_time = max(self.time)
            min_time = min(self.time)

            if (max_time - min_time) < 20:
                self.lower_bound = min_time
                self.upper_bound = max_time

            elif (self.echo_time - (20/2)) < min_time:
                self.lower_bound = min_time
                self.upper_bound = min_time + 20

            elif (self.echo_time + (20/2)) > max_time:
                self.lower_bound = max_time - 20
                self.upper_bound = max_time
                
            else:
                self.lower_bound = self.echo_time - (20/2)
                self.upper_bound = self.echo_time + (20/2)

    def plot_signals(self):
        # Create figure
        self.fig, self.ax = plt.subplots()

        self.ax.get_yaxis().set_visible(False)
        self.ax.set_xlabel("Time [s]")
        
        # Plot ECG, Phono and Seismo
        self.signal_line, = self.ax.plot(self.time, self.signal, label = "ECG", linewidth = 0.5, c = "b")
        self.seis_line, = self.ax.plot(self.time, self.seis,   label = "Seismo", linewidth = 0.5, c = "r")
        self.phono_line, = self.ax.plot(self.time, self.phono,  label = "Phono", linewidth = 0.5, c = "g")
        
        sig_min = min(self.signal)
        sig_max = max(self.signal)
        self.ax.set_xlim(self.time[0] - 0.1*(self.time[-1] - self.time[0]), self.time[-1] + 0.1*(self.time[-1] - self.time[0]))
        self.ax.set_ylim(sig_min - 0.1*(sig_max - sig_min), sig_max + 0.1*(sig_max - sig_min))

        # Echo Line
        signal_max = max(self.signal)
        signal_min = min(self.signal)
        self.echo_line = self.ax.axvline(self.echo_time,
                                             ymin = signal_min - abs(signal_max - signal_min),
                                             ymax = signal_max + abs(signal_max - signal_min),
                                             label = "2D Echo Time", c = "k", linewidth = 2)
        plt.legend(loc = "upper right")

        # Set endpoints
        self.bound_span   = self.ax.axvspan(self.lower_bound,
                                            self.upper_bound,
                                            facecolor='g', alpha=0.25)

        self.bound_text_height = -0.1
        self.lower_bound_text = self.ax.text(self.lower_bound, self.bound_text_height, transform = self.ax.get_xaxis_transform(),
                                            s = "Lower Bound\n" + str(self.lower_bound), fontsize=12, horizontalalignment = 'center')
        self.upper_bound_text = self.ax.text(self.upper_bound, self.bound_text_height, transform = self.ax.get_xaxis_transform(),
                                            s = "Upper Bound\n" + str(self.upper_bound), fontsize=12, horizontalalignment = 'center')
                                            

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
        self.b_save.on_clicked(self.save_intervals)
        
        # Add Line hide buttons
        self.ax.text(1.015, 0.97, transform = self.ax.transAxes,
                    s = "Hide Signal:", fontsize=12, horizontalalignment = 'left')
        ax_switch_signals = plt.axes([0.91, 0.7, 0.07, 0.15])
        self.b_switch_signals = CheckButtons(ax_switch_signals, ('ECG', 'Seismo', 'Phono'))

        self.b_switch_signals.on_clicked(self.switch_signal)

        # Add Sliders
        self.signal_amp_slider = Slider(plt.axes([0.91, 0.15, 0.01, 0.475]),
                                        label = "ECG\nA",
                                        valmin = 0.01,
                                        valmax = 10, 
                                        valinit = 1,
                                        orientation = 'vertical')
        self.signal_amp_slider.on_changed(self.switch_signal)
        self.signal_amp_slider.valtext.set_visible(False)

        self.seis_height_slider = Slider(plt.axes([0.93, 0.15, 0.01, 0.475]),
                                        label = "   Seis\nH",
                                        valmin = 1.5 * min(self.signal),
                                        valmax = 1.5 * max(self.signal), 
                                        valinit = 0,
                                        orientation = 'vertical')
        self.seis_height_slider.on_changed(self.switch_signal)
        self.seis_height_slider.valtext.set_visible(False)

        self.seis_amp_slider = Slider(plt.axes([0.94, 0.15, 0.01, 0.475]),
                                        label = "\nA",
                                        valmin = 0.01,
                                        valmax = 10, 
                                        valinit = 1,
                                        orientation = 'vertical')
        self.seis_amp_slider.on_changed(self.switch_signal)
        self.seis_amp_slider.valtext.set_visible(False)

        self.phono_height_slider = Slider(plt.axes([0.96, 0.15, 0.01, 0.475]),
                                        label = "    Phono\nH",
                                        valmin = 1.5 * min(self.signal),
                                        valmax = 1.5 * max(self.signal), 
                                        valinit = 0,
                                        orientation = 'vertical')
        self.phono_height_slider.on_changed(self.switch_signal)
        self.phono_height_slider.valtext.set_visible(False)

        self.phono_amp_slider = Slider(plt.axes([0.97, 0.15, 0.01, 0.475]),
                                        label = "A",
                                        valmin = .01,
                                        valmax = 10, 
                                        valinit = 1,
                                        orientation = 'vertical')
        self.phono_amp_slider.on_changed(self.switch_signal)
        self.phono_amp_slider.valtext.set_visible(False)

        # Add Clicking Actions
        self.fig.canvas.mpl_connect('button_press_event',   self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.off_click)

        # Maximize frame
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

        plt.show()

    def switch_signal(self, label):

        self.x = self.time
        self.y = [np.mean(self.signal)] * len(self.time)

        label = self.b_switch_signals.get_status()
        self.signal_line.set_data(self.time, self.signal_amp_slider.val * self.signal)
        self.seis_line.set_data(self.time, (self.seis_amp_slider.val * self.seis) + self.seis_height_slider.val)
        self.phono_line.set_data(self.time, (self.phono_amp_slider.val * self.phono) + self.phono_height_slider.val)


        if label[0]: # ECG
            self.signal_line.set_linewidth(0)
            
        else: 
            self.signal_line.set_linewidth(0.5)
            

        if label[1]: # Seismo
            self.seis_line.set_linewidth(0)

        else:
            self.seis_line.set_linewidth(0.5)

        if label[2]: # Phono
            self.phono_line.set_linewidth(0)

        else:
            self.phono_line.set_linewidth(0.5)

        # Update green shaded region
        self.bound_span.remove()
        self.bound_span = self.ax.axvspan(self.lower_bound, self.upper_bound, facecolor='g', alpha=0.25)

        self.lower_bound_text.set_position((self.lower_bound, self.bound_text_height))
        self.lower_bound_text.set_text("Lower Bound\n" + str(self.lower_bound))

        self.upper_bound_text.set_position((self.upper_bound, self.bound_text_height))
        self.upper_bound_text.set_text("Upper Bound\n" + str(self.upper_bound))

        # Update x and y lim
        if self.new_bounds:
            sig_min = min(self.signal)
            sig_max = max(self.signal)
            self.ax.set_xlim(self.time[0] - 0.1*(self.time[-1] - self.time[0]), self.time[-1] + 0.1*(self.time[-1] - self.time[0]))
            self.ax.set_ylim(sig_min - 0.1*(sig_max - sig_min), sig_max + 0.1*(sig_max - sig_min))

            self.new_bounds = False

        self.fig.canvas.draw()
            
    def on_click(self, event):
        threshold = 2
        self.update_point = None
        self.new_bounds = False
        # Make sure a click happened inside the subplot
        if (event.xdata is not None) and (str(type(event.inaxes)) == "<class 'matplotlib.axes._subplots.AxesSubplot'>"):

            if abs(self.lower_bound - event.xdata) < threshold:
                self.lx.set_color('g')
                self.ly.set_color('g')

                self.lx.set_linewidth(1)
                self.ly.set_linewidth(1)
                
                self.fig.canvas.draw()
                
                self.update_point = "lower_bound"

            elif abs(self.upper_bound - event.xdata) < threshold:
                self.lx.set_color('g')
                self.ly.set_color('g')

                self.lx.set_linewidth(1)
                self.ly.set_linewidth(1)
                
                self.fig.canvas.draw()
                
                self.update_point = "upper_bound"

    def off_click(self, event):
        self.lx.set_color('k')
        self.ly.set_color('k')

        self.lx.set_linewidth(0.2)
        self.ly.set_linewidth(0.2)

        if event.xdata is not None:
            if self.update_point == "lower_bound":

                self.lower_bound = max(self.time[0], round(event.xdata, 1))

            if self.update_point == "upper_bound":
                
                self.upper_bound = min(self.time[-1], round(event.xdata, 1))

            lower = min(self.lower_bound, self.upper_bound)
            upper = max(self.lower_bound, self.upper_bound)
            self.lower_bound = lower
            self.upper_bound = upper

            # Update on signal
            self.switch_signal(self.b_switch_signals.get_status())

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
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
        self.new_bounds = True
        self.update_plot() 

    def prev(self, event):
        self.dosage -= 10
        if self.dosage < 0:
            self.dosage = 40
        self.new_bounds = True
        self.update_plot()

    def save_intervals(self, event):
        # Save bounds
        save_filename = "data/Derived/intervals/Interval_Dict_" + self.folder_name
        self.files[self.folder_name][self.dosage][self.file_number]["intervals"][1][0] = self.lower_bound
        self.files[self.folder_name][self.dosage][self.file_number]["intervals"][1][1] = self.upper_bound

        # Save Bounds
        with open(save_filename + '.pkl', 'wb') as output:
            pickle.dump(self.files, output, pickle.HIGHEST_PROTOCOL)
        print("Saved")

    def save_signals(self):

        print("Saving Signals...")
        for dosage in self.files[self.folder_name]:
            # Load data
            self.time, self.signal, self.seis, _, self.phono, _ = hb.load_file_data(files = self.files, 
                                                                                    folder_name = self.folder_name, 
                                                                                    dosage = dosage,
                                                                                    file_number = self.file_number)
            
            # Load Echo Time
            self.echo_time = self.files[self.folder_name][dosage][self.file_number]["echo_time"]

            # Clip signal size about echo time
            self.clip_signals()

            # Save File Name
            save_file_name = self.folder_name + "_d" + str(dosage)
            assert not os.path.isfile('data/Derived/signals/time_'  + save_file_name + '.csv'), "Saved signals already exist, please delete before saving new signals."

            # Save Signals
            np.savetxt('data/Derived/signals/time_'  + save_file_name + '.csv', self.time, delimiter=',')
            np.savetxt('data/Derived/signals/signal_'+ save_file_name + '.csv', self.signal, delimiter=',')
            np.savetxt('data/Derived/signals/seis_'  + save_file_name + '.csv', self.seis, delimiter=',')
            np.savetxt('data/Derived/signals/phono_' + save_file_name + '.csv', self.phono, delimiter=',')

            print("\tDosage " + str(dosage) + " done")

        print("...Done Saving Signals")

        # Use these saved files from now on
        self.preloaded_signal = True
        
    def load_signals(self):

        if self.preloaded_signal:
            
            save_file_name = self.folder_name + "_d" + str(self.dosage)
            self.time   = np.loadtxt('data/Derived/signals/time_' + save_file_name + '.csv', delimiter=',')
            self.signal = np.loadtxt('data/Derived/signals/signal_' + save_file_name + '.csv', delimiter=',')
            self.seis   = np.loadtxt('data/Derived/signals/seis_' + save_file_name + '.csv', delimiter=',')
            self.phono  = np.loadtxt('data/Derived/signals/phono_' + save_file_name + '.csv', delimiter=',')

        else:
            self.time, self.signal, self.seis, _, self.phono, _ = hb.load_file_data(files = self.files, 
                                                                                folder_name = self.folder_name, 
                                                                                dosage = self.dosage,
                                                                                file_number = self.file_number)

            # Clip Signals
            self.clip_signals()


        self.echo_time = self.files[self.folder_name][self.dosage][self.file_number]["echo_time"]

    def update_plot(self):
        # Display Loading Screen
        self.dosage_text.set_text("Loading")
        self.fig.canvas.draw()

        # Update index
        self.folder_text.set_text("Folder: " + self.folder_name)
        self.dosage_text.set_text("Dosage: " + str(self.dosage))
        self.file_name_text.set_text("File: " + self.files[self.folder_name][self.dosage][self.file_number]["file_name"])
        
        # Load composite signals
        self.load_signals()

        # Update Echo Line
        self.echo_line.set_xdata(self.echo_time)

        # Find new orginal bounds
        self.initialize_bounds()

        # Update lines
        self.switch_signal(self.b_switch_signals.get_status())


    


