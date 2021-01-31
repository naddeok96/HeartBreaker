import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from peaks import Peaks
import heartbreaker as hb
from matplotlib.widgets import Button, RadioButtons, Slider, CheckButtons

class PeakHeartbeatVerifier(object):
    """
    Cursor for editing heartbeat signals for use of verification
    """

    def __init__(self, peaks, 
                       index = 0,
                       folder_name = "",
                       dosage = "",
                       file_name = "",
                       interval_number = ""):

        super(PeakHeartbeatVerifier, self).__init__()
        # Save Peaks
        self.peaks = peaks
        self.index = index

        self.folder_name = folder_name
        self.dosage      = dosage
        self.file_name   = file_name
        self.interval_number = interval_number
        self.update_point    = None

        # Plot signals
        self.plot_signals()       

    def plot_signals(self):
        # Create figure
        self.fig, self.ax = plt.subplots()
        self.signal = hb.normalize(self.peaks.signal[range(self.peaks.R.data[self.index], self.peaks.R.data[self.index + 1])])

        # Determine what cutoff freq to use
        cutoff_freq = 15 if np.mean(np.diff(self.peaks.R.data)) > 2500 else 10

        # Pass through a Low pass
        smoothed_signal = hb.lowpass_filter(signal = self.signal,
                                            cutoff_freq = cutoff_freq)

        # Calculate first derivative
        self.first, _ = hb.get_derivatives(smoothed_signal)

        # Plot ECG, Phono and Seismo
        self.signal_line, = self.ax.plot(range(len(self.signal)), self.signal, linewidth = 1, c = "b", label = "ECG")

        self.first_line, = self.ax.plot(range(len(self.signal)), 1 + 5*self.first,
                                        '--', linewidth = 0.5, c = 'k', label = "ECG 1st Derv.")

        self.ax.set_xlim(0, len(self.signal))

        sig_min = min(self.signal)
        sig_max = max(self.signal)

        self.ax.set_ylim(sig_min - 0.2*(sig_max - sig_min), sig_max + 0.2*(sig_max - sig_min))
        plt.legend(loc='upper right')

        # # T Peak
        # self.T_point = self.ax.scatter(self.peaks.T.data[self.index] - self.peaks.R.data[self.index], self.signal[self.peaks.T.data[self.index] - self.peaks.R.data[self.index]], c = '#9467bd')
        # self.T_text  = self.ax.text(self.peaks.T.data[self.index] - self.peaks.R.data[self.index], self.signal[self.peaks.T.data[self.index] - self.peaks.R.data[self.index]] + 0.2, "T", fontsize=9, horizontalalignment = 'center')

        # # T''max Peak
        self.dT_point = self.ax.scatter(self.peaks.dT.data[self.index] - self.peaks.R.data[self.index], self.signal[self.peaks.dT.data[self.index] - self.peaks.R.data[self.index]], c = '#2ca02c')
        self.dT_text  = self.ax.text(self.peaks.dT.data[self.index] - self.peaks.R.data[self.index], self.signal[self.peaks.dT.data[self.index] - self.peaks.R.data[self.index]] + 0.2, "T'max", fontsize=9, horizontalalignment = 'center')

        # # T''max Peak
        # self.ddT_point = self.ax.scatter(self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index], self.signal[self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index]], c = '#2ca02c')
        # self.ddT_text  = self.ax.text(self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index], self.signal[self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index]] + 0.2, "T''max", fontsize=9, horizontalalignment = 'center')

        # Initalize axes and data points
        self.x = range(len(self.signal))
        self.y = self.signal

        # Cross hairs
        self.lx = self.ax.axhline(color='k', linewidth=0.2)  # the horiz line
        self.ly = self.ax.axvline(color='k', linewidth=0.2)  # the vert line

        # Add data
        left_shift = 0.45
        start = 0.96
        space = 0.04
        self.ax.text(0.01, start, transform = self.ax.transAxes,
                    s = "Folder: " + self.folder_name, fontsize=12, horizontalalignment = 'left')
        self.ax.text(0.01, start - space, transform = self.ax.transAxes,
                    s = "Dosage: " + str(self.dosage), fontsize=12, horizontalalignment = 'left')
        self.i_text = self.ax.text(0.60 - left_shift, 1.1 - space, transform = self.ax.transAxes,
                                    s = "Heartbeat: " + str(self.index + 1) + "/" + str(len(self.peaks.R.data) - 1), fontsize=12, horizontalalignment = 'left')

        # Add index buttons
        ax_prev = plt.axes([0.575 - left_shift, 0.9, 0.1, 0.075])
        self.bprev = Button(ax_prev, 'Previous')
        self.bprev.on_clicked(self.prev)
        
        ax_next = plt.axes([0.8 - left_shift, 0.9, 0.1, 0.075])
        self.b_next = Button(ax_next, 'Next')
        self.b_next.on_clicked(self.next)

        self.fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.off_click)

        # Add Sliders
        start = 0.91
        slider_width = 0.0075
        slider_height = 0.47

        self.signal_amp_slider = Slider(plt.axes([start, 0.15, slider_width, slider_height]),
                                                label = "ECG\n\nA",
                                                valmin = 0.01,
                                                valmax = 10, 
                                                valinit = 1,
                                                orientation = 'vertical')
        self.signal_amp_slider.label.set_size(8)
        self.signal_amp_slider.on_changed(self.switch_signal)
        self.signal_amp_slider.valtext.set_visible(False)

        self.first_height_slider = Slider(plt.axes([start + 2*slider_width, 0.15, slider_width, slider_height]),
                                                    label = "   2nd\n    Derv.\nH",
                                                    valmin = 1.5 * min(self.signal),
                                                    valmax = 1.5 * max(self.signal), 
                                                    valinit = 0,
                                                    orientation = 'vertical')
        self.first_height_slider.label.set_size(8)
        self.first_height_slider.on_changed(self.switch_signal)
        self.first_height_slider.valtext.set_visible(False)

        self.first_amp_slider = Slider(plt.axes([start + 3*slider_width, 0.15, slider_width, slider_height]),
                                        label = "\nA",
                                        valmin = 0.01,
                                        valmax = 10, 
                                        valinit = 1,
                                        orientation = 'vertical')
        self.first_amp_slider.label.set_size(8)
        self.first_amp_slider.on_changed(self.switch_signal)
        self.first_amp_slider.valtext.set_visible(False)

        # Maximize frame
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

        plt.show()

    def switch_signal(self, label):

        # Update Lines
        self.signal_line.set_data(range(len(self.signal)), self.signal_amp_slider.val * self.signal)
        self.first_line.set_data(range(len(self.signal)), (self.first_amp_slider.val * 5* self.first) + self.first_height_slider.val + 1)
        
        # # T Peak
        # self.T_point.set_offsets((self.peaks.T.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.T.data[self.index] - self.peaks.R.data[self.index]]))
        # self.T_text.set_position((self.peaks.T.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.T.data[self.index] - self.peaks.R.data[self.index]] + 0.2))

        # T'max Peak
        self.dT_point.set_offsets((self.peaks.dT.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.dT.data[self.index] - self.peaks.R.data[self.index]]))
        self.dT_text.set_position((self.peaks.dT.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.dT.data[self.index] - self.peaks.R.data[self.index]] + 0.2))


        # # T''max Peak
        # self.ddT_point.set_offsets((self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index]]))
        # self.ddT_text.set_position((self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index]] + 0.2))

        self.y = self.signal_amp_slider.val * self.signal
        self.fig.canvas.draw()

    def off_click(self, event):
        self.lx.set_color('k')
        self.ly.set_color('k')

        self.lx.set_linewidth(0.2)
        self.ly.set_linewidth(0.2)

        if event.xdata is not None:
            if self.update_point == "T":
                self.T_point.set_offsets((int(event.xdata), self.signal_amp_slider.val * self.signal[int(event.xdata)]))
                self.T_text.set_position((int(event.xdata), self.signal_amp_slider.val * self.signal[int(event.xdata)] + 0.2))

                self.peaks.T.data[self.index] = int(event.xdata) + self.peaks.R.data[self.index]

            if self.update_point == "dT":
                self.dT_point.set_offsets((int(event.xdata), self.signal_amp_slider.val * self.signal[int(event.xdata)]))
                self.dT_text.set_position((int(event.xdata), self.signal_amp_slider.val * self.signal[int(event.xdata)] + 0.2))

                self.peaks.ddT.data[self.index] = int(event.xdata) + self.peaks.R.data[self.index]

            if self.update_point == "ddT":
                self.ddT_point.set_offsets((int(event.xdata), self.signal_amp_slider.val * self.signal[int(event.xdata)]))
                self.ddT_text.set_position((int(event.xdata), self.signal_amp_slider.val * self.signal[int(event.xdata)] + 0.2))

                self.peaks.ddT.data[self.index] = int(event.xdata) + self.peaks.R.data[self.index]

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_click(self, event):
        threshold = 40
        self.update_point = None

        if event.xdata is not None:    
            if abs(self.peaks.T.data[self.index] - self.peaks.R.data[self.index] - event.xdata) < threshold:
                self.lx.set_color('#9467bd')
                self.ly.set_color('#9467bd')

                self.lx.set_linewidth(1)
                self.ly.set_linewidth(1)

                self.fig.canvas.draw()
                
                self.update_point = "T"

            if abs(self.peaks.dT.data[self.index] - self.peaks.R.data[self.index] - event.xdata) < threshold:
                self.lx.set_color('#2ca02c')
                self.ly.set_color('#2ca02c')

                self.lx.set_linewidth(1)
                self.ly.set_linewidth(1)

                self.fig.canvas.draw()
                
                self.update_point = "dT"

            if abs(self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index] - event.xdata) < threshold:
                self.lx.set_color('#2ca02c')
                self.ly.set_color('#2ca02c')

                self.lx.set_linewidth(1)
                self.ly.set_linewidth(1)

                self.fig.canvas.draw()
                
                self.update_point = "ddT"

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
        
        self.index += 1
        if self.index > len(self.peaks.R.data) - 2:
            self.index = 0

        self.update_plot() 

    def prev(self, event):

        self.index -= 1
        if self.index < 0:
            self.index = int(len(self.peaks.R.data) - 2)

        self.update_plot()

    def update_plot(self):
        # Update index
        self.i_text.set_text("Heartbeat: " + str(self.index + 1) + "/" + str(len(self.peaks.R.data) - 1))

        self.signal = hb.normalize(self.peaks.signal[range(self.peaks.R.data[self.index], self.peaks.R.data[self.index + 1])])
        
        # Determine what cutoff freq to use
        cutoff_freq = 15 if np.mean(np.diff(self.peaks.R.data)) > 2500 else 10

        # Pass through a Low pass
        smoothed_signal = hb.lowpass_filter(signal = self.signal,
                                            cutoff_freq = cutoff_freq)

        # Calculate first derivative
        self.first, _ = hb.get_derivatives(smoothed_signal)

        # Update cross hairs
        self.switch_signal(None)

        # Plot ECG, Phono and Seismo
        self.signal_line.set_data(range(len(self.signal)), self.signal_amp_slider.val * self.signal)
        self.first_line.set_data(range(len(self.signal)), (self.first_amp_slider.val * 5*self.first) + self.first_height_slider.val + 1)
       
        self.ax.set_xlim(0, len(self.signal))

        # # T Peak
        # self.T_point.set_offsets((self.peaks.T.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.T.data[self.index] - self.peaks.R.data[self.index]]))
        # self.T_text.set_position((self.peaks.T.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.T.data[self.index] - self.peaks.R.data[self.index]] + 0.2))

        # T''max Peak
        self.dT_point.set_offsets((self.peaks.dT.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.dT.data[self.index] - self.peaks.R.data[self.index]]))
        self.dT_text.set_position((self.peaks.dT.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.dT.data[self.index] - self.peaks.R.data[self.index]] + 0.2))
        
        # # T''max Peak
        # self.ddT_point.set_offsets((self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index]]))
        # self.ddT_text.set_position((self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index], self.signal_amp_slider.val * self.signal[self.peaks.ddT.data[self.index] - self.peaks.R.data[self.index]] + 0.2))

        self.fig.canvas.draw()

