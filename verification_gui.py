import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from peaks import Peaks
import heartbreaker as hb
from composite_peaks import CompositePeaks
from matplotlib.widgets import Button, RadioButtons, Slider

class HeartbeatVerifier(object):
    """
    Cursor for editing heartbeat signals for use of verification
    """

    def __init__(self, composite_peaks, 
                       index = 0,
                       folder_name = "",
                       dosage = "",
                       file_name = "",
                       interval_number = ""):

        super(HeartbeatVerifier, self).__init__()
        # Save Composite Peaks
        self.composite_peaks = composite_peaks
        self.index = index

        self.folder_name = folder_name
        self.dosage      = dosage
        self.file_name   = file_name
        self.interval_number = interval_number

        # Plot signals
        self.plot_signals()       

    def plot_signals(self):
        # Create figure
        self.fig, self.ax = plt.subplots()
        
        # Load composite signals
        self.time, self.signal, self.seis, self.phono = self.composite_peaks.composites[self.index]
        _, self.second = hb.get_derivatives(self.signal)

        # Plot ECG, Phono and Seismo
        self.signal_line, = self.ax.plot(self.signal, linewidth = 1, c = "b", label = "ECG")

        self.second_line, = self.ax.plot(range(self.composite_peaks.ST_start.data[self.index], int(np.median([self.composite_peaks.T.data[self.index], len(self.signal)]))), 
                                        2 + 5*self.second[range(self.composite_peaks.ST_start.data[self.index], int(np.median([self.composite_peaks.T.data[self.index], len(self.signal)])))],
                                        '--', linewidth = 0.5, c = 'k', label = "ECG 2nd Derv.")

        self.seis_line,   = self.ax.plot(self.seis , '--', linewidth = 0.5, c = 'r', label = "Seis")
        self.phono_line,  = self.ax.plot(self.phono, '--', linewidth = 0.5, c = 'g', label = "Phono")
        
        self.ax.set_xlim(0, len(self.signal))

        sig_min = min(self.signal)
        sig_max = max(self.signal)

        self.ax.set_ylim(sig_min - 0.1*(sig_max - sig_min), sig_max + 0.1*(sig_max - sig_min))
        plt.legend(loc='upper right')

        # Q Peak
        self.q_point = self.ax.scatter(self.composite_peaks.Q.data[self.index], self.signal[self.composite_peaks.Q.data[self.index]], c = '#ff7f0e')
        self.q_text  = self.ax.text(self.composite_peaks.Q.data[self.index], self.signal[self.composite_peaks.Q.data[self.index]] + 0.2, "Q", fontsize=9, horizontalalignment = 'center')

        # QM Seismo
        self.qm_seis_point = self.ax.scatter(self.composite_peaks.QM_seis.data[self.index], self.seis[self.composite_peaks.QM_seis.data[self.index]], c = '#d62728')
        self.qm_seis_text  = self.ax.text(self.composite_peaks.QM_seis.data[self.index], self.seis[self.composite_peaks.QM_seis.data[self.index]] + 0.2, "QM Seis", fontsize=9, horizontalalignment = 'center')

        # QM Phono
        self.qm_phono_point = self.ax.scatter(self.composite_peaks.QM_phono.data[self.index], self.phono[self.composite_peaks.QM_phono.data[self.index]], c = '#8c564b')
        self.qm_phono_text  = self.ax.text(self.composite_peaks.QM_phono.data[self.index], self.phono[self.composite_peaks.QM_phono.data[self.index]] + 0.2, "QM Phono", fontsize=9, horizontalalignment = 'center')

        # T''max Peak
        self.ddT_max_point = self.ax.scatter(self.composite_peaks.ddT_max.data[self.index], self.signal[self.composite_peaks.ddT_max.data[self.index]], c = '#2ca02c')
        self.ddT_max_text  = self.ax.text(self.composite_peaks.ddT_max.data[self.index], self.signal[self.composite_peaks.ddT_max.data[self.index]] + 0.2, "T''max", fontsize=9, horizontalalignment = 'center')

        # TM Seismo
        self.tm_seis_point = self.ax.scatter(self.composite_peaks.TM_seis.data[self.index], self.seis[self.composite_peaks.TM_seis.data[self.index]], c = '#9467bd')
        self.tm_seis_text  = self.ax.text(self.composite_peaks.TM_seis.data[self.index], self.seis[self.composite_peaks.TM_seis.data[self.index]] + 0.2, "TM Seis", fontsize=9, horizontalalignment = 'center')

        # TM Phono
        self.tm_phono_point = self.ax.scatter(self.composite_peaks.TM_phono.data[self.index], self.phono[self.composite_peaks.TM_phono.data[self.index]], c = '#e377c2')
        self.tm_phono_text  = self.ax.text(self.composite_peaks.TM_phono.data[self.index], self.phono[self.composite_peaks.TM_phono.data[self.index]] + 0.2, "TM Phono", fontsize=9, horizontalalignment = 'center')

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
        self.ax.text(0.01, start - 2*space, transform = self.ax.transAxes,
                    s = "File: " + self.file_name, fontsize=12, horizontalalignment = 'left')
        self.ax.text(0.01, start - 3*space, transform = self.ax.transAxes,
                    s = "File #: " + str(self.interval_number), fontsize=12, horizontalalignment = 'left')
        self.i_text = self.ax.text(0.61 - left_shift, 1.1 - space, transform = self.ax.transAxes,
                                    s = "Interval: " + str(self.index + 1) + "/" + str(len(self.composite_peaks.composites)), fontsize=12, horizontalalignment = 'left')

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
        ax_switch_signals = plt.axes([0.91, 0.7, 0.07, 0.15])
        self.b_switch_signals = RadioButtons(ax_switch_signals, ('ECG', 'Seismo', 'Phono'))
        for c in self.b_switch_signals.circles:
            c.set_radius(0.05)

        self.b_switch_signals.on_clicked(self.switch_signal)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.off_click)

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

        # Maximize frame
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()

        plt.show()

    def switch_signal(self, label):

        # Update Lines
        self.signal_line.set_data(range(len(self.signal)), self.signal_amp_slider.val * self.signal)
        self.seis_line.set_data(range(len(self.signal)),  (self.seis_amp_slider.val * self.seis) + self.seis_height_slider.val)
        self.phono_line.set_data(range(len(self.signal)), (self.phono_amp_slider.val * self.phono) + self.phono_height_slider.val)

        # Q Peaks
        self.q_point.set_offsets((self.composite_peaks.Q.data[self.index], self.signal_amp_slider.val * self.signal[self.composite_peaks.Q.data[self.index]]))
        self.q_text.set_position((self.composite_peaks.Q.data[self.index], self.signal_amp_slider.val * self.signal[self.composite_peaks.Q.data[self.index]] + 0.2))

        # QM Seismo
        self.qm_seis_point.set_offsets((self.composite_peaks.QM_seis.data[self.index], self.seis_amp_slider.val *  self.seis[self.composite_peaks.QM_seis.data[self.index]] + self.seis_height_slider.val))
        self.qm_seis_text.set_position((self.composite_peaks.QM_seis.data[self.index], self.seis_amp_slider.val *  self.seis[self.composite_peaks.QM_seis.data[self.index]] + 0.2 + self.seis_height_slider.val))
        
        # QM Phono
        self.qm_phono_point.set_offsets((self.composite_peaks.QM_phono.data[self.index], self.phono_amp_slider.val * self.phono[self.composite_peaks.QM_phono.data[self.index]] + self.phono_height_slider.val))
        self.qm_phono_text.set_position((self.composite_peaks.QM_phono.data[self.index], self.phono_amp_slider.val * self.phono[self.composite_peaks.QM_phono.data[self.index]] + 0.2+ self.phono_height_slider.val))
        
        # T''max Peak
        self.ddT_max_point.set_offsets((self.composite_peaks.ddT_max.data[self.index], self.signal_amp_slider.val * self.signal[self.composite_peaks.ddT_max.data[self.index]]))
        self.ddT_max_text.set_position((self.composite_peaks.ddT_max.data[self.index], self.signal_amp_slider.val * self.signal[self.composite_peaks.ddT_max.data[self.index]] + 0.2))

        # TM Seismo
        self.tm_seis_point.set_offsets((self.composite_peaks.TM_seis.data[self.index], self.seis_amp_slider.val * self.seis[self.composite_peaks.TM_seis.data[self.index]] + self.seis_height_slider.val))
        self.tm_seis_text.set_position((self.composite_peaks.TM_seis.data[self.index], self.seis_amp_slider.val * self.seis[self.composite_peaks.TM_seis.data[self.index]] + 0.2 + self.seis_height_slider.val))
        
        # TM Phono
        self.tm_phono_point.set_offsets((self.composite_peaks.TM_phono.data[self.index], self.phono_amp_slider.val * self.phono[self.composite_peaks.TM_phono.data[self.index]] + self.phono_height_slider.val))
        self.tm_phono_text.set_position((self.composite_peaks.TM_phono.data[self.index], self.phono_amp_slider.val * self.phono[self.composite_peaks.TM_phono.data[self.index]] + 0.2 + self.phono_height_slider.val))

        # Update Cross-hairs
        label = self.b_switch_signals.value_selected
        self.x = range(len(self.signal))
        if label == 'ECG':
            self.y = self.signal_amp_slider.val * self.signal

        if label == 'Seismo':
            self.y = self.seis_amp_slider.val * self.seis + self.seis_height_slider.val

        if label == 'Phono':
            self.y = self.phono_amp_slider.val * self.phono + self.phono_height_slider.val

        self.fig.canvas.draw()

    def off_click(self, event):
        self.lx.set_color('k')
        self.ly.set_color('k')

        self.lx.set_linewidth(0.2)
        self.ly.set_linewidth(0.2)

        if event.xdata is not None:
            if self.update_point == "Q":
                self.q_point.set_offsets((int(event.xdata), self.signal_amp_slider.val *  self.signal[int(event.xdata)]))
                self.q_text.set_position((int(event.xdata), self.signal_amp_slider.val *  self.signal[int(event.xdata)] + 0.2))

                self.composite_peaks.Q.data[self.index] = int(event.xdata)

            if self.update_point == "ddT_max":
                self.ddT_max_point.set_offsets((int(event.xdata), self.signal_amp_slider.val * self.signal[int(event.xdata)]))
                self.ddT_max_text.set_position((int(event.xdata), self.signal_amp_slider.val * self.signal[int(event.xdata)] + 0.2))

                self.composite_peaks.ddT_max.data[self.index] = int(event.xdata)

            if self.update_point == "QM Seismo":
                self.qm_seis_point.set_offsets((int(event.xdata), self.seis_amp_slider.val * self.seis[int(event.xdata)] + self.seis_height_slider.val))
                self.qm_seis_text.set_position((int(event.xdata), self.seis_amp_slider.val * self.seis[int(event.xdata)] + 0.2 + self.seis_height_slider.val))

                self.composite_peaks.QM_seis.data[self.index] = int(event.xdata)

            if self.update_point == "TM Seismo":
                self.tm_seis_point.set_offsets((int(event.xdata), self.seis_amp_slider.val * self.seis[int(event.xdata)] + self.seis_height_slider.val))
                self.tm_seis_text.set_position((int(event.xdata), self.seis_amp_slider.val * self.seis[int(event.xdata)] + 0.2 + self.seis_height_slider.val))

                self.composite_peaks.TM_seis.data[self.index] = int(event.xdata)

            if self.update_point == "QM Phono":
                self.qm_phono_point.set_offsets((int(event.xdata), self.phono_amp_slider.val * self.phono[int(event.xdata)] + self.phono_height_slider.val))
                self.qm_phono_text.set_position((int(event.xdata), self.phono_amp_slider.val * self.phono[int(event.xdata)] + 0.2 + self.phono_height_slider.val))

                self.composite_peaks.QM_phono.data[self.index] = int(event.xdata)

            if self.update_point == "TM Phono":
                self.tm_phono_point.set_offsets((int(event.xdata), self.phono_amp_slider.val * self.phono[int(event.xdata)] + self.phono_height_slider.val))
                self.tm_phono_text.set_position((int(event.xdata), self.phono_amp_slider.val * self.phono[int(event.xdata)] + 0.2 + self.phono_height_slider.val))

                self.composite_peaks.TM_phono.data[self.index] = int(event.xdata)
                
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_click(self, event):
        threshold = 40
        current_signal = self.b_switch_signals.value_selected
        self.update_point = None

        if event.xdata is not None:
            if current_signal == 'ECG':
                if abs(self.composite_peaks.Q.data[self.index] - event.xdata) < threshold:
                    self.lx.set_color('#ff7f0e')
                    self.ly.set_color('#ff7f0e')

                    self.lx.set_linewidth(1)
                    self.ly.set_linewidth(1)

                    self.fig.canvas.draw()
                    
                    self.update_point = "Q"
                    
                if abs(self.composite_peaks.ddT_max.data[self.index] - event.xdata) < threshold:
                    self.lx.set_color('#2ca02c')
                    self.ly.set_color('#2ca02c')

                    self.lx.set_linewidth(1)
                    self.ly.set_linewidth(1)

                    self.fig.canvas.draw()
                    
                    self.update_point = "ddT_max"

            if current_signal == 'Seismo':
                if abs(self.composite_peaks.QM_seis.data[self.index] - event.xdata) < threshold:
                    self.lx.set_color('#d62728')
                    self.ly.set_color('#d62728')

                    self.lx.set_linewidth(1)
                    self.ly.set_linewidth(1)
                    
                    self.fig.canvas.draw()
                    
                    self.update_point = "QM Seismo"

                if abs(self.composite_peaks.TM_seis.data[self.index] - event.xdata) < threshold:
                    self.lx.set_color('#9467bd')
                    self.ly.set_color('#9467bd')

                    self.lx.set_linewidth(1)
                    self.ly.set_linewidth(1)
                    
                    self.fig.canvas.draw()
                    
                    self.update_point = "TM Seismo"
            
            if current_signal == 'Phono':
                if abs(self.composite_peaks.QM_phono.data[self.index] - event.xdata) < threshold:
                    self.lx.set_color('#8c564b')
                    self.ly.set_color('#8c564b')

                    self.lx.set_linewidth(1)
                    self.ly.set_linewidth(1)
                    
                    self.fig.canvas.draw()
                    
                    self.update_point = "QM Phono"

                if abs(self.composite_peaks.TM_phono.data[self.index] - event.xdata) < threshold:
                    self.lx.set_color('#e377c2')
                    self.ly.set_color('#e377c2')

                    self.lx.set_linewidth(1)
                    self.ly.set_linewidth(1)
                    
                    self.fig.canvas.draw()
                    
                    self.update_point = "TM Phono"

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
        if self.index > len(self.composite_peaks.composites) - 1:
            self.index = 0
        self.update_plot() 

    def prev(self, event):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.composite_peaks.composites) - 1
        self.update_plot()

    def save(self, event):
        # Get File Name
        save_file_name = "composites_" + self.folder_name + "_" + self.file_name + "_d" + str(self.dosage) 

        # Save
        self.composite_peaks.save("data/Derived/composites/" + save_file_name)
        print("Saved")

    def update_plot(self):
        # Update index
        self.i_text.set_text("Interval: " + str(self.index + 1) + "/" + str(len(self.composite_peaks.composites)))

        # Load composite signals
        self.time, self.signal, self.seis, self.phono = self.composite_peaks.composites[self.index]
        _, self.second = hb.get_derivatives(self.signal)

        # Update cross hairs
        self.switch_signal(None)

        # Plot ECG, Phono and Seismo
        self.signal_line.set_data(range(len(self.signal)), self.signal)
        self.second_line.set_data(range(self.composite_peaks.ST_start.data[self.index], int(np.median([self.composite_peaks.T.data[self.index], len(self.signal)]))),
                                  2 + 5*self.second[range(self.composite_peaks.ST_start.data[self.index], int(np.median([self.composite_peaks.T.data[self.index], len(self.signal)])))])
        self.seis_line.set_data(range(len(self.seis)), self.seis)
        self.phono_line.set_data(range(len(self.phono)), self.phono)
        self.ax.set_xlim(0, len(self.signal))

        # Q Peaks
        self.q_point.set_offsets((self.composite_peaks.Q.data[self.index], self.signal[self.composite_peaks.Q.data[self.index]]))
        self.q_text.set_position((self.composite_peaks.Q.data[self.index], self.signal[self.composite_peaks.Q.data[self.index]] + 0.2))

        # QM Seismo
        self.qm_seis_point.set_offsets((self.composite_peaks.QM_seis.data[self.index], self.seis[self.composite_peaks.QM_seis.data[self.index]]))
        self.qm_seis_text.set_position((self.composite_peaks.QM_seis.data[self.index], self.seis[self.composite_peaks.QM_seis.data[self.index]] + 0.2))
        
        # QM Phono
        self.qm_phono_point.set_offsets((self.composite_peaks.QM_phono.data[self.index], self.phono[self.composite_peaks.QM_phono.data[self.index]]))
        self.qm_phono_text.set_position((self.composite_peaks.QM_phono.data[self.index], self.phono[self.composite_peaks.QM_phono.data[self.index]] + 0.2))
        
        # T''max Peak
        self.ddT_max_point.set_offsets((self.composite_peaks.ddT_max.data[self.index], self.signal[self.composite_peaks.ddT_max.data[self.index]]))
        self.ddT_max_text.set_position((self.composite_peaks.ddT_max.data[self.index], self.signal[self.composite_peaks.ddT_max.data[self.index]] + 0.2))

        # TM Seismo
        self.tm_seis_point.set_offsets((self.composite_peaks.TM_seis.data[self.index], self.seis[self.composite_peaks.TM_seis.data[self.index]]))
        self.tm_seis_text.set_position((self.composite_peaks.TM_seis.data[self.index], self.seis[self.composite_peaks.TM_seis.data[self.index]] + 0.2))
        
        # TM Phono
        self.tm_phono_point.set_offsets((self.composite_peaks.TM_phono.data[self.index], self.phono[self.composite_peaks.TM_phono.data[self.index]]))
        self.tm_phono_text.set_position((self.composite_peaks.TM_phono.data[self.index], self.phono[self.composite_peaks.TM_phono.data[self.index]] + 0.2))
    
        self.fig.canvas.draw()

