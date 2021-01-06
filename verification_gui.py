import numpy as np
import matplotlib.pyplot as plt
from peaks import Peaks
import heartbreaker as hb
from composite_peaks import CompositePeaks
from matplotlib.widgets import Button, RadioButtons

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
        self.plot_signals(self.composite_peaks, index)       

    def plot_signals(self, composite_peaks, i):
        # Create figure
        self.fig, self.ax = plt.subplots()
        
        # Load composite signals
        time, signal, seis, phono = composite_peaks.composites[i]
        _, second = hb.get_derivatives(signal)

        self.signal_i = signal
        self.seis_i   = seis
        self.phono_i  = phono

        # Plot ECG, Phono and Seismo
        self.signal_line, = self.ax.plot(signal, linewidth = 1, c = "b", label = "ECG")

        self.second_line, = self.ax.plot(range(composite_peaks.ST_start.data[i], len(signal) - 1), 
                                        2 + 5*second[range(composite_peaks.ST_start.data[i], len(signal) - 1],
                                        '--', linewidth = 0.5, c = 'k', label = "ECG 2nd Derv.")

        self.seis_line,   = self.ax.plot(seis , '--', linewidth = 0.5, c = 'r', label = "Seis")
        self.phono_line,  = self.ax.plot(phono, '--', linewidth = 0.5, c = 'g', label = "Phono")
        
        self.ax.set_xlim(0, len(self.signal_i))
        plt.legend(loc='upper right')

        # Q Peak
        self.q_point = self.ax.scatter(composite_peaks.Q.data[i], signal[composite_peaks.Q.data[i]], c = '#ff7f0e')
        self.q_text  = self.ax.text(composite_peaks.Q.data[i], signal[composite_peaks.Q.data[i]] + 0.2, "Q", fontsize=9, horizontalalignment = 'center')

        # QM Seismo
        self.qm_seis_point = self.ax.scatter(composite_peaks.QM_seis.data[i], seis[composite_peaks.QM_seis.data[i]], c = '#d62728')
        self.qm_seis_text  = self.ax.text(composite_peaks.QM_seis.data[i], seis[composite_peaks.QM_seis.data[i]] + 0.2, "QM Seis", fontsize=9, horizontalalignment = 'center')

        # QM Phono
        self.qm_phono_point = self.ax.scatter(composite_peaks.QM_phono.data[i], phono[composite_peaks.QM_phono.data[i]], c = '#8c564b')
        self.qm_phono_text  = self.ax.text(composite_peaks.QM_phono.data[i], phono[composite_peaks.QM_phono.data[i]] + 0.2, "QM Phono", fontsize=9, horizontalalignment = 'center')

        # T''max Peak
        self.ddT_max_point = self.ax.scatter(composite_peaks.ddT_max.data[i], signal[composite_peaks.ddT_max.data[i]], c = '#2ca02c')
        self.ddT_max_text  = self.ax.text(composite_peaks.ddT_max.data[i], signal[composite_peaks.ddT_max.data[i]] + 0.2, "T''max", fontsize=9, horizontalalignment = 'center')

        # TM Seismo
        self.tm_seis_point = self.ax.scatter(composite_peaks.TM_seis.data[i], seis[composite_peaks.TM_seis.data[i]], c = '#9467bd')
        self.tm_seis_text  = self.ax.text(composite_peaks.TM_seis.data[i], seis[composite_peaks.TM_seis.data[i]] + 0.2, "TM Seis", fontsize=9, horizontalalignment = 'center')

        # TM Phono
        self.tm_phono_point = self.ax.scatter(composite_peaks.TM_phono.data[i], phono[composite_peaks.TM_phono.data[i]], c = '#e377c2')
        self.tm_phono_text  = self.ax.text(composite_peaks.TM_phono.data[i], phono[composite_peaks.TM_phono.data[i]] + 0.2, "TM Phono", fontsize=9, horizontalalignment = 'center')

        # Initalize axes and data points
        self.x = range(len(self.signal_i))
        self.y = self.signal_i

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
                                    s = "Interval: " + str(i + 1) + "/" + str(len(composite_peaks.composites)), fontsize=12, horizontalalignment = 'left')

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
            self.x = range(len(self.signal_i))
            self.y = self.signal_i

        if label == 'Seismo':
            self.x = range(len(self.signal_i))
            self.y = self.seis_i

        if label == 'Phono':
            self.x = range(len(self.signal_i))
            self.y = self.phono_i

    def off_click(self, event):
        self.lx.set_color('k')
        self.ly.set_color('k')

        self.lx.set_linewidth(0.2)
        self.ly.set_linewidth(0.2)

        if event.xdata is not None:
            if self.update_point == "Q":
                self.q_point.set_offsets((int(event.xdata), self.signal_i[int(event.xdata)]))
                self.q_text.set_position((int(event.xdata), self.signal_i[int(event.xdata)] + 0.2))

                self.composite_peaks.Q.data[self.index] = int(event.xdata)

            if self.update_point == "ddT_max":
                self.ddT_max_point.set_offsets((int(event.xdata), self.signal_i[int(event.xdata)]))
                self.ddT_max_text.set_position((int(event.xdata), self.signal_i[int(event.xdata)] + 0.2))

                self.composite_peaks.ddT_max.data[self.index] = int(event.xdata)

            if self.update_point == "QM Seismo":
                self.qm_seis_point.set_offsets((int(event.xdata), self.seis_i[int(event.xdata)]))
                self.qm_seis_text.set_position((int(event.xdata), self.seis_i[int(event.xdata)] + 0.2))

                self.composite_peaks.QM_seis.data[self.index] = int(event.xdata)

            if self.update_point == "TM Seismo":
                self.tm_seis_point.set_offsets((int(event.xdata), self.seis_i[int(event.xdata)]))
                self.tm_seis_text.set_position((int(event.xdata), self.seis_i[int(event.xdata)] + 0.2))

                self.composite_peaks.TM_seis.data[self.index] = int(event.xdata)

            if self.update_point == "QM Phono":
                self.qm_phono_point.set_offsets((int(event.xdata), self.phono_i[int(event.xdata)]))
                self.qm_phono_text.set_position((int(event.xdata), self.phono_i[int(event.xdata)] + 0.2))

                self.composite_peaks.QM_phono.data[self.index] = int(event.xdata)

            if self.update_point == "TM Phono":
                self.tm_phono_point.set_offsets((int(event.xdata), self.phono_i[int(event.xdata)]))
                self.tm_phono_text.set_position((int(event.xdata), self.phono_i[int(event.xdata)] + 0.2))

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
        self.update_plot(self.composite_peaks, self.index) 

    def prev(self, event):
        self.index -= 1
        if self.index < 0:
            self.index = len(self.composite_peaks.composites) - 1
        self.update_plot(self.composite_peaks, self.index)

    def save(self, event):
        # Get File Name
        save_file_name = self.folder_name + "_d" + str(self.dosage) + "_" + self.file_name + "_i" + str(self.interval_number)

        # Save
        self.composite_peaks.save(save_file_name)
        print("Saved")

    def update_plot(self, composite_peaks, i):
        # Update index
        self.i_text.set_text("Interval: " + str(self.index + 1) + "/" + str(len(composite_peaks.composites)))

        # Load composite signals
        time, signal, seis, phono = composite_peaks.composites[i]
        _, second = hb.get_derivatives(signal)

        self.signal_i = signal
        self.seis_i = seis
        self.phono_i = phono

        # Update cross hairs
        self.switch_signal(self.b_switch_signals.value_selected)

        # Plot ECG, Phono and Seismo
        self.signal_line.set_data(range(len(signal)), signal)
        self.second_line.set_data(range(composite_peaks.ST_start.data[i], composite_peaks.T.data[i]),
                                  2 + 5*second[range(composite_peaks.ST_start.data[i], composite_peaks.T.data[i])])
        self.seis_line.set_data(range(len(seis)), seis)
        self.phono_line.set_data(range(len(phono)), phono)
        self.ax.set_xlim(0, len(self.signal_i))

        # Q Peaks
        self.q_point.set_offsets((composite_peaks.Q.data[i], signal[composite_peaks.Q.data[i]]))
        self.q_text.set_position((composite_peaks.Q.data[i], signal[composite_peaks.Q.data[i]] + 0.2))

        # QM Seismo
        self.qm_seis_point.set_offsets((composite_peaks.QM_seis.data[i], seis[composite_peaks.QM_seis.data[i]]))
        self.qm_seis_text.set_position((composite_peaks.QM_seis.data[i], seis[composite_peaks.QM_seis.data[i]] + 0.2))
        
        # QM Phono
        self.qm_phono_point.set_offsets((composite_peaks.QM_phono.data[i], phono[composite_peaks.QM_phono.data[i]]))
        self.qm_phono_text.set_position((composite_peaks.QM_phono.data[i], phono[composite_peaks.QM_phono.data[i]] + 0.2))
        
        # T''max Peak
        self.ddT_max_point.set_offsets((composite_peaks.ddT_max.data[i], signal[composite_peaks.ddT_max.data[i]]))
        self.ddT_max_text.set_position((composite_peaks.ddT_max.data[i], signal[composite_peaks.ddT_max.data[i]] + 0.2))

        # TM Seismo
        self.tm_seis_point.set_offsets((composite_peaks.TM_seis.data[i], seis[composite_peaks.TM_seis.data[i]]))
        self.tm_seis_text.set_position((composite_peaks.TM_seis.data[i], seis[composite_peaks.TM_seis.data[i]] + 0.2))
        
        # TM Phono
        self.tm_phono_point.set_offsets((composite_peaks.TM_phono.data[i], phono[composite_peaks.TM_phono.data[i]]))
        self.tm_phono_text.set_position((composite_peaks.TM_phono.data[i], phono[composite_peaks.TM_phono.data[i]] + 0.2))
    
        self.fig.canvas.draw()


