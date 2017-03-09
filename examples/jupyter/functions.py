import numpy as np
import matplotlib
import urllib2
import dolfyn.adv.api as avm
import dolfyn.adv.turbulence as turb
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import matplotlib.ticker as mtick
from mpld3 import plugins
import os.path

def download_file(file_path, filename, url):
    """Downloads the raw ADV file"""
    # If the file exists...

    # ....as an '.h5' file, read it
    if os.path.isfile(file_path + '.h5'):
        dat_raw = avm.load(file_path + '.h5')
    # ....as a '.VEC' file, save it as an '.h5' and then read it using dolfyn library
    elif os.path.isfile(file_path + '.VEC'):
        dat_raw = avm.read_nortek(file_path + '.VEC')
        dat_raw.save(file_path + '.h5')

        # If the file does not exist as either a '.VEC' or '.h5', download it from the internet, save it as a '.h5' file and read it
    else:
        response = urllib2.urlopen(url)
        with open(filename, 'wb') as f:
            f.write(response.read())
        dat_raw = avm.read_nortek(f)
        dat_raw.save(file_path + '.h5')
    return dat_raw

class DataDisplayDownsampler(object):
    """A class that will downsample the data and recompute when zoomed."""
    def __init__(self, xdata, ydata):
        self.origYData = ydata
        self.origXData = xdata
        self.ratio = 5
        self.delta = xdata[-1] - xdata[0]

    def downsample(self, xstart, xend):
        # Very simple downsampling that takes the points within the range
        # and picks every Nth point
        mask = (self.origXData > xstart) & (self.origXData < xend)
        xdata = self.origXData[mask]
        xdata = xdata[::self.ratio]

        ydata = self.origYData[mask]
        ydata = ydata[::self.ratio]

        return xdata, ydata

    def update(self, ax):
        # Update the line
        lims = ax.viewLim
        if np.abs(lims.width - self.delta) > 1e-8:
            self.delta = lims.width
            xstart, xend = lims.intervalx
            self.line.set_data(*self.downsample(xstart, xend))
            ax.figure.canvas.draw_idle()
        fig.canvas.mpl_connect('pick_event', onpick)

def t_range_estimator(dat_raw):
    n_bin = 10000
    n_fft = 4096

    # create a TurbBinner object
    calculator = turb.TurbBinner(n_bin, dat_raw.fs, n_fft=n_fft)
    out = turb.VelBinnerSpec.__call__(calculator, dat_raw)

    # perform the average and variance
    calculator.do_var(dat_raw, out)
    calculator.do_avg(dat_raw, out)

    # add the standard deviation
    out.add_data('sigma_Uh',
                 np.std(calculator.reshape(dat_raw.U_mag), -1, dtype=np.float64) -
                 (dat_raw.noise[0] + dat_raw.noise[1]) / 2, 'main')

    # define just the u velocity variance
    u_var = out.vel_var[0]
    # u_vel = out.vel[0]

    timeset = []
    for i, j in zip(u_var, out.mpltime):
        if i < 1:
            timeset.append(j)

    start = timeset[0]
    end = timeset[len(timeset) - 1]
    t_range = [start, end]
    # display the output below
    return t_range

def raw_data_plot(dat_raw):
    """Plots the raw ADV data"""
    #fig, ax = plt.subplots()
    fig = plt.figure(1, figsize=[8, 4])
    fig.clf()
    ax = fig.add_axes([.14, .14, .8, .74])
    # Create a downsampling object
    d = DataDisplayDownsampler(dat_raw.mpltime, dat_raw.u)
    # Hook up the line
    d.line, = ax.plot(dat_raw.mpltime, dat_raw.u, 'o-')
    ax.set_autoscale_on(False)  # Otherwise, infinite loop
    # Connect for changing the view limits
    ax.callbacks.connect('xlim_changed', d.update)

    ax.set_ylabel('$u\,\mathrm{[m/s]}$', size='large')
    ax.set_xlabel('Time [May 03, 2015]')
    ax.set_title('Raw (unscreened) Data')

    plt.show()

def crop_plot(dat_raw, t_range):
    """Plots the cropped data over the raw data once a time range has been specified"""
    fig = plt.figure(1, figsize=[8, 4])
    fig.clf()
    ax = fig.add_axes([.14, .14, .8, .74])

    # Plot the raw (unscreened) data:
    d = DataDisplayDownsampler(dat_raw.mpltime, dat_raw.u)
    d.line, = ax.plot(dat_raw.mpltime, dat_raw.u, 'o-')
    ax.set_autoscale_on(False)  # Otherwise, infinite loop
    ax.callbacks.connect('xlim_changed', d.update)

    # Plot the screened data:
    t_range_inds = (t_range[0] < dat_raw.mpltime) & (dat_raw.mpltime < t_range[1])
    dat_crop = dat_raw.subset(t_range_inds)
    d1 = DataDisplayDownsampler(dat_crop.mpltime, dat_crop.u)
    d1.line, = ax.plot(dat_crop.mpltime, dat_crop.u, 'o-', rasterized=True)

    bads = np.abs(dat_crop.u - dat_raw.u[t_range_inds])
    ax.text(0.55, 0.95, (np.float(sum(bads > 0)) / len(bads) * 100),
            transform=ax.transAxes,
            va='top',
            ha='left',
            )

    ax.set_ylabel('$u\,\mathrm{[m/s]}$', size='large')
    ax.set_xlabel('Time [May, 2015]')
    ax.set_title('Data cropping')

    plt.show()
    return dat_crop, t_range_inds

def clean(t_range_inds, t_range, dat_raw):
    """Cleans the data"""
    dat = dat_raw.subset(t_range_inds)
    # Then clean the file using the Goring+Nikora method:
    avm.clean.GN2002(dat)
    dat_cln = dat.copy()

    return dat_cln, dat

def clean_plot(dat_raw, dat_crop, dat, t_range_inds, t_range):
    """Plots the cleaned data"""
    fig = plt.figure(1, figsize=[8, 4])
    fig.clf()
    ax = fig.add_axes([.14, .14, .8, .74])

    # Plot the raw (unscreened) data:
    d = DataDisplayDownsampler(dat_raw.mpltime, dat_raw.u)
    d.line, = ax.plot(dat_raw.mpltime, dat_raw.u, 'o-')
    ax.set_autoscale_on(False)  # Otherwise, infinite loop
    ax.callbacks.connect('xlim_changed', d.update)

    # Plot the screened data:
    d1 = DataDisplayDownsampler(dat_crop.mpltime, dat_crop.u)
    d1.line, = ax.plot(dat_crop.mpltime, dat.u, 'o-', rasterized=True)

    bads = np.abs(dat.u - dat_raw.u[t_range_inds])
    ax.text(0.55, 0.95,
            "%0.2f%% of the data were 'cleaned'\nby the Goring and Nikora method."
            % (np.float(sum(bads > 0)) / len(bads) * 100),
            transform=ax.transAxes,
            va='top',
            ha='left',
            )

    # Add some annotations:
    ax.axvspan(dat_raw.mpltime[0], t_range[0], zorder=-10, facecolor='0.9', edgecolor='none')
    ax.text(0.13, 1.0, 'Mooring falling\ntoward seafloor', ha='center', va='top', transform=ax.transAxes, size='small')
    ax.text(0.3, 0.6, 'Mooring on seafloor', ha='center', va='top', transform=ax.transAxes, size='small')
    ax.annotate('', (0.25, 0.4), (0.4, 0.4), arrowprops=dict(facecolor='black'))

    ax.set_ylabel('$u\,\mathrm{[m/s]}$', size='large')
    ax.set_xlabel('Time [May, 2015]')
    ax.set_title('Data cropping and cleaning')

    plt.show()

def motion_correct(dat_crop, accel_filter, dat_cln):
    """Corrects the motion"""
    avm.motion.correct_motion(dat_crop, accel_filter)

    # Rotate the uncorrected data into the earth frame,
    # for comparison to motion correction:
    avm.rotate.inst2earth(dat_cln)

    # ax.plot(dat.mpltime, dat.u, 'b-')

    # Then rotate it into a 'principal axes frame':
    avm.rotate.earth2principal(dat_crop)
    avm.rotate.earth2principal(dat_cln)

    return dat_crop, dat_cln

def spectra(dat, dat_cln):
    """Plots the turbulence spectra"""
    # Average the data and compute turbulence statistics
    dat_bin = avm.calc_turbulence(dat, n_bin=19200,
                                  n_fft=4096)
    dat_cln_bin = avm.calc_turbulence(dat_cln, n_bin=19200,
                                      n_fft=4096)

    fig2 = plt.figure(2, figsize=[6, 6])
    fig2.clf()
    ax = fig2.add_axes([.14, .14, .8, .74])

    ax.loglog(dat_bin.freq, dat_bin.Suu_hz.mean(0),
              'b-', label='motion corrected')
    ax.loglog(dat_cln_bin.freq, dat_cln_bin.Suu_hz.mean(0),
              'r-', label='no motion correction')

    ax.set_xlim([1e-3, 20])
    ax.set_ylim([1e-4, 1])
    ax.set_xlabel('frequency [hz]')
    ax.set_ylabel('$\mathrm{[m^2s^{-2}/hz]}$', size='large')

    f_tmp = np.logspace(-3, 1)
    ax.plot(f_tmp, 4e-5 * f_tmp ** (-5. / 3), 'k--')

    ax.set_title('Velocity Spectra')
    ax.legend()

    plt.show()
    return dat_bin, dat_cln_bin


def tke_plot(dat_cln_bin):
    """Plots the Turbulent Kinetic Energy"""
    fig = plt.figure(1, figsize=[8, 4])
    fig.clf()
    ax = fig.add_axes([.14, .14, .8, .74])

    # first, convert the num_time to date_time, and plot this versus dat_raw.u
    date_time = dt.num2date(dat_cln_bin.mpltime)

    # plot the data
    ax.plot(date_time, dat_cln_bin.upup_, 'r-', rasterized=True)
    ax.plot(date_time, dat_cln_bin.vpvp_, 'g-', rasterized=True)
    ax.plot(date_time, dat_cln_bin.wpwp_, 'b-', rasterized=True)

    # label axes
    ax.set_xlabel('Time')
    ax.set_ylabel('Turbulent Energy $\mathrm{[m^2/s^2]}$', size='large')

    plt.show()


def reynolds_stress(dat_cln_bin):
    """Plots the Reynold's Stress"""
    fig = plt.figure(1, figsize=[8, 4])
    fig.clf()
    ax = fig.add_axes([.14, .14, .8, .74])

    # first, convert the num_time to date_time, and plot this versus dat_raw.u
    date_time = dt.num2date(dat_cln_bin.mpltime)

    # plot the data
    ax.plot(date_time, dat_cln_bin.upvp_, 'r-', rasterized=True)
    ax.plot(date_time, dat_cln_bin.upwp_, 'g-', rasterized=True)
    ax.plot(date_time, dat_cln_bin.vpwp_, 'b-', rasterized=True)

    # label axes
    ax.set_xlabel('Time')
    ax.set_ylabel('Reynolds Stresses $\mathrm{[m^2/s^2]}$', size='large')

    plt.show()