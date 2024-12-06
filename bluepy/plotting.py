""" Module containing the different free functions for plotting.
 These functions are dynamically loaded and provided to the SimulationPlotHelper if
 the function is listed in the SIMULATION_PLOTS variable.
 """
import logging

import numpy as np

from bluepy.enums import Cell
from bluepy.exceptions import BluePyError
from bluepy.utils import ensure_list

try:
    from matplotlib import gridspec
    from matplotlib import pyplot as plt
except ImportError:
    raise BluePyError("Please install matplolib to use this module or install bluepy with the "
                      "[all] requirement.")

L = logging.getLogger(__name__)

SIMULATION_PLOTS = ['firing_rate_histogram',
                    'raster',
                    'isi',
                    'voltage_collage',
                    'firing_animation',
                    'trace',
                    'multi_trace',
                    'spikeraster_and_psth',
                    'population_summary',
                    'voltage_histogram'
                    ]


def group_to_str(group, sample=None, nb_max=5):
    """ Returns a string representing the group

    Args:
        group: Cell group of interest, like in `bluepy.CellCollection.ids`
        sample: Sample size
        nb_max: The number of displayed gids for gids lists

    Returns:
        a string representing the group

    Raises:
        if group is not str|int|list of int|array of int/np.integer
    """
    if isinstance(group, str):
        if sample is None:
            return group
        else:
            return f"{group}, sample = {sample}"
    elif isinstance(group, (int, np.integer, dict)):
        return str(group)
    elif isinstance(group, (list, np.ndarray)) and all(
            isinstance(gid, (int, np.integer)) for gid in group):
        return str(group) if len(group) < nb_max else str(
            list(group[:nb_max]) + ["..."])
    elif group is None:
        return ""
    else:
        raise BluePyError(f'Group {group} cannot be stringifyied')


def select_gids(sim, group, sample, report_name=None):
    """ Select some GIDs from simulation target GIDs.

    Args:
        sim: Bluepy simulation
        group: cell group of interest, like in `bluepy.CellCollection.ids`
        report_name: name of the report to retrieve
        sample: sample size (None for no sampling)

    Returns:
        the selected gids from group present in the report or in targets
    """
    if report_name is None:
        result = sim.target_gids
    else:
        result = sim.report(report_name).gids
    if group is not None:
        group = sim.circuit.cells.ids(group)
        result = np.intersect1d(result, group)
        if len(result) < len(group):
            excluded = np.setdiff1d(group, result)
            L.warning(
                "Excluding GIDs not observed in the report: [%s]",
                ",".join(map(str, sorted(excluded)))
            )
    if sample is not None and len(result) > sample:
        result = np.random.choice(result, sample, replace=False)
    return result


def get_report_data(sim, report_name, group, sample, t_start, t_end, t_step):
    """ Extract data from report

     Args:
        sim: Bluepy simulation
        report_name: name of the report to retrieve
        group: cell group of interest, like in `bluepy.CellCollection.ids`
        sample: sample size (default = 1000, None for no sampling)
        t_start: Starting time of the simulation
        t_end: Ending time of the simulation
        t_step: time steps for the simulation

    Returns:
        DataFrame with multi index: time as columns and gid as row indexing. The values
        are the voltage of cells at a specific time.

    Raises:
        if no gid can be returned
    """
    report = sim.report(report_name)
    gids = select_gids(sim, group, sample)
    if len(gids) == 0:
        raise BluePyError("No GIDs to return")
    return report.get(t_start=t_start, t_end=t_end, t_step=t_step, gids=gids)


def check_times(sim, t_start, t_end, report_name=None):
    """ Ensure a correct range of dates for the simulation analysis

    Args:
        sim: Bluepy simulation
        t_start: Starting time of the simulation
        t_end: Ending time of the simulation
        report_name: A report name if needed

    Returns:
        t_start, t_end if set correctly or self._sim.t_start, self._sim.t_end if None

    Notes:
        In case of report name == None then the function allows the 'out of range values'
        for t_start and t_end to zoom out of plots. Only check for t_start >= t_end.
        In case of report name != None if t_start or t_end are out of range for the report then
        the report t_start and/or t_end are used instead.

    Raises:
        if t_start >= t_end
    """
    min_t_start = sim.t_start if report_name is None else sim.report(report_name).t_start
    max_t_end = sim.t_end if report_name is None else sim.report(report_name).t_end

    t_start = min_t_start if t_start is None else t_start
    t_end = max_t_end if t_end is None else t_end

    if report_name is not None:
        if not min_t_start <= t_start <= max_t_end:
            t_start = min_t_start
            L.warning("t_start not in report time range. Set t_start to %f instead", t_start)
        if not min_t_start <= t_end <= max_t_end:
            t_end = max_t_end
            L.warning("t_end not in report time range. Set t_end to %f instead", t_end)
    if t_start >= t_end:
        raise BluePyError("Starting time is bigger than ending time")
    return t_start, t_end


def get_figure(width=10, height=8):
    """ Return a mpl figure """
    return plt.figure(figsize=(width, height))


def potential_axes_update(ax, plot_type, xlegend=True):
    """ Update the axes labels for potential plots

    Args:
        ax: matplotlib Axes to draw on
        plot_type: the plot time for potential plots ('mean'|'all')
        xlegend: Set the xlabel legend (boolean, default set to True)
    """
    if plot_type == "mean":
        ax.set_ylabel('Avg volt. [mV]')
    elif plot_type == "all":
        ax.set_ylabel('Voltage [mV]')
    else:
        ax.set_ylabel('')
    if xlegend:
        ax.set_xlabel("Time [ms]")
    else:
        ax.set_xlabel('')
        ax.set_xticklabels([])


def firing_rate_histogram(sim, group=None, sample=1000, t_start=None,
                          t_end=None, binsize=None, label=None, ax=None):  # pragma: no cover
    """
    Firing rate histogram.

    Args:
        sim: Bluepy simulation
        group: cell group of interest, like in `bluepy.CellCollection.ids`
        sample: sample size (default = 1000, None for no sampling)
        t_start: Starting time of the simulation
        t_end: Ending time of the simulation
        binsize: bin size (milliseconds)
        label: label to use for the plot legend
        ax: matplotlib Axes to draw on (if not specified, pyplot.gca() is used)

    Returns:
        matplotlib Axes with firing rate histogram.
    """
    t_start, t_end = check_times(sim, t_start, t_end)

    gids = select_gids(sim, group, sample)
    times = sim.spikes.get(gids=gids, t_start=t_start, t_end=t_end).index.values

    if binsize is None:
        # heuristic for a nice bin size (~100 spikes per bin on average)
        binsize = min(50.0, (t_end - t_start) / ((len(times) / 100.) + 1.))

    bins = np.append(np.arange(t_start, t_end, binsize), t_end)
    hist, bin_edges = np.histogram(times, bins=bins)
    freq = 1.0 * hist / len(gids) / (0.001 * binsize)

    if ax is None:
        ax = plt.gca()
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('PSTH [Hz]')

    ax.plot(0.5 * (bin_edges[1:] + bin_edges[:-1]), freq, label=label, drawstyle='steps-mid')
    return ax


def raster(sim, group=None, sample=1000, t_start=None, t_end=None, groupby=None, label=None,
           ax=None):  # pragma: no cover
    """Spikes raster plot.

    The plot's y limit is set to the number of gids inside the circuit allowing comparison between
    2 simulations for the same circuit. The plot's x limit is set to the global simulation time
    range.

    Args:
        sim: Bluepy simulation
        group: cell group of interest, like in `bluepy.CellCollection.ids`
        sample: sample size (default = 1000, None for no sampling)
        t_start, t_end: time range of interest for the spike report query (not x the axis limit)
        groupby: use different colours for the different groups of cells.
        label: label to use for the plot legend (ignored if `groupby` is specified)
        ax: matplotlib Axes to draw on (if not specified, pyplot.gca() is used)

    Returns:
        matplotlib Axes with spike raster plot
    """
    t_start, t_end = check_times(sim, t_start, t_end)

    def _plot_spikes(gids, label=None):
        """ Scatter plot spike times for given GIDs. """
        spikes = sim.spikes.get(gids=gids, t_start=t_start, t_end=t_end)
        ts = spikes.index
        ys = spikes.to_numpy()
        ax.scatter(ts, ys, s=10, marker='|', label=label)

    gids = select_gids(sim, group, sample)

    if ax is None:
        ax = plt.gca()
        ax.xaxis.grid()
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("GID")
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_xlim(sim.t_start, sim.t_end)
        ax.set_ylim(0, len(sim.circuit.cells.ids()))

    if groupby is None:
        _plot_spikes(gids, label=label)
    else:
        prop = sim.circuit.cells.get(gids, groupby)
        for value, subgroup in prop.groupby(prop):
            if subgroup.empty:
                # grouping by Categorical property can produce empty groups
                continue
            gids = subgroup.index.values
            _plot_spikes(gids, label=value)
    return ax


def isi(sim, group=None, sample=1000, t_start=None, t_end=None,
        freq=False, binsize=None, label=None, ax=None):  # pragma: no cover
    # pylint: disable=too-many-arguments
    """
    Interspike interval histogram.

    Args:
        sim: Bluepy simulation
        group: cell group of interest, like in `bluepy.CellCollection.ids`
        sample: sample size (default = 1000, None for no sampling)
        t_start, t_end: time range of interest
        freq: use inverse interspike interval times (Hz)
        binsize: bin size (milliseconds or Hz)
        label: label to use for the plot legend
        ax: matplotlib Axes to draw on (if not specified, pyplot.gca() is used)

    Returns:
        matplotlib Axes with interspike interval histogram.
    """
    t_start, t_end = check_times(sim, t_start, t_end)

    gids = select_gids(sim, group, sample)
    spikes = sim.spikes.get(gids=gids, t_start=t_start, t_end=t_end)

    values = [np.diff(gid_spikes.index.values)
              for _, gid_spikes in spikes.groupby(spikes)]
    if values:
        values = np.concatenate(values)
    else:
        values = np.array(values)

    if freq:
        values = values[values > 0]  # filter out zero intervals (well, you never know)
        values = 1000.0 / values

    if binsize is None:
        bins = 'auto'
    else:
        bins = np.arange(0, np.max(values), binsize)

    if ax is None:
        ax = plt.gca()
        if freq:
            ax.set_xlabel('Frequency [Hz]')
        else:
            ax.set_xlabel('Interspike interval [ms]')
        ax.set_ylabel('Bin weight')

    ax.hist(values, bins=bins, edgecolor='black', density=True, label=label)
    return ax


def voltage_collage(sim, report_name, group=None, sample=10, t_start=None,
                    t_end=None, t_step=None):  # pragma: no cover
    # pylint: disable=too-many-locals
    """
    Voltage traces collage.

    Args:
        sim: Bluepy simulation
        report_name: report to use
        group: cell group of interest, like in `bluepy.CellCollection.ids`
        sample: sample size (default = 10, None for no sampling)
        t_start, t_end: time range of interest
        t_step: X-axis (time) resolution

    Returns:
        (Figure, [Axes]) pair with voltage collage.
    """
    t_start, t_end = check_times(sim, t_start, t_end, report_name=report_name)

    gids = select_gids(sim, group, sample, report_name=report_name)
    ncells = len(gids)
    if ncells == 0:
        raise BluePyError("No GIDs to plot")
    etypes = sim.circuit.cells.get(gids, Cell.ETYPE)
    report = sim.report(report_name)
    fig, axs = plt.subplots(nrows=ncells, sharex=True, sharey=False, figsize=(6, 1.5 * ncells))
    axs = ensure_list(axs)
    for ax, gid, etype in zip(axs, gids, etypes):
        # Force the  y tick legend formatting. This avoids problem with data with very small spread.
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_ylabel(f"{gid}\n{etype}")
        data = report.get_gid(gid, t_start=t_start, t_end=t_end, t_step=t_step)
        ax.plot(np.array(data.index), data.values)
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(f"a{gid}\n{etype}")
        ax.grid()
    axs[-1].set_xlabel('Time [ms]')
    fig.text(0.02, 0.5, 'Voltage [mV]', va='center', rotation='vertical')
    return fig, axs


def firing_animation(sim, x_axis=Cell.X, y_axis=Cell.Y, group=None, sample=7000, t_start=None,
                     t_end=None, dt=20, x_limits=None, y_limits=None):  # pragma: no cover
    # pylint: disable=too-many-locals,too-many-arguments,anomalous-backslash-in-string
    """ Simple animation of simulation spikes

    Args:
        sim: Bluepy simulation
        x_axis : Cell enum that will determine the animation x_axis
        y_axis : Cell enum that will determine the animation y_axis
        group : Cell group of interest, like in `bluepy.CellCollection.ids`
        sample: Sample size (default = 7000, None for no sampling)
        t_start: Starting time of the simulation
        t_end: Ending time of the simulation
        dt : int (ms) the time bin size of each frame in the video
        x_limits: the x axis limits of the plot | None use the max and min for each axis
        y_limits: the y axis limits of the plot | None use the max and min for each axis

    Returns :
        a matplotlib animation and an axis object

    Notes:
        Usage in scripts:
        import matplotlib.pyplot as plt
        anim, ax = sim.plot.firing_animation()
        plt.show()
        # to save the animation : do not plt.show() and just anim.save('my_movie.mp4')

        Usage in notebooks:
        from IPython.display import HTML
        anim, ax = sim.plot.firing_animation()
        HTML(anim.to_html5_video())
    """
    from matplotlib.animation import FuncAnimation

    t_start, t_end = check_times(sim, t_start, t_end)

    if not isinstance(t_start, (int, float)) or not isinstance(t_end, (int, float)):
        raise BluePyError("t_start and t_end must be numerical values")
    if t_start > t_end:
        raise BluePyError("t_start must be smaller than t_end")

    def _check_axis(axis):
        """ Verifies axes values """
        axes = {Cell.X, Cell.Y, Cell.Z}
        if axis not in axes:
            raise BluePyError(f'{axis} is not a valid axis')

    _check_axis(x_axis)
    _check_axis(y_axis)

    gids = select_gids(sim, group, sample)
    if len(gids) == 0:
        raise BluePyError("No GIDs to plot")
    spikes = sim.spikes.get(gids, t_start=t_start, t_end=t_end)
    active_gids = np.unique(spikes.values)
    positions = sim.circuit.cells.get(active_gids, properties=[x_axis, y_axis])

    if x_limits is None:
        x_limits = [positions[x_axis].min(), positions[x_axis].max()]
    if y_limits is None:
        y_limits = [positions[y_axis].min(), positions[y_axis].max()]

    fig, ax = plt.subplots()
    dots = ax.plot([], [], '.k')

    def init():
        """ Init the animation axes """
        ax.set_title('time = ' + str(t_start) + ' ms')
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xlabel(rf'{x_axis} $\mu$m')  # noqa
        ax.set_ylabel(rf'{y_axis} $\mu$m')  # noqa
        return dots

    def update_animation(frame):
        """ Update the animation plots and axes"""
        ax.set_title('time = ' + str(frame * dt) + ' ms')
        mask = (spikes.index >= frame * dt) & (spikes.index <= (frame + 1) * dt)
        frame_gids = np.unique(spikes[mask].values)
        x = positions.loc[frame_gids, x_axis].values
        y = positions.loc[frame_gids, y_axis].values
        dots[0].set_data(x, y)
        return dots

    frames = list(range(int(t_start / dt), int(t_end / dt)))
    anim = FuncAnimation(fig, update_animation, frames=frames, init_func=init)
    return anim, ax


def trace(sim, report_name, group=None, sample=1000, t_start=None, t_end=None, t_step=1,
          plot_type='mean', label=None, ax=None):  # pragma: no cover
    # pylint: disable=too-many-arguments
    """ potential plot displaying the voltage as a function of time from a report

    Args:
        sim: Bluepy simulation
        report_name: Report containing the simulation
        group:  Cell group of interest, like in `bluepy.CellCollection.ids`
        sample: Sample size (default = 1000)
        t_start: Starting time of the simulation
        t_end: Ending time of the simulation
        t_step: time steps for the simulation (default=1)
        plot_type: string either 'all' or 'mean'
        label: Set the label of the plot
        ax: A plot axis object that will be updated
    """
    t_start, t_end = check_times(sim, t_start, t_end, report_name=report_name)

    if ax is None:
        ax = plt.gca()
        potential_axes_update(ax, plot_type)
    ax.set_xlim([t_start, t_end])

    data = get_report_data(sim, report_name, group, sample, t_start, t_end, t_step).T
    if plot_type == "mean":
        ax.plot(data.mean())
    elif plot_type == "all":
        if sample > 15:
            L.warning('Sample too big. We will only keep the first 15 gids.')
        for _, row in data[:15].iterrows():
            ax.plot(row)
    ax.set_title(label)
    ax.text(0.01, 0.92, group_to_str(group, nb_max=1),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='black', fontsize='small',
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1, 'edgecolor': 'none'})

    return ax


def multi_trace(sim, report_name, groups, sample=1000,
                t_start=None, t_end=None, t_step=1,
                plot_type='mean', label=None,
                xlegend=True, fig=None, gs_global=None):  # pragma: no cover

    # pylint: disable=too-many-arguments,too-many-locals
    """ Display multiple potential plot as a function of time

    Args:
        sim: Bluepy simulation
        report_name: Report containing the simulation
        groups: list of Cell group of interest. Ex: [42, 'All', [12,34,45]]
        sample: Sample size (default = 1000, None for no sampling)
        t_start: Starting time of the simulation
        t_end: Ending time of the simulation
        t_step: time steps for the simulation (default=1)
        plot_type: string either 'all' or 'mean'
        label: Title of the plot
        xlegend: Set the xlabel legend
        fig: A matplotlib figure object
        gs_global: A GridSpec to place this plot in a bigger figure

    Returns:
        a matplotlib figure object

    Notes:
        if the fig and gs_global are provided by the user then this plot is placed inside the
        gridspec from fig
    """
    t_start, t_end = check_times(sim, t_start, t_end, report_name=report_name)

    if fig is None:
        fig = get_figure()

    if gs_global is None:
        grid_spec = gridspec.GridSpec(len(groups), 1)
    else:
        grid_spec = gridspec.GridSpecFromSubplotSpec(len(groups), 1, subplot_spec=gs_global)

    for i, group in enumerate(groups):
        ax = fig.add_subplot(grid_spec[i])
        potential_axes_update(ax, plot_type, xlegend=xlegend and (i == len(groups) - 1))
        trace(sim, report_name, group, sample, t_start, t_end, t_step=t_step,
              ax=ax, plot_type=plot_type)
    fig.suptitle(label, fontsize=12)
    return fig


def spikeraster_and_psth(sim, report_name, groups, sample=7000, t_start=None, t_end=None, t_step=1,
                         plot_type="raster", label=None, xlegend=True, fig=None, gs_global=None,
                         gs_colorbar=None, vmin=None, vmax=None):  # pragma: no cover
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    """ Display multiple firing rate plots on top of a raster plot or cumulative plot

    Args:
        sim: Bluepy simulation
        report_name: Report containing the simulation; not used when plot_type is 'raster'
        groups: list of Cell group of interest. Ex: [42, 'All', [12,34,45]]
        sample: Sample size (default = 7000, None for no sampling)
        t_start: Starting time of the simulation
        t_end: Ending time of the simulation
        t_step: time steps for the simulation (default=1)
        plot_type: string either 'raster' or 'mean'
        label: Title of the plot
        xlegend: Set the xlabel legend
        fig: A matplotlib figure object
        gs_global: A GridSpec to place this plot in a bigger figure
        gs_colorbar: A GridSpec to place the colorbar for the 'mean' type of plots
        vmin: for plot_type=mean, all values < vmin will be plotted as = vmin
        vmax: for plot_type=mean, all values > vmax will be plotted as = vmax

    Returns:
        a matplotlib figure object containing the plot

    Notes:
        If the fig and gs_global are provided by the user then this plot is placed inside the
        gridspec from fig.
        The gs_colorbar is used to guaranty the correct placement of the colorbar for the
        'mean' kind of plots. This also guaranty time alignment in case of combination of
        plots.
    """
    if plot_type == 'raster':
        report_name = None

    t_start, t_end = check_times(sim, t_start, t_end, report_name=report_name)

    if fig is None:
        fig = get_figure()

    if gs_global is None:
        if plot_type == 'mean':
            grid_spec = gridspec.GridSpec(len(groups), 2, width_ratios=[30, 1])
            gs_colorbar = grid_spec[:, 1]
        else:
            grid_spec = gridspec.GridSpec(len(groups), 1)
    else:
        grid_spec = gridspec.GridSpecFromSubplotSpec(len(groups), 1, subplot_spec=gs_global)

    group_gids = []
    for i, group in enumerate(groups):
        group_gids.append(select_gids(sim, group, sample))

    if plot_type == "mean":
        group_data = {}
        dtypes = []
        for i, group in enumerate(groups):
            group_data[i] = get_report_data(
                sim, report_name, group_gids[i], None, t_start, t_end, t_step).T

            dtypes.extend(group_data[i].dtypes.values.tolist())

        info_result_type = np.finfo(np.result_type(*dtypes))
        fmin, fmax = info_result_type.max, info_result_type.min
        for i, group in enumerate(groups):
            if vmin is None:
                vmin = min(fmin, group_data[i].values.min())
            if vmax is None:
                vmax = max(fmax, group_data[i].values.max())

    for i, group in enumerate(groups):
        title = group_to_str(group, nb_max=1)
        current_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec[i, 0],
                                                      height_ratios=[1, 3], hspace=0.0)
        ax_up = fig.add_subplot(current_gs[0])
        ax_bottom = fig.add_subplot(current_gs[1])

        firing_rate_histogram(sim, group=group_gids[i], label=title, ax=ax_up, binsize=3)

        if plot_type == "mean":
            heatmap = ax_bottom.imshow(group_data[i].values, interpolation='bilinear',
                                       aspect='auto', cmap="jet",
                                       vmin=vmin, vmax=vmax,
                                       extent=[t_start, t_end, 0, len(group_data)])
            if i == 0 and gs_colorbar:
                ax_col = fig.add_subplot(gs_colorbar)
                fig.colorbar(heatmap, cax=ax_col, label="mV")

        elif plot_type == "raster":
            raster(sim, group=group_gids[i], groupby=Cell.MORPHOLOGY, ax=ax_bottom,
                   t_start=t_start, t_end=t_end)
            ax_bottom.set_xlim([t_start, t_end])
        else:
            raise BluePyError("Type must be 'mean' or 'raster'")

        ax_up.set_xlim([t_start, t_end])
        ax_up.set_xticklabels([])
        ax_bottom.set_yticklabels([])
        ax_bottom.set_ylabel(title, size="small")

        if xlegend and i == len(groups) - 1:
            ax_bottom.set_xlabel("Time [ms]")
        else:
            ax_bottom.set_xticklabels([])

    fig.suptitle(label, fontsize=12)
    return fig


def population_summary(sim, report_name, groups,
                       mean_sample=7000, superimpose_sample=3,
                       t_start=None, t_end=None, t_step=1, plot_type="raster",
                       label=None):  # pragma: no cover
    # pylint: disable=too-many-arguments
    """ A specific plot display combining spikeraster_and_psth and multi_potential plots

    Args:
        sim: Bluepy simulation
        report_name: Report containing the simulation
        groups: list of Cell group of interest. Ex: [42, 'All', [12,34,45]]
        mean_sample: Sample size for cumulative plots (default = 7000, None for no sampling)
        superimpose_sample: Number of superimpose graphs in potential plots (default=3, None for no
        sampling)
        t_start: Starting time of the simulation
        t_end: Ending time of the simulation
        t_step: time steps for the simulation (default=1)
        plot_type: string either 'raster' or 'mean' (default='raster')
        label: Title of the plot
        xlegend: Set the xlabel legend

    Returns:
        a matplotlib figure object containing the plot

    Notes:
        Since it is already a complex figure the fig and gridspec cannot be provided
    """
    t_start, t_end = check_times(sim, t_start, t_end, report_name=report_name)

    fig = get_figure()

    if plot_type == "mean":
        gs_global = gridspec.GridSpec(3, 2, height_ratios=[1, len(groups), len(groups)],
                                      width_ratios=[30, 1], hspace=0.1, wspace=0.03)
        gs_color = gs_global[1, 1]
    else:
        gs_global = gridspec.GridSpec(3, 1, height_ratios=[1, len(groups), len(groups)],
                                      hspace=0.1, wspace=0.03)
        gs_color = None

    ax_pot = fig.add_subplot(gs_global[0, 0])
    potential_axes_update(ax_pot, 'mean', xlegend=False)
    trace(sim, report_name, group=groups[0], sample=mean_sample, t_start=t_start,
          t_end=t_end, plot_type='mean',
          ax=ax_pot)

    spikeraster_and_psth(sim, report_name, groups, sample=mean_sample, t_start=t_start,
                         t_end=t_end, t_step=t_step, plot_type=plot_type, xlegend=False,
                         fig=fig, gs_global=gs_global[1, 0], gs_colorbar=gs_color)

    multi_trace(sim, report_name, groups, sample=superimpose_sample, t_start=t_start,
                t_end=t_end, plot_type='all', xlegend=True,
                fig=fig, gs_global=gs_global[2, 0])

    fig.suptitle(label, fontsize=12)
    return fig


def voltage_histogram(sim, report_name, group=None, sample=1000, t_start=None, t_end=None, t_step=1,
                      voltage_min=None, voltage_max=None, time_bin_size=20, voltage_bin_size=5,
                      log_scale=None, label=None):  # pragma: no cover
    # pylint: disable=too-many-arguments,too-many-locals
    """ potential plot displaying the voltage as a function of time from a report

    Args:
        sim: Bluepy simulation
        report_name: Report containing the simulation
        group:  Cell group of interest, like in `bluepy.CellCollection.ids`
        sample: Sample size (default = 1000, None for no sampling)
        t_start: Starting time of the simulation
        t_end: Ending time of the simulation
        t_step: time steps for the simulation (default=1)
        voltage_min: Minimum voltage to display
        voltage_max: Maximum voltage to display
        time_bin_size: Time bin size (default = 20)
        voltage_bin_size: Voltage bin size (default = 5)
        log_scale: Goes logscale
    """
    from matplotlib import colors

    t_start, t_end = check_times(sim, t_start, t_end, report_name=report_name)
    fig = get_figure()

    grid_spec = gridspec.GridSpec(2, 2, hspace=0.0, wspace=0.01, height_ratios=[1, 10],
                                  width_ratios=[30, 1])

    ax_top = fig.add_subplot(grid_spec[0, 0])
    firing_rate_histogram(sim, group=group, t_start=t_start, t_end=t_end, sample=sample, ax=ax_top,
                          binsize=3)
    ax_top.set_xlim([t_start, t_end])
    ax_top.set_xticklabels([])

    data = get_report_data(sim, report_name, group, sample, t_start, t_end, t_step).T.melt()

    voltage_min = data["value"].min() if voltage_min is None else voltage_min
    voltage_max = data["value"].max() if voltage_max is None else voltage_max
    if voltage_max - voltage_min < voltage_bin_size:
        L.warning("voltage_bin_size is bigger than  voltage_min - voltage_max")

    data = data[(data["value"] < voltage_max) & (data["value"] > voltage_min)]

    count, x, y = np.histogram2d(x=data["time"].values, y=data["value"].values,
                                 bins=[np.arange(t_start, t_end, time_bin_size),
                                       np.arange(voltage_min, voltage_max, voltage_bin_size)])
    xx, yy = np.meshgrid(x, y)

    ax_2d = fig.add_subplot(grid_spec[1, 0])

    # pylint: disable=redundant-keyword-arg
    heatmap = ax_2d.pcolormesh(xx, yy, count.T,
                               norm=colors.SymLogNorm(1, vmax=np.max(count),
                                                      vmin=np.min(count)) if log_scale else None)

    ax_2d.set_ylabel("Membrane potential [mV]")
    ax_2d.set_xlabel("Time [ms]")
    fig.colorbar(heatmap, cax=fig.add_subplot(grid_spec[1, 1]), label="nb. gids")
    fig.suptitle(label, fontsize=12)
    return fig
