"""
Graph
======

The :class:`Graph` widget is a widget for displaying plots. It supports
drawing multiple plot with different colors on the Graph. It also supports
axes titles, ticks, labeled ticks, grids and a log or linear representation on
both the x and y axis, independently.

To display a plot. First create a graph which will function as a "canvas" for
the plots. Then create plot objects e.g. MeshLinePlot and add them to the
graph.

To create a graph with x-axis between 0-100, y-axis between -1 to 1, x and y
labels of and X and Y, respectively, x major and minor ticks every 25, 5 units,
respectively, y major ticks every 1 units, full x and y grids and with
a red line plot containing a sin wave on this range::

    from kivy_garden.graph import Graph, MeshLinePlot
    graph = Graph(xlabel='X', ylabel='Y', x_ticks_minor=5,
                  x_ticks_major=25, y_ticks_major=1,
                  y_grid_label=True, x_grid_label=True, padding=5,
                  x_grid=True, y_grid=True, xmin=-0, xmax=100, ymin=-1, ymax=1)
    plot = MeshLinePlot(color=[1, 0, 0, 1])
    plot.points = [(x, sin(x / 10.)) for x in range(0, 101)]
    graph.add_plot(plot)

The MeshLinePlot plot is a particular plot which draws a set of points using
a mesh object. The points are given as a list of tuples, with each tuple
being a (x, y) coordinate in the graph's units.

You can create different types of plots other than MeshLinePlot by inheriting
from the Plot class and implementing the required functions. The Graph object
provides a "canvas" to which a Plot's instructions are added. The plot object
is responsible for updating these instructions to show within the bounding
box of the graph the proper plot. The Graph notifies the Plot when it needs
to be redrawn due to changes. See the MeshLinePlot class for how it is done.

The current availables plots are:

    * `MeshStemPlot`
    * `MeshLinePlot`
    * `SmoothLinePlot`
    * `MeshLinePlot`
    * `MeshStemPlot`
    * `LinePlot`
    * `SmoothLinePlot`
    * `OptimizedSmoothLinePlot`
    * `AALineStripPlot`
    * `MeshStripPlot`
    * `ContourPlot`
    * `ScatterPlot`
    * `PointPlot`
    * `LineAndMarkerPlot`
    - require Kivy 2.3.0

.. note::

    The graph uses a stencil view to clip the plots to the graph display area.
    As with the stencil graphics instructions, you cannot stack more than 8
    stencil-aware widgets.

"""

__all__ = (
    "Graph",
    "GraphAA",
    "Plot",
    "MeshLinePlot",
    "MeshStemPlot",
    "LinePlot",
    "SmoothLinePlot",
    "OptimizedSmoothLinePlot",
    "AALineStripPlot",
    "MeshStripPlot",
    "ContourPlot",
    "ScatterPlot",
    "PointPlot",
    "LineAndMarkerPlot",
)

from collections import deque
from itertools import chain
from math import log10, floor, ceil, sqrt
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.graphics import (
    Color,
    Ellipse,
    Fbo,
    Line,
    Mesh,
    Point,
    RenderContext,
    Rectangle,
    SmoothLine,
)
from kivy.graphics.instructions import InstructionGroup, Instruction
from kivy.graphics.opengl import glGetString, GL_EXTENSIONS
from kivy.graphics.texture import Texture

# from kivy.graphics.vertex_instructions import Line, SmoothLine, Ellipse
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.metrics import dp
from kivy.properties import (
    NumericProperty,
    BooleanProperty,
    BoundedNumericProperty,
    StringProperty,
    ListProperty,
    ObjectProperty,
    DictProperty,
    AliasProperty,
    ReferenceListProperty,
    ColorProperty,
    OptionProperty,
)
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stencilview import StencilView

try:
    import numpy as np
except ImportError:
    np = None

from ._version import __version__


def identity(x):
    """Return the input value unchanged."""
    return x


def exp10(x):
    """Return 10 raised to the power of x."""
    return 10**x


Builder.load_string(
    """
<GraphRotatedLabel>:
    canvas.before:
        PushMatrix
        Rotate:
            angle: root.angle
            axis: 0, 0, 1
            origin: root.center
    canvas.after:
        PopMatrix
"""
)


class GraphRotatedLabel(Label):
    """Label widget that can be rotated by a specified angle."""

    angle = NumericProperty(0)


class Axis(EventDispatcher):
    """Base class for graph axes."""

    pass


class XAxis(Axis):
    """X-axis representation."""

    pass


class YAxis(Axis):
    """Y-axis representation."""

    pass


Builder.load_string(
    """
<GraphLegend>:
    orientation: "vertical"
    pos_hint: {"top": 1, "right": 1}
    size: self.minimum_size
    spacing: self.parent.padding if self.parent else 0
    padding: (self.parent.padding * 2 + self.marker_width, \
        self.parent.padding, self.parent.padding, self.parent.padding) \
        if self.parent else (0, 0, 0, 0)
    canvas.before:
        Color:
            rgba: self.parent.background_color if self.parent else (0, 0, 0, 0)
        Rectangle:
            size: self.size
            pos: self.pos
        Color:
            rgba: self.parent.border_color if self.parent else (0, 0, 0, 0)
        Line:
            rectangle: self.pos + self.size
"""
)


class GraphLegend(BoxLayout):
    """The Legend of a Graph.

    Set the legend's pos_hint property to position it relative to the graph's
    plotting area. Here, the graph's padding property is taken into account. So, for example
    a value of {'top': 1, 'right': 1} (the default) will position the legend in the
    top right corner inside the plotting area with a distance set by the graph's padding
    property's value to the plotting area's edges.
    Note: The position of the window has no influence to the graph's size.
        Meaning, setting the legend's pos_hint property to {'x': 1, 'center_y': .5}
        will position the legend vertically centered to the right of the plotting
        area. But this will probably be outside of the graph widget's area, so the
        legend might lie outside of the screen or under another widget.

    Example:

        >>> legend = []
        >>> graph = Graph(xmin=0, xmax=10, ymin=-1, ymax=1)
        >>> plot = LinePlot(color=[1, 0, 0, 1])
        >>> plot.points = [(x / 10, sin(x / 10)) for x in range(-0, 101)]
        >>> graph.add_plot(plot)
        >>> legend.append(('sine', plot))
        >>> plot = LinePlot(color=[0, 1, 0, 1])
        >>> plot.points = [(x / 10, cos(x / 10)) for x in range(-0, 101)]
        >>> graph.add_plot(plot)
        >>> legend.append(('cosine', plot))
        >>> graph.legend = legend
        >>> graph.legend.pos_hint = {'top': 1, 'center_x': .5}
    """

    marker_width = BoundedNumericProperty(dp(20), min=0)
    """Maximum width of the markers in the legend.

    Each marker displayed in the legend is of the same size as the plot's markers,
    but with this value as a maximum width. For line plots, this will be the length of the
    line displayed in the legend.

    Negative values are not allowed.

    :data:`marker_width` is a :class:`kivy.properties.BoundedNumericProperty`,
    defaults to 20 dp.
    """

    marker_height = BoundedNumericProperty(dp(12), min=0)
    """Maximum height of the markers in the legend.

    Each marker displayed in the legend is of the same size as the plot's markers,
    but with this value as a maximum height.

    Negative values are not allowed.

    :data:`marker_height` is a :class:`kivy.properties.BoundedNumericProperty`,
    defaults to 12 dp.
    """

    marker_size = ReferenceListProperty(marker_width, marker_height)
    """Maximum size of the markers in the legend.

    :data:`marker_size` is a :class:`kivy.properties.ReferenceListProperty` of
    (:data:`marker_width`, :data:`marker_height`) properties.
    """


class Graph(Widget):
    """Graph class for (real-time) plotting applications."""

    # Triggers for different update types
    _trigger = ObjectProperty(None)  # Full graphics reload
    _trigger_size = ObjectProperty(None)  # Size/position updates only
    _trigger_color = ObjectProperty(None)  # Color updates only

    # Widget references for labels
    _xlabel = ObjectProperty(None, allownone=True)  # X-axis label widget
    _ylabel = ObjectProperty(None, allownone=True)  # Y-axis label widget
    _x_grid_label = ListProperty([])  # X-axis tick mark labels
    _y_grid_label = ListProperty([])  # Y-axis tick mark labels

    # Mesh objects for drawing
    _mesh_ticks = ObjectProperty(None)  # Mesh for ticks/grids
    _mesh_rect = ObjectProperty(None)  # Mesh for surrounding rectangle

    # Tick locations (in axis min-max range)
    _ticks_majorx = ListProperty([])
    _ticks_minorx = ListProperty([])
    _ticks_majory = ListProperty([])
    _ticks_minory = ListProperty([])

    # Optimization flags
    _dirty_full = BooleanProperty(False)
    _dirty_size = BooleanProperty(False)
    _dirty_colors = BooleanProperty(False)
    _dirty_legend = BooleanProperty(False)

    # Cached calculations
    _cached_bounds: Tuple[float, float, float, float] = (0, 0, 0, 0)
    _cached_params_hash: int = 0

    # Color properties
    tick_color = ListProperty([0.25, 0.25, 0.25, 1])
    """Color of the grid/ticks, default to 1/4 grey."""

    background_color = ListProperty([0, 0, 0, 0])
    """Color of the background, defaults to transparent."""

    border_color = ListProperty([1, 1, 1, 1])
    """Color of the border, defaults to white."""

    # Label configuration properties
    label_options = DictProperty()
    """Label options that will be passed to :class:`kivy.uix.Label`.

    These options are applied to axis labels, tick labels and legend labels,
    if not overwritten by :data:`tick_label_options` or :data:`legend_label_options`
    respectively.

    :data:`label_options` is an :class:`kivy.properties.DictProperty`
    and defaults to the empty dict.
    """

    tick_label_options = DictProperty()
    """Additional label options for the tick labels.

    These options are applied to tick labels and do overwrite the values in
    :data:`label_options` for the tick labels.

    :data:`tick_label_options` is an :class:`kivy.properties.DictProperty`
    and defaults to the empty dict.
    """

    legend_label_options = DictProperty()
    """Additional label options for the legend labels.

    These options are applied to legend labels and do overwrite the values in
    :data:`label_options` for the legend labels.

    :data:`legend_label_options` is an :class:`kivy.properties.DictProperty`
    and defaults to the empty dict.
    """

    def _get_legend(self) -> Optional["GraphLegend"]:
        """Get the current legend."""
        return self._legend

    def _set_legend(
        self, legend: Union[List[Tuple[str, Any]], "GraphLegend", None]
    ) -> bool:
        """Set or update the legend with plot data."""
        if legend and self._legend:
            # Clear existing legend children efficiently
            while self._legend.children:
                self._legend.remove_widget(self._legend.children[0])
        elif legend:
            # Create new legend
            self._legend = GraphLegend()
            self._legend.bind(
                pos=self._mark_legend_dirty,
                size=self._mark_legend_dirty,
                pos_hint=self._mark_legend_dirty,
                marker_size=self._mark_legend_dirty,
            )
            self.add_widget(self._legend)
        elif self._legend:
            # Remove existing legend
            self.remove_widget(self._legend)
            self._legend = None
            self._legend_plots = []
            return True
        else:
            return False

        # Build legend content
        self._legend_plots = []
        self._legend.canvas.clear()

        for name, plot in legend:
            # Create label with combined options
            options = self.label_options.copy()
            options.update(**self.legend_label_options)
            label = Label(text=name, **options, size_hint=(None, None))
            label.bind(
                texture_size=lambda instance, size: setattr(instance, "size", size)
            )

            # Add plot drawings to legend
            drawings = plot.create_legend_drawings()
            for drawing in drawings:
                self._legend.canvas.add(drawing)

            self._legend_plots.append(plot)
            self._legend.add_widget(label)
        return True

    legend: Union[List[Tuple[str, Any]], "GraphLegend", None] = AliasProperty(
        _get_legend, _set_legend
    )
    """Legend of graph's plots.

    You set the legend with an iterable yielding tuples containing the name
    and :class:`Plot` instance, you want to be displayed.
    Getting the legend returns an instance of :class:`GraphLegend`.

    See the :class:`GraphLegend` class documentation for a usage example.

    :data:`legend` is a :class:`~kivy.properties.AliasProperty`,
    defaults to None.
    """

    _with_stencilbuffer = BooleanProperty(True)
    """Whether :class:`Graph`'s FBO should use FrameBuffer (True) or not (False).

    .. warning:: This property is internal and so should be used with care.
    It can break some other graphic instructions used by the :class:`Graph`,
    for example you can have problems when drawing :class:`SmoothLinePlot`
    plots, so use it only when you know what exactly you are doing.

    :data:`_with_stencilbuffer` is a :class:`~kivy.properties.BooleanProperty`,
    defaults to True.
    """

    def __init__(self, **kwargs):
        # Initialize legend-related attributes
        self._legend: Optional["GraphLegend"] = None
        self._legend_plots: List[Any] = []

        super(Graph, self).__init__(**kwargs)

        # Create FBO for off-screen rendering
        with self.canvas:
            self._fbo = Fbo(size=self.size, with_stencilbuffer=self._with_stencilbuffer)

        # Set up FBO drawing instructions
        with self._fbo:
            self._background_color = Color(*self.background_color)
            self._background_rect = Rectangle(size=self.size)
            self._mesh_ticks_color = Color(*self.tick_color)
            self._mesh_ticks = Mesh(mode="lines")
            self._mesh_rect_color = Color(*self.border_color)
            self._mesh_rect = Mesh(mode="line_strip")

        # Add FBO texture to main canvas
        with self.canvas:
            Color(1, 1, 1)
            self._fbo_rect = Rectangle(size=self.size, texture=self._fbo.texture)

        # Initialize rectangle mesh
        mesh = self._mesh_rect
        mesh.vertices = [0] * (5 * 4)
        mesh.indices = list(range(5))

        # Create plot area with stencil clipping
        self._plot_area = StencilView()
        self.add_widget(self._plot_area)

        # Create update triggers with immediate execution
        self._trigger = Clock.create_trigger(self._process_full_redraw, -1)
        self._trigger_size = Clock.create_trigger(self._process_size_update, -1)
        self._trigger_color = Clock.create_trigger(self._process_color_update, -1)
        self._trigger_legend = Clock.create_trigger(self._process_legend_update, -1)

        # Bind properties to appropriate triggers
        self.bind(
            center=self._mark_size_dirty,
            padding=self._mark_size_dirty,
            precision=self._mark_full_dirty,
            plots=self._mark_size_dirty,
            x_grid=self._mark_size_dirty,
            y_grid=self._mark_size_dirty,
            draw_border=self._mark_size_dirty,
        )

        self.bind(
            xmin=self._mark_full_dirty,
            xmax=self._mark_full_dirty,
            xlog=self._mark_full_dirty,
            x_ticks_major=self._mark_full_dirty,
            x_ticks_minor=self._mark_full_dirty,
            xlabel=self._mark_full_dirty,
            x_grid_label=self._mark_full_dirty,
            ymin=self._mark_full_dirty,
            ymax=self._mark_full_dirty,
            ylog=self._mark_full_dirty,
            y_ticks_major=self._mark_full_dirty,
            y_ticks_minor=self._mark_full_dirty,
            ylabel=self._mark_full_dirty,
            y_grid_label=self._mark_full_dirty,
            label_options=self._mark_full_dirty,
            x_ticks_angle=self._mark_full_dirty,
            tick_label_options=self._mark_full_dirty,
            legend_label_options=self._mark_full_dirty,
        )

        self.bind(
            tick_color=self._mark_color_dirty,
            background_color=self._mark_color_dirty,
            border_color=self._mark_color_dirty,
        )

        self.bind(
            legend=self._mark_legend_dirty,
            view_pos=self._mark_legend_dirty,
            view_size=self._mark_legend_dirty,
            pos=self._mark_legend_dirty,
        )

        # Initial draw
        self._mark_full_dirty()

    def _mark_full_dirty(self, *args: Any) -> None:
        """Mark full redraw as needed."""
        self._dirty_full = True
        self._trigger()

    def _mark_size_dirty(self, *args: Any) -> None:
        """Mark size update as needed."""
        self._dirty_size = True
        self._trigger_size()

    def _mark_color_dirty(self, *args: Any) -> None:
        """Mark color update as needed."""
        self._dirty_colors = True
        self._trigger_color()

    def _mark_legend_dirty(self, *args: Any) -> None:
        """Mark legend update as needed."""
        self._dirty_legend = True
        self._trigger_legend()

    def _process_full_redraw(self, *args: Any) -> None:
        """Process full redraw request."""
        if self._dirty_full:
            self._redraw_all()
            self._dirty_full = False

    def _process_size_update(self, *args: Any) -> None:
        """Process size update request."""
        if self._dirty_size:
            self._redraw_size()
            self._dirty_size = False

    def _process_color_update(self, *args: Any) -> None:
        """Process color update request."""
        if self._dirty_colors:
            self._update_colors()
            self._dirty_colors = False

    def _process_legend_update(self, *args: Any) -> None:
        """Process legend update request."""
        if self._dirty_legend:
            self._update_legend()
            self._dirty_legend = False

    def add_widget(self, widget: Widget) -> None:
        """Add widget, handling special case for plot area."""
        if widget is self._plot_area:
            canvas = self.canvas
            self.canvas = self._fbo
        super(Graph, self).add_widget(widget)
        if widget is self._plot_area:
            self.canvas = canvas

    def remove_widget(self, widget: Widget) -> None:
        """Remove widget, handling special case for plot area."""
        if widget is self._plot_area:
            canvas = self.canvas
            self.canvas = self._fbo
        super(Graph, self).remove_widget(widget)
        if widget is self._plot_area:
            self.canvas = canvas

    def _get_ticks(
        self,
        major: Union[float, List[float], Tuple[float, ...]],
        minor: Union[float, List[float], Tuple[float, ...]],
        log: bool,
        s_min: float,
        s_max: float,
    ) -> Tuple[List[float], List[float]]:
        """Calculate tick positions for major and minor ticks."""
        # Ensure major and minor are same type
        if isinstance(major, (list, tuple)) != isinstance(minor, (list, tuple)):
            minor = type(major)()

        points_major: List[float] = []
        points_minor: List[float] = []

        if isinstance(major, (list, tuple)):
            # Use provided tick positions
            points_major = [
                p for p in major if s_min <= p <= s_max or s_min >= p >= s_max
            ]
            points_minor = [
                p
                for p in minor
                if (s_min <= p <= s_max or s_min >= p >= s_max)
                and p not in points_major
            ]
            if log:
                points_major = [log10(p) for p in points_major]
                points_minor = [log10(p) for p in points_minor]

        elif major > 0:
            minor = max(minor, 0)

            if log and s_max > s_min:
                # Logarithmic scale tick calculation
                s_min_log = log10(s_min)
                s_max_log = log10(s_max)

                # Count decades in range
                n_decades = floor(s_max_log - s_min_log)

                # Handle fractional part of last decade
                if floor(s_min_log + n_decades) != floor(s_max_log):
                    n_decades += 1 - (
                        10 ** (s_min_log + n_decades + 1) - 10**s_max_log
                    ) / 10 ** floor(s_max_log + 1)
                else:
                    n_decades += (
                        10**s_max_log - 10 ** (s_min_log + n_decades)
                    ) / 10 ** floor(s_max_log + 1)

                # Calculate number of ticks needed
                n_ticks_major = n_decades / float(major)
                n_ticks = (
                    int(floor(n_ticks_major * (minor if minor >= 1.0 else 1.0))) + 2
                )
                decade_dist = major / float(minor if minor else 1.0)

                points_minor = [0] * n_ticks
                points_major = [0] * n_ticks
                k = k2 = 0
                min_pos = 0.1 - 0.00001 * decade_dist
                s_min_low = floor(s_min_log)

                # Calculate first tick position
                start_dec = (
                    ceil((10 ** (s_min_log - s_min_low - 1)) / decade_dist)
                    * decade_dist
                )
                count_min = 0 if not minor else floor(start_dec / decade_dist) % minor
                start_dec += s_min_low
                count = 0

                # Generate tick positions
                while True:
                    pos_dec = start_dec + decade_dist * count
                    pos_dec_low = floor(pos_dec)
                    diff = pos_dec - pos_dec_low
                    zero = abs(diff) < 0.001 * decade_dist

                    if zero:
                        pos_log = pos_dec_low
                    else:
                        pos_log = log10((pos_dec - pos_dec_low) * 10 ** ceil(pos_dec))

                    if pos_log > s_max_log:
                        break

                    count += 1
                    if zero or diff >= min_pos:
                        if minor and not count_min % minor:
                            points_major[k] = pos_log
                            k += 1
                        else:
                            points_minor[k2] = pos_log
                            k2 += 1
                    count_min += 1

            elif not log and s_max != s_min:
                # Linear scale tick calculation
                tick_dist = major / float(minor if minor else 1.0)
                n_ticks = int(floor(abs((s_max - s_min) / tick_dist)) + 1)
                tick_dist = abs(tick_dist) * (1 if s_min < s_max else -1)

                points_major = [0] * int(floor(abs((s_max - s_min) / float(major))) + 1)
                points_minor = [0] * (n_ticks - len(points_major) + 1)
                k = k2 = 0

                for m in range(0, n_ticks):
                    if minor and m % minor:
                        points_minor[k2] = m * tick_dist + s_min
                        k2 += 1
                    else:
                        points_major[k] = m * tick_dist + s_min
                        k += 1
            else:
                k = k2 = 1

            # Trim unused positions
            del points_major[k:]
            del points_minor[k2:]

        return (
            sorted(points_major, reverse=s_min > s_max),
            sorted(points_minor, reverse=s_min > s_max),
        )

    def _update_labels(self) -> Tuple[float, float, float, float]:
        """Update position and content of axis labels and tick labels."""
        xlabel = self._xlabel
        ylabel = self._ylabel
        x, y = self.x, self.y
        width, height = self.width, self.height
        padding = self.padding

        # Calculate initial boundaries
        x_next = padding + x
        y_next = padding + y
        xextent = width + x
        yextent = height + y

        # Get axis parameters
        ymin, ymax, xmin = self.ymin, self.ymax, self.xmin
        precision = self.precision
        x_overlap = y_overlap = False

        # Position axis labels
        if xlabel:
            xlabel.text = self.xlabel
            xlabel.texture_update()
            xlabel.size = xlabel.texture_size
            xlabel.pos = (int(x + width / 2.0 - xlabel.width / 2.0), int(padding + y))
            y_next += padding + xlabel.height

        if ylabel:
            ylabel.text = self.ylabel
            ylabel.texture_update()
            ylabel.size = ylabel.texture_size
            ylabel.x = padding + x - (ylabel.width / 2.0 - ylabel.height / 2.0)
            x_next += padding + ylabel.height

        # Get tick data
        xpoints = self._ticks_majorx
        xlabels = self._x_grid_label
        xlabel_grid = self.x_grid_label
        ylabel_grid = self.y_grid_label
        ypoints = self._ticks_majory
        ylabels = self._y_grid_label

        # Update Y-axis tick labels
        if len(ylabels) and ylabel_grid:
            funcexp = lambda x: 10**x if self.ylog else lambda x: x
            funclog = log10 if self.ylog else lambda x: x

            # Determine text generation function
            if callable(self.y_grid_label):
                get_text = lambda k: self.y_grid_label(funcexp(ypoints[k]))
            elif isinstance(self.y_grid_label, (tuple, list, str)):
                get_text = lambda k: self.y_grid_label[k]
            else:
                get_text = lambda k: precision % funcexp(ypoints[k])

            # Calculate label dimensions and positions
            ylabels[0].text = get_text(0)
            ylabels[0].texture_update()
            y1 = ylabels[0].texture_size

            y_start = (
                y_next
                + (padding + y1[1] if len(xlabels) and xlabel_grid else 0)
                + (padding + y1[1] if not y_next else 0)
            )
            yextent = y + height - padding - y1[1] / 2.0

            ymin_log = funclog(ymin)
            ratio = (yextent - y_start) / float(funclog(ymax) - ymin_log)
            y_start -= y1[1] / 2.0
            y1 = y1[0]

            # Update all Y labels efficiently
            texts_to_update = []
            for k in range(len(ylabels)):
                try:
                    text = get_text(k)
                    texts_to_update.append((k, text))
                except IndexError:
                    break

            for k, text in texts_to_update:
                ylabels[k].text = text
                ylabels[k].texture_update()
                ylabels[k].size = ylabels[k].texture_size
                y1 = max(y1, ylabels[k].texture_size[0])

            positions_to_update = []
            for k in range(len(ylabels)):
                pos = (
                    int(x_next) - ylabels[k].width + y1,
                    int(y_start + (ypoints[k] - ymin_log) * ratio),
                )
                positions_to_update.append((k, pos))

            for k, pos in positions_to_update:
                ylabels[k].pos = pos

            # Check for overlap
            if len(ylabels) > 1 and ylabels[0].top > ylabels[1].y:
                y_overlap = True
            else:
                x_next += y1 + padding

        # Update X-axis tick labels
        if len(xlabels) and xlabel_grid:
            funcexp = lambda x: 10**x if self.xlog else lambda x: x
            funclog = log10 if self.xlog else lambda x: x

            # Determine text generation function
            if callable(self.x_grid_label):
                get_text = lambda k: self.x_grid_label(funcexp(xpoints[k]))
            elif isinstance(self.x_grid_label, (tuple, list, str)):
                get_text = lambda k: self.x_grid_label[k]
            else:
                get_text = lambda k: precision % funcexp(xpoints[k])

            # Calculate boundaries for X labels
            xlabels[0].text = get_text(-1)
            xlabels[0].texture_update()
            xextent = x + width - xlabels[0].texture_size[0] / 2.0 - padding

            if not x_next:
                xlabels[0].text = get_text(0)
                xlabels[0].texture_update()
                x_next = padding + xlabels[0].texture_size[0] / 2.0

            xmin_log = funclog(xmin)
            ratio = (xextent - x_next) / float(funclog(self.xmax) - xmin_log)
            right = -1

            # Position X labels and check for overlap
            texts_and_positions = []
            for k in range(len(xlabels)):
                try:
                    text = get_text(k)
                    half_ts = 0  # Will be calculated during update
                    texts_and_positions.append((k, text, half_ts))
                except IndexError:
                    pass

            # Update textures efficiently
            for k, text, half_ts in texts_and_positions:
                xlabels[k].text = text
                xlabels[k].texture_update()
                xlabels[k].size = xlabels[k].texture_size
                half_ts = xlabels[k].texture_size[0] / 2.0
                pos = (
                    int(x_next + (xpoints[k] - xmin_log) * ratio - half_ts),
                    int(y_next),
                )
                xlabels[k].pos = pos

                if xlabels[k].x < right:
                    x_overlap = True
                    break
                right = xlabels[k].right

            if not x_overlap:
                y_next += padding + (xlabels[0].texture_size[1] if xlabels else 0)

        # Re-center axis labels
        if xlabel:
            xlabel.x = int(x_next + (xextent - x_next) / 2.0 - xlabel.width / 2.0)
        if ylabel:
            ylabel.y = int(y_next + (yextent - y_next) / 2.0 - ylabel.height / 2.0)
            ylabel.angle = 90

        # Hide overlapping labels
        if x_overlap:
            for label in xlabels:
                label.text = ""
        if y_overlap:
            for label in ylabels:
                label.text = ""

        return x_next - x, y_next - y, xextent - x, yextent - y

    def _update_ticks(self, size: Tuple[float, float, float, float]) -> None:
        """Update tick and grid line positions."""
        # Update border rectangle
        mesh = self._mesh_rect
        vert = mesh.vertices

        if self.draw_border:
            s0, s1, s2, s3 = size  # left, bottom, right, top
            vert[0:20] = [
                s0,
                s1,
                0,
                0,  # bottom-left
                s2,
                s1,
                0,
                0,  # bottom-right
                s2,
                s3,
                0,
                0,  # top-right
                s0,
                s3,
                0,
                0,  # top-left
                s0,
                s1,
                0,
                0,  # close
            ]
            mesh.indices = list(range(5))
        else:
            vert[0:20] = [0] * 20
        mesh.vertices = vert

        # Update tick positions
        mesh = self._mesh_ticks
        vert = mesh.vertices
        start = 0

        # Get tick arrays and axis parameters
        xpoints, ypoints = self._ticks_majorx, self._ticks_majory
        xpoints2, ypoints2 = self._ticks_minorx, self._ticks_minory
        ylog, xlog = self.ylog, self.xlog
        xmin, xmax = self.xmin, self.xmax
        ymin, ymax = self.ymin, self.ymax

        # Convert to log scale if needed
        if xlog:
            xmin, xmax = log10(xmin), log10(xmax)
        if ylog:
            ymin, ymax = log10(ymin), log10(ymax)

        # Pre-calculate ratios for better performance
        x_ratio = (size[2] - size[0]) / float(xmax - xmin) if xmax != xmin else 0
        y_ratio = (size[3] - size[1]) / float(ymax - ymin) if ymax != ymin else 0

        # Collect all vertex data
        new_vertices = []

        # Draw major X ticks
        if len(xpoints):
            top = size[3] if self.x_grid else dp(12) + size[1]
            for point in xpoints:
                x_pos = size[0] + (point - xmin) * x_ratio
                new_vertices.extend([x_pos, size[1], 0, 0, x_pos, top, 0, 0])

        # Draw minor X ticks
        if len(xpoints2):
            top = dp(8) + size[1]
            for point in xpoints2:
                x_pos = size[0] + (point - xmin) * x_ratio
                new_vertices.extend([x_pos, size[1], 0, 0, x_pos, top, 0, 0])

        # Draw major Y ticks
        if len(ypoints):
            top = size[2] if self.y_grid else dp(12) + size[0]
            for point in ypoints:
                y_pos = size[1] + (point - ymin) * y_ratio
                new_vertices.extend([size[0], y_pos, 0, 0, top, y_pos, 0, 0])

        # Draw minor Y ticks
        if len(ypoints2):
            top = dp(8) + size[0]
            for point in ypoints2:
                y_pos = size[1] + (point - ymin) * y_ratio
                new_vertices.extend([size[0], y_pos, 0, 0, top, y_pos, 0, 0])

        # Update mesh with all vertices at once
        mesh.vertices = new_vertices
        mesh.indices = list(range(len(new_vertices) // 4))

    # Axis management properties
    x_axis = ListProperty([None])
    y_axis = ListProperty([None])

    def get_x_axis(self, axis: int = 0) -> Tuple[bool, float, float]:
        """Get X-axis parameters for specified axis index."""
        if axis == 0:
            return self.xlog, self.xmin, self.xmax
        info = self.x_axis[axis]
        return info["log"], info["min"], info["max"]

    def get_y_axis(self, axis: int = 0) -> Tuple[bool, float, float]:
        """Get Y-axis parameters for specified axis index."""
        if axis == 0:
            return self.ylog, self.ymin, self.ymax
        info = self.y_axis[axis]
        return info["log"], info["min"], info["max"]

    def add_x_axis(
        self, xmin: float, xmax: float, xlog: bool = False
    ) -> Dict[str, Any]:
        """Add additional X-axis with specified parameters."""
        data = {"log": xlog, "min": xmin, "max": xmax}
        self.x_axis.append(data)
        return data

    def add_y_axis(
        self, ymin: float, ymax: float, ylog: bool = False
    ) -> Dict[str, Any]:
        """Add additional Y-axis with specified parameters."""
        data = {"log": ylog, "min": ymin, "max": ymax}
        self.y_axis.append(data)
        return data

    def _update_plots(self, size: Tuple[float, float, float, float]) -> None:
        """Update all plots with current size and axis parameters."""
        for plot in self.plots:
            xlog, xmin, xmax = self.get_x_axis(plot.x_axis)
            ylog, ymin, ymax = self.get_y_axis(plot.y_axis)
            plot.update(xlog, xmin, xmax, ylog, ymin, ymax, size)

    def _update_colors(self, *args: Any) -> None:
        """Update color properties of graph elements."""
        self._mesh_ticks_color.rgba = tuple(self.tick_color)
        self._background_color.rgba = tuple(self.background_color)
        self._mesh_rect_color.rgba = tuple(self.border_color)

    def _redraw_all(self, *args: Any) -> None:
        """Perform complete redraw of graph including labels and ticks."""
        # Update axis labels and ticks
        xpoints_major, xpoints_minor = self._redraw_x(*args)
        ypoints_major, ypoints_minor = self._redraw_y(*args)

        # Update legend labels if present
        if self._legend:
            options = self.label_options.copy()
            options.update(**self.legend_label_options)
            for child in self._legend.children:
                for key, value in options.items():
                    setattr(child, key, value)

        # Pre-size mesh for ticks to avoid reallocations
        n_points = (
            len(xpoints_major)
            + len(xpoints_minor)
            + len(ypoints_major)
            + len(ypoints_minor)
        )

        if (
            hasattr(self._mesh_ticks, "vertices")
            and len(self._mesh_ticks.vertices) != n_points * 8
        ):
            self._mesh_ticks.vertices = [0] * (n_points * 8)
            self._mesh_ticks.indices = list(range(n_points * 2))

        self._redraw_size()

    def _redraw_x(self, *args: Any) -> Tuple[List[float], List[float]]:
        """Redraw X-axis labels and calculate tick positions."""
        # Handle X-axis label
        if self.xlabel:
            xlabel = self._xlabel
            if not xlabel:
                xlabel = Label()
                self.add_widget(xlabel)
                self._xlabel = xlabel
            # Apply label options
            for key, value in self.label_options.items():
                setattr(xlabel, key, value)
        else:
            xlabel = self._xlabel
            if xlabel:
                self.remove_widget(xlabel)
                self._xlabel = None

        # Calculate tick positions
        grids = self._x_grid_label
        xpoints_major, xpoints_minor = self._get_ticks(
            self.x_ticks_major, self.x_ticks_minor, self.xlog, self.xmin, self.xmax
        )
        self._ticks_majorx = xpoints_major
        self._ticks_minorx = xpoints_minor

        # Determine number of labels needed
        if not self.x_grid_label:
            n_labels = 0
        elif isinstance(self.x_grid_label, (tuple, list, str)):
            n_labels = min(len(xpoints_major), len(self.x_grid_label))
        else:
            n_labels = len(xpoints_major)

        # Remove excess labels efficiently
        if len(grids) > n_labels:
            for i in range(n_labels, len(grids)):
                self.remove_widget(grids[i])
            del grids[n_labels:]

        # Add new labels as needed
        grid_len = len(grids)
        if len(grids) < n_labels:
            grids.extend([None] * (n_labels - len(grids)))
            options = self.label_options.copy()
            options.update(**self.tick_label_options)

            for k in range(grid_len, n_labels):
                grids[k] = GraphRotatedLabel(angle=self.x_ticks_angle, **options)
                self.add_widget(grids[k])

        return xpoints_major, xpoints_minor

    def _redraw_y(self, *args: Any) -> Tuple[List[float], List[float]]:
        """Redraw Y-axis labels and calculate tick positions."""
        # Handle Y-axis label
        if self.ylabel:
            ylabel = self._ylabel
            if not ylabel:
                ylabel = GraphRotatedLabel()
                self.add_widget(ylabel)
                self._ylabel = ylabel
            # Apply label options
            for key, value in self.label_options.items():
                setattr(ylabel, key, value)
        else:
            ylabel = self._ylabel
            if ylabel:
                self.remove_widget(ylabel)
                self._ylabel = None

        # Calculate tick positions
        grids = self._y_grid_label
        ypoints_major, ypoints_minor = self._get_ticks(
            self.y_ticks_major, self.y_ticks_minor, self.ylog, self.ymin, self.ymax
        )
        self._ticks_majory = ypoints_major
        self._ticks_minory = ypoints_minor

        # Determine number of labels needed
        if not self.y_grid_label:
            n_labels = 0
        elif isinstance(self.y_grid_label, (tuple, list, str)):
            n_labels = min(len(ypoints_major), len(self.y_grid_label))
        else:
            n_labels = len(ypoints_major)

        # Remove excess labels efficiently
        if len(grids) > n_labels:
            for i in range(n_labels, len(grids)):
                self.remove_widget(grids[i])
            del grids[n_labels:]

        # Add new labels as needed
        grid_len = len(grids)
        if len(grids) < n_labels:
            grids.extend([None] * (n_labels - len(grids)))
            options = self.label_options.copy()
            options.update(**self.tick_label_options)

            for k in range(grid_len, n_labels):
                grids[k] = Label(**options)
                self.add_widget(grids[k])

        return ypoints_major, ypoints_minor

    def _redraw_size(self, *args: Any) -> None:
        """Update graph layout and plot positions based on current size."""
        self._clear_buffer()
        size = self._update_labels()

        # Update plot area position and size
        self.view_pos = self._plot_area.pos = (size[0], size[1])
        self.view_size = self._plot_area.size = (size[2] - size[0], size[3] - size[1])

        # Update FBO size efficiently
        if self.size[0] and self.size[1]:
            target_size = self.size
        else:
            target_size = (1, 1)  # Prevent GL errors

        if self._fbo.size != target_size:
            self._fbo.size = target_size

        # Update FBO rectangle
        self._fbo_rect.texture = self._fbo.texture
        self._fbo_rect.size = self.size
        self._fbo_rect.pos = self.pos
        self._background_rect.size = self.size

        # Update ticks and plots
        self._update_ticks(size)
        self._update_plots(size)

    def _update_legend(self, *_: Any) -> None:
        """Update legend position and marker drawings."""
        if not self.legend:
            return

        self.legend.size = self.legend.minimum_size
        pos, size = self.view_pos, self.view_size
        ph = self.legend.pos_hint

        # Calculate X position
        if "x" in ph or "center_x" in ph or "right" in ph:
            x = (
                size[0] * (ph.get("x", ph.get("center_x", ph.get("right"))))
                - (
                    self.legend.width
                    * (0 if "x" in ph else 0.5 if "center_x" in ph else 1)
                )
                - (self.padding * (-1 if "x" in ph else 0 if "center_x" in ph else 1))
            )
            self.legend.x = self.x + pos[0] + x

        # Calculate Y position
        if "y" in ph or "center_y" in ph or "top" in ph:
            y = (
                size[1] * (ph.get("y", ph.get("center_y", ph.get("top"))))
                - (
                    self.legend.height
                    * (0 if "y" in ph else 0.5 if "center_y" in ph else 1)
                )
                - (self.padding * (-1 if "y" in ph else 0 if "center_y" in ph else 1))
            )
            self.legend.y = self.y + pos[1] + y

        # Update legend markers
        for i, plot in enumerate(self._legend_plots):
            if i < len(self.legend.children):
                label = self.legend.children[-(i + 1)]
                drawing_center = (
                    label.x - self.legend.spacing - self.legend.marker_width / 2,
                    label.center_y,
                )
                plot.draw_legend(drawing_center, self.legend.marker_size)

    def _clear_buffer(self, *largs: Any) -> None:
        """Clear the FBO buffer."""
        fbo = self._fbo
        fbo.bind()
        fbo.clear_buffer()
        fbo.release()

    def add_plot(self, plot: Any) -> None:
        """Add a new plot to this graph.

        :Parameters:
            `plot`: Plot to add to this graph.

        >>> graph = Graph()
        >>> plot = MeshLinePlot(mode='line_strip', color=[1, 0, 0, 1])
        >>> plot.points = [(x / 10., sin(x / 50.)) for x in range(-0, 101)]
        >>> graph.add_plot(plot)
        """
        if plot in self.plots:
            return

        # Add plot drawing instructions to canvas
        add = self._plot_area.canvas.add
        for instr in plot.get_drawings():
            add(instr)

        plot.bind(on_clear_plot=self._clear_buffer)
        self.plots.append(plot)

    def remove_plot(self, plot: Any) -> None:
        """Remove a plot from this graph.

        :Parameters:
            `plot`: Plot to remove from this graph.

        >>> graph = Graph()
        >>> plot = MeshLinePlot(mode='line_strip', color=[1, 0, 0, 1])
        >>> plot.points = [(x / 10., sin(x / 50.)) for x in range(-0, 101)]
        >>> graph.add_plot(plot)
        >>> graph.remove_plot(plot)
        """
        if plot not in self.plots:
            return

        # Remove plot drawing instructions from canvas
        remove = self._plot_area.canvas.remove
        for instr in plot.get_drawings():
            remove(instr)

        plot.unbind(on_clear_plot=self._clear_buffer)
        self.plots.remove(plot)
        self._clear_buffer()

    def collide_plot(self, x: float, y: float) -> bool:
        """Determine if the given coordinates fall inside the plot area.

        Use `x, y = self.to_widget(x, y, relative=True)` to first convert into
        widget coordinates if it's in window coordinates because it's assumed
        to be given in local widget coordinates, relative to the graph's pos.

        :Parameters:
            `x, y`: The coordinates to test.
        """
        adj_x = x - self._plot_area.pos[0]
        adj_y = y - self._plot_area.pos[1]
        return (
            0 <= adj_x <= self._plot_area.size[0]
            and 0 <= adj_y <= self._plot_area.size[1]
        )

    def to_data(self, x: float, y: float) -> List[float]:
        """Convert widget coords to data coords.

        Use `x, y = self.to_widget(x, y, relative=True)` to first convert into
        widget coordinates if it's in window coordinates because it's assumed
        to be given in local widget coordinates, relative to the graph's pos.

        :Parameters:
            `x, y`: The coordinates to convert.

        If the graph has multiple axes, use :class:`Plot.unproject` instead.
        """
        adj_x = float(x - self._plot_area.pos[0])
        adj_y = float(y - self._plot_area.pos[1])
        norm_x = adj_x / self._plot_area.size[0]
        norm_y = adj_y / self._plot_area.size[1]

        # Convert normalized coordinates to data coordinates
        if self.xlog:
            xmin, xmax = log10(self.xmin), log10(self.xmax)
            conv_x = 10.0 ** (norm_x * (xmax - xmin) + xmin)
        else:
            conv_x = norm_x * (self.xmax - self.xmin) + self.xmin

        if self.ylog:
            ymin, ymax = log10(self.ymin), log10(self.ymax)
            conv_y = 10.0 ** (norm_y * (ymax - ymin) + ymin)
        else:
            conv_y = norm_y * (self.ymax - self.ymin) + self.ymin

        return [conv_x, conv_y]

    # Axis range properties
    xmin = NumericProperty(0.0)
    """The x-axis minimum value.

    If :data:`xlog` is True, xmin must be larger than zero.
    If :data:`xmax` < :data:`xmin`, the x axis will be displayed reversed.

    :data:`xmin` is a :class:`~kivy.properties.NumericProperty`, defaults to 0.
    """

    xmax = NumericProperty(100.0)
    """The x-axis maximum value.

    If :data:`xmax` < :data:`xmin`, the x axis will be displayed reversed.

    :data:`xmax` is a :class:`~kivy.properties.NumericProperty`, defaults to 0.
    """

    xlog = BooleanProperty(False)
    """Determines whether the x-axis should be displayed logarithmically (True)
    or linearly (False).

    :data:`xlog` is a :class:`~kivy.properties.BooleanProperty`, defaults
    to False.
    """

    x_ticks_major = ObjectProperty(0)
    """Positioning of major tick marks on the x-axis.

    May either be a numerical value indicating the distance between ticks
    or a list or tuple giving the absolute tick positions.

    If :data:`x_ticks_major` is a numeric value greater than zero, it
    determines the distance between the major tick marks. Major tick marks
    start from min and re-occur at every ticks_major until :data:`xmax`.
    If :data:`xmax` doesn't overlap with a integer multiple of ticks_major,
    no tick will occur at :data:`xmax`. Values below or equal to zero
    indicate no tick marks.

    If :data:`xlog` is true, then this indicates the distance between ticks
    in multiples of current decade. E.g. if :data:`xmin` is 0.1 and
    ticks_major is 0.1, it means there will be a tick at every 10th of the
    decade, i.e. 0.1 ... 0.9, 1, 2... If it is 0.3, the ticks will occur at
    0.1, 0.3, 0.6, 0.9, 2, 5, 8, 10. You'll notice that it went from 8 to 10
    instead of to 20, that's so that we can say 0.5 and have ticks at every
    half decade, e.g. 0.1, 0.5, 1, 5, 10, 50... Similarly, if ticks_major is
    1.5, there will be ticks at 0.1, 5, 100, 5,000... Also notice, that there's
    always a major tick at the start. Finally, if e.g. :data:`xmin` is 0.6
    and this 0.5 there will be ticks at 0.6, 1, 5...

    If :data:`x_ticks_major` is a list or tuple, a major tick will be displayed
    at each of the given values, if it is in the range of :data:`xmin` to :data:`xmax`.

    :data:`x_ticks_major` is a
    :class:`~kivy.properties.ObjectProperty`, defaults to 0.
    """

    x_ticks_minor = ObjectProperty(0)
    """Positioning of minor tick marks on the x-axis.

    May either be a numerical value indicating the number of subintervals
    between major ticks or a list or tuple giving the absolute tick positions.
    If either one of :data:`x_ticks_minor` and :data:`x_ticks_major` is a
    numerical value and the other one is not, :data:`x_ticks_minor` is ignored.

    If :data:`x_ticks_major` is a numerical value greater than zero, it
    determines the number of sub-intervals into which ticks_major is divided.
    The actual number of minor ticks between the major ticks is
    ticks_minor - 1. Only used if ticks_major is non-zero. If there's no major
    tick at xmax then the number of minor ticks after the last major
    tick will be however many ticks fit until xmax.

    If self.xlog is true, then this indicates the number of intervals the
    distance between major ticks is divided. The result is the number of
    multiples of decades between ticks. I.e. if ticks_minor is 10, then if
    ticks_major is 1, there will be ticks at 0.1, 0.2...0.9, 1, 2, 3... If
    ticks_major is 0.3, ticks will occur at 0.1, 0.12, 0.15, 0.18... Finally,
    as is common, if ticks major is 1, and ticks minor is 5, there will be
    ticks at 0.1, 0.2, 0.4... 0.8, 1, 2...

    If :data:`x_ticks_minor` is a list or tuple, a minor tick will be displayed
    at each of the given values, if it is in the range of :data:`xmin` to :data:`xmax`.

    :data:`x_ticks_minor` is a
    :class:`~kivy.properties.ObjectProperty`, defaults to 0.
    """

    x_grid = BooleanProperty(False)
    """Determines whether the x-axis has tick marks or a full grid.

    If :data:`x_ticks_major` is non-zero, then if x_grid is False tick marks
    will be displayed at every major tick. If x_grid is True, instead of ticks,
    a vertical line will be displayed at every major tick.

    :data:`x_grid` is a :class:`~kivy.properties.BooleanProperty`, defaults
    to False.
    """

    x_grid_label: Union[
        bool, Callable[[float], str], List[str], Tuple[str, ...], str
    ] = ObjectProperty(False)
    """Whether and how labels should be displayed beneath each major tick.

    If false, no tick labels are shown.
    If true, each major tick will have a label containing the axis value displayed at precision :data:`precision`.
    It can also be a callable returning the string representation for a given float value.
    It can also be a list or tuple of strings or a string itself. In that case the k-th item will be displayed at the
    k-th major label.

    :data:`x_grid_label` is a :class:`~kivy.properties.ObjectProperty`,
    defaults to False.
    """

    xlabel = StringProperty("")
    """The label for the x-axis. If not empty it is displayed in the center of
    the axis.

    :data:`xlabel` is a :class:`~kivy.properties.StringProperty`,
    defaults to ''.
    """

    ymin = NumericProperty(0.0)
    """The y-axis minimum value.

    If :data:`ylog` is True, ymin must be larger than zero.
    If :data:`ymax` < :data:`ymin`, the y axis will be displayed reversed.

    :data:`ymin` is a :class:`~kivy.properties.NumericProperty`, defaults to 0.
    """

    ymax = NumericProperty(100.0)
    """The y-axis maximum value.

    If :data:`ymax` < :data:`ymin`, the y axis will be displayed reversed.

    :data:`ymax` is a :class:`~kivy.properties.NumericProperty`, defaults to 0.
    """

    ylog = BooleanProperty(False)
    """Determines whether the y-axis should be displayed logarithmically (True)
    or linearly (False).

    :data:`ylog` is a :class:`~kivy.properties.BooleanProperty`, defaults
    to False.
    """

    y_ticks_major = ObjectProperty(0)
    """Positioning of major tick marks on the y-axis.
    See :data:`x_ticks_major`.

    :data:`y_ticks_major` is a
    :class:`~kivy.properties.ObjectProperty`, defaults to 0.
    """

    y_ticks_minor = ObjectProperty(0)
    """Positioning of minor tick marks on the y-axis.
    See :data:`x_ticks_minor`.

    :data:`y_ticks_minor` is a
    :class:`~kivy.properties.ObjectProperty`, defaults to 0.
    """

    y_grid = BooleanProperty(False)
    """Determines whether the y-axis has tick marks or a full grid. See
    :data:`x_grid`.

    :data:`y_grid` is a :class:`~kivy.properties.BooleanProperty`, defaults
    to False.
    """

    y_grid_label: Union[
        bool, Callable[[float], str], List[str], Tuple[str, ...], str
    ] = ObjectProperty(False)
    """Whether and how labels should be displayed beneath each major tick.

    If false, no tick labels are shown.
    If true, each major tick will have a label containing the axis value displayed at precision :data:`precision`.
    It can also be a callable returning the string representation for a given float value.
    It can also be a list or tuple of strings or a string itself. In that case the k-th item will be displayed at the
    k-th major label.

    :data:`y_grid_label` is a :class:`~kivy.properties.ObjectProperty`,
    defaults to False.
    """

    ylabel = StringProperty("")
    """The label for the y-axis. If not empty it is displayed in the center of
    the axis.

    :data:`ylabel` is a :class:`~kivy.properties.StringProperty`,
    defaults to ''.
    """

    padding = NumericProperty("5dp")
    """Padding distances between the labels, axes titles and graph, as
    well between the widget and the objects near the boundaries.

    :data:`padding` is a :class:`~kivy.properties.NumericProperty`, defaults
    to 5dp.
    """

    x_ticks_angle = NumericProperty(0)
    """Rotate angle of the x-axis tick marks.

    :data:`x_ticks_angle` is a :class:`~kivy.properties.NumericProperty`,
    defaults to 0.
    """

    precision = StringProperty("%g")
    """Determines the numerical precision of the tick mark labels. This value
    governs how the numbers are converted into string representation. Accepted
    values are those listed in Python's manual in the
    "String Formatting Operations" section.

    :data:`precision` is a :class:`~kivy.properties.StringProperty`, defaults
    to '%g'.
    """

    draw_border = BooleanProperty(True)
    """Whether a border is drawn around the canvas of the graph where the
    plots are displayed.

    :data:`draw_border` is a :class:`~kivy.properties.BooleanProperty`,
    defaults to True.
    """

    plots = ListProperty([])
    """Holds a list of all the plots in the graph. To add and remove plots
    from the graph use :data:`add_plot` and :data:`add_plot`. Do not add
    directly edit this list.

    :data:`plots` is a :class:`~kivy.properties.ListProperty`,
    defaults to [].
    """

    view_size = ObjectProperty((0, 0))
    """The size of the graph viewing area - the area where the plots are
    displayed, excluding labels etc.
    """

    view_pos = ObjectProperty((0, 0))
    """The pos of the graph viewing area - the area where the plots are
    displayed, excluding labels etc. It is relative to the graph's pos.
    """


class GraphAA(Graph):
    """
    Graph with anti-aliasing support.

    Supports two anti-aliasing modes:
    - fxaa (default): Fast Approximate Anti-Aliasing - a screen-space, edge-only
      AA post-pass applied to the Graph's FBO. It's GLES2-safe (no derivatives),
      tunable via fxaa_threshold, fxaa_strength, extra_blur, and works on Android
      and desktop.
    - ssaa: True 2× supersampling. The graph renders into a 2× FBO and
      then downscales with a 4-tap box filter in approximate linear color
      (toLinear/toSRGB). It's also GLES2-safe. Performance cost is higher but
      quality is better, especially for very thin lines.

    Properties persist and can be changed at runtime.
    """

    FXAA_FS = """
    $HEADER$
    uniform vec2 inv_tex_size;   // 1.0 / texture size
    uniform float fxaa_threshold; // edge sensitivity
    uniform float fxaa_strength;  // smoothing strength

    vec4 fetch4(vec2 uv){ return texture2D(texture0, uv); }
    vec3 fetch(vec2 uv){ return fetch4(uv).rgb; }
    float luma(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }

    void main(void){
        vec2 uv = tex_coord0;
        vec2 px = inv_tex_size;

        vec4 sM4 = fetch4(uv);
        vec3 cM = sM4.rgb;
        float aM = sM4.a;

        float lM = luma(cM);
        float lN = luma(fetch(uv + vec2( 0.0, -px.y)));
        float lS = luma(fetch(uv + vec2( 0.0,  px.y)));
        float lE = luma(fetch(uv + vec2( px.x,  0.0)));
        float lW = luma(fetch(uv + vec2(-px.x,  0.0)));

        float lMax = max(max(lN, lS), max(lE, lW));
        float lMin = min(min(lN, lS), min(lE, lW));
        float contrast = lMax - lMin;

        if (contrast < fxaa_threshold){
            gl_FragColor = vec4(cM, aM) * frag_color;
            return;
        }

        vec3 cH = (fetch(uv + vec2(px.x, 0.0)) + fetch(uv + vec2(-px.x, 0.0))) * 0.5;
        vec3 cV = (fetch(uv + vec2(0.0, px.y)) + fetch(uv + vec2(0.0, -px.y))) * 0.5;
        vec3 blur = (cH + cV) * 0.5;

        float w = clamp(contrast * (fxaa_strength * 8.0), 0.0, 1.0);
        vec3 outc = mix(cM, blur, w);

        gl_FragColor = vec4(outc, aM) * frag_color;
    }
    """

    # 2x SSAA downsample (box filter in linear color) — GLES2-safe
    SSAA2_FS = """
    $HEADER$
    uniform vec2  inv_tex_size; // of the HIGH-RES (2x) texture
    uniform float ssaa_scale;   // 2.0 when SSAA is enabled

    vec3 toLinear(vec3 c){ return pow(c, vec3(2.2)); }
    vec3 toSRGB(vec3 c){ return pow(c, vec3(1.0/2.2)); }

    void main(void){
        vec2 uv = tex_coord0;
        float step = 0.5 / max(ssaa_scale, 1.0);    // sample half a low-res pixel in hi-res space
        vec2 off = inv_tex_size * step;

        vec4 s00 = texture2D(texture0, uv + vec2(-off.x, -off.y));
        vec4 s10 = texture2D(texture0, uv + vec2( off.x, -off.y));
        vec4 s01 = texture2D(texture0, uv + vec2(-off.x,  off.y));
        vec4 s11 = texture2D(texture0, uv + vec2( off.x,  off.y));

        vec3 lin = 0.25 * (toLinear(s00.rgb) + toLinear(s10.rgb) + toLinear(s01.rgb) + toLinear(s11.rgb));
        float a  = 0.25 * (s00.a + s10.a + s01.a + s11.a);

        gl_FragColor = vec4(toSRGB(lin), a) * frag_color;
    }
    """

    aa_mode = OptionProperty("fxaa", options=("fxaa", "ssaa"))
    """Anti-aliasing mode. Can be either 'fxaa' or 'ssaa'.

    :data:`aa_mode` is an :class:`~kivy.properties.OptionProperty`,
    defaults to 'fxaa'.
    """

    ssaa_scale = NumericProperty(2)
    """SSAA scale factor. Only 1 or 2 are meaningful here; 2 = 2x FBO.

    :data:`ssaa_scale` is an :class:`~kivy.properties.NumericProperty`,
    defaults to 2.
    """

    fxaa_strength = NumericProperty(0.6)
    """FXAA smoothing strength parameter.

    :data:`fxaa_strength` is an :class:`~kivy.properties.NumericProperty`,
    defaults to 0.6.
    """

    fxaa_threshold = NumericProperty(0.07)
    """FXAA edge sensitivity threshold.

    :data:`fxaa_threshold` is an :class:`~kivy.properties.NumericProperty`,
    defaults to 0.07.
    """

    extra_blur = NumericProperty(0.3)
    """FXAA extra blur parameter.

    :data:`extra_blur` is an :class:`~kivy.properties.NumericProperty`,
    defaults to 0.3.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize GraphAA instance.

        :param kwargs: Additional keyword arguments passed to parent class
        """
        # Allow overriding via kwargs
        self.aa_mode = kwargs.pop("aa_mode", self.aa_mode)
        self.ssaa_scale = float(kwargs.pop("ssaa_scale", self.ssaa_scale))
        self._ssaa_resizing = False
        self.fxaa_strength = float(kwargs.pop("fxaa_strength", self.fxaa_strength))
        self.fxaa_threshold = float(kwargs.pop("fxaa_threshold", self.fxaa_threshold))
        self.extra_blur = float(kwargs.pop("extra_blur", self.extra_blur))

        super().__init__(**kwargs)

        # Clear FBO with background
        try:
            self._fbo.clear_color = tuple(self.background_color)
        except Exception:
            pass
        self.bind(background_color=self._update_clear_color)

        # Remove original FBO rectangle
        try:
            self.canvas.remove(self._fbo_rect)
        except Exception:
            pass

        # Build post-pass
        self._build_post_context()

        # Bind uniforms to properties (persist + live updates)
        self.bind(
            fxaa_strength=self._on_fxaa_strength_change,
            fxaa_threshold=self._on_fxaa_threshold_change,
            extra_blur=self._on_extra_blur_change,
            aa_mode=self._on_aa_changed,
            ssaa_scale=self._on_ssaa_scale_change,
        )

        self.canvas.add(self._post_rc)

        self.bind(size=self._on_resize_or_move, pos=self._on_resize_or_move)
        Clock.schedule_once(self._post_init, 0)

    def _on_fxaa_strength_change(self, _: Any, value: float) -> None:
        """
        Handle fxaa_strength property change.

        :param _: Instance reference (unused)
        :param value: New fxaa_strength value
        """
        self._set_uniform("fxaa_strength", float(value))

    def _on_fxaa_threshold_change(self, _: Any, value: float) -> None:
        """
        Handle fxaa_threshold property change.

        :param _: Instance reference (unused)
        :param value: New fxaa_threshold value
        """
        self._set_uniform("fxaa_threshold", float(value))

    def _on_extra_blur_change(self, _: Any, value: float) -> None:
        """
        Handle extra_blur property change.

        :param _: Instance reference (unused)
        :param value: New extra_blur value
        """
        self._set_uniform("extra_blur", float(value))

    def _on_ssaa_scale_change(self, _: Any, __: float) -> None:
        """
        Handle ssaa_scale property change.

        :param _: Instance reference (unused)
        :param __: New ssaa_scale value (unused)
        """
        self._on_aa_changed(None, self.aa_mode)

    def _post_init(self, *args: Any) -> None:
        """Ensure FBO and uniforms are correct after construction."""
        self._apply_ssaa_target()
        self._update_post_rect()

    def _build_post_context(self) -> None:
        """Build the post-processing render context."""
        fs_src = self.SSAA2_FS if self.aa_mode == "ssaa" else self.FXAA_FS
        self._post_rc: RenderContext = RenderContext(
            fs=fs_src, use_parent_modelview=True, use_parent_projection=True
        )
        with self._post_rc:
            Color(1, 1, 1, 1)
            self._post_rect: Rectangle = Rectangle(
                size=self.size, pos=self.pos, texture=self._fbo.texture
            )

        # Push initial uniforms for both modes (harmless if unused)
        self._set_uniform("fxaa_strength", float(self.fxaa_strength))
        self._set_uniform("fxaa_threshold", float(self.fxaa_threshold))
        self._set_uniform("extra_blur", float(self.extra_blur))
        self._set_uniform("ssaa_scale", float(self.ssaa_scale))

    def _on_aa_changed(self, _: Any, __: str) -> None:
        """
        Rebuild post context on mode/scale change.

        :param _: Instance reference (unused)
        :param __: New aa_mode value (unused)
        """
        try:
            if hasattr(self, "_post_rc") and self._post_rc in self.canvas.children:
                self.canvas.remove(self._post_rc)
        except Exception:
            pass
        self._build_post_context()
        self.canvas.add(self._post_rc)
        self._apply_ssaa_target()
        self._update_post_rect()

    def _apply_ssaa_target(self) -> None:
        """Resize the internal FBO when ssaa is active; else adjust filters.
        Guarded to avoid recursion on resize/redraw signals."""
        if self._ssaa_resizing:
            return
        try:
            if self.aa_mode == "ssaa":
                scale = max(1, int(self.ssaa_scale))
                new_w = max(1, int(self.width * scale))
                new_h = max(1, int(self.height * scale))
                if tuple(self._fbo.size) != (new_w, new_h):
                    self._ssaa_resizing = True
                    self._fbo.size = (new_w, new_h)
                    if self._fbo.texture:
                        # avoid hardware minify mixing with our box filter
                        self._fbo.texture.min_filter = "nearest"
                        self._fbo.texture.mag_filter = "nearest"
                    # defer finish to break synchronous recursion
                    Clock.schedule_once(self._finish_ssaa_resize, 0)
            else:
                # FXAA path: default filtering
                if self._fbo.texture:
                    self._fbo.texture.min_filter = "linear"
                    self._fbo.texture.mag_filter = "linear"
        except Exception as e:
            Logger.warning(f"GraphAA: SSAA FBO apply failed: {e}")
            self._ssaa_resizing = False

    def _finish_ssaa_resize(self, *args: Any) -> None:
        """Finish SSAA resize operation."""
        self._ssaa_resizing = False
        # update uniforms/rect with new hi-res texture size
        self._update_post_rect()
        # optional: request a redraw without touching size to avoid loops
        try:
            super()._redraw_all()
        except Exception:
            pass

    def _update_clear_color(self, *args: Any) -> None:
        """Update FBO clear color."""
        try:
            self._fbo.clear_color = tuple(self.background_color)
        except Exception:
            pass

    def _redraw_all(self, *args: Any) -> None:
        """Override to include post-processing update."""
        super()._redraw_all(*args)
        self._update_post_rect()

    def _redraw_size(self, *args: Any) -> None:
        """Override to include post-processing update."""
        super()._redraw_size(*args)
        self._update_post_rect()

    def _on_resize_or_move(self, *args: Any) -> None:
        """Called on widget size/pos changes."""
        self._apply_ssaa_target()
        self._update_post_rect()

    def _update_post_rect(self, *args: Any) -> None:
        """Update post-processing rectangle properties."""
        try:
            tex = self._fbo.texture
            self._post_rect.texture = tex
            self._post_rect.size = self.size
            self._post_rect.pos = self.pos
            tw, th = tex.size if tex else (1, 1)
            iw = 1.0 / float(tw if tw else 1.0)
            ih = 1.0 / float(th if th else 1.0)
            self._post_rc["inv_tex_size"] = (iw, ih)
            # Keep ssaa_scale uniform up to date (used only in SSAA FS)
            self._post_rc["ssaa_scale"] = float(self.ssaa_scale)
        except Exception as e:
            Logger.warning(f"GraphAA: post rect update failed: {e}")

    def _set_uniform(self, name: str, value: float) -> None:
        """
        Set shader uniform value safely.

        :param name: Uniform name
        :param value: Uniform value
        """
        try:
            if hasattr(self, "_post_rc"):
                self._post_rc[name] = value
        except Exception as e:
            Logger.warning(f"GraphAA: failed to set uniform {name}: {e}")

    def set_fxaa(
        self,
        threshold: Optional[float] = None,
        strength: Optional[float] = None,
        extra_blur: Optional[float] = None,
    ) -> None:
        """
        Set FXAA parameters.

        :param threshold: FXAA edge sensitivity threshold
        :param strength: FXAA smoothing strength
        :param extra_blur: FXAA extra blur parameter
        """
        if threshold is not None:
            self.fxaa_threshold = float(threshold)
        if strength is not None:
            self.fxaa_strength = float(strength)
        if extra_blur is not None:
            self.extra_blur = float(extra_blur)


class Plot(EventDispatcher):
    """Optimized Plot class for real-time applications.

    :Events:
        `on_clear_plot`
            Fired before a plot updates the display and lets the fbo know that
            it should clear the old drawings.

    .. versionadded:: 0.4
    """

    __events__ = ("on_clear_plot",)

    # Most recent values of the params used to draw the plot
    params = DictProperty(
        {
            "xlog": False,
            "xmin": 0,
            "xmax": 100,
            "ylog": False,
            "ymin": 0,
            "ymax": 100,
            "size": (0, 0, 0, 0),
        }
    )

    color = ColorProperty([1, 1, 1, 1])
    """Color of the plot."""

    points = ListProperty([])
    """List of (x, y) points to be displayed in the plot.

    The elements of points are 2-tuples, (x, y). The points are displayed
    based on the mode setting.

    :data:`points` is a :class:`~kivy.properties.ListProperty`, defaults to [].
    """

    x_axis = NumericProperty(0)
    """Index of the X axis to use, defaults to 0."""

    y_axis = NumericProperty(0)
    """Index of the Y axis to use, defaults to 0."""

    def __init__(self, **kwargs):
        super(Plot, self).__init__(**kwargs)
        self._dirty = False
        self._transform_cache_valid = False
        self._x_px_func: Optional[Callable] = None
        self._y_px_func: Optional[Callable] = None
        self._params_hash: int = 0

        self.ask_draw = Clock.create_trigger(self._delayed_draw, -1)
        self.bind(params=self._mark_dirty, points=self._mark_dirty)
        self._drawings = self.create_drawings()

    def _mark_dirty(self, *args: Any) -> None:
        """Mark the plot as dirty and schedule a redraw."""
        self._dirty = True
        self.ask_draw()

    def _delayed_draw(self, *largs: Any) -> None:
        """Delayed draw execution to prevent multiple rapid redraws."""
        if self._dirty:
            self.draw()

    def _compute_params_hash(self) -> int:
        """Compute hash of current parameters to detect changes."""
        p = self.params
        return hash(
            (
                p["xlog"],
                p["xmin"],
                p["xmax"],
                p["ylog"],
                p["ymin"],
                p["ymax"],
                tuple(p["size"]),
            )
        )

    def _update_transform_cache(self) -> None:
        """Cache transformation functions for better performance."""
        current_hash = self._compute_params_hash()
        if current_hash == self._params_hash and self._transform_cache_valid:
            return

        self._params_hash = current_hash
        self._transform_cache_valid = True

        # Precompute transformation functions
        if self.params["xlog"]:

            def funcx(x: float) -> float:
                return log10(x) if x > 0 else 0

        else:
            funcx = lambda x: x

        if self.params["ylog"]:

            def funcy(y: float) -> float:
                return log10(y) if y > 0 else 0

        else:
            funcy = lambda y: y

        params = self.params
        size = params["size"]

        # Precalculate constants
        try:
            xmin_trans = funcx(params["xmin"])
            xmax_trans = funcx(params["xmax"])
            xrange = float(xmax_trans - xmin_trans)
            self._ratiox = (size[2] - size[0]) / xrange if xrange != 0 else 0
            self._xmin_base = xmin_trans
            self._x_offset = size[0]

            ymin_trans = funcy(params["ymin"])
            ymax_trans = funcy(params["ymax"])
            yrange = float(ymax_trans - ymin_trans)
            self._ratioy = (size[3] - size[1]) / yrange if yrange != 0 else 0
            self._ymin_base = ymin_trans
            self._y_offset = size[1]

            # Create optimized transform functions
            self._x_px_func = (
                lambda x: (funcx(x) - self._xmin_base) * self._ratiox + self._x_offset
            )
            self._y_px_func = (
                lambda y: (funcy(y) - self._ymin_base) * self._ratioy + self._y_offset
            )
        except Exception:
            self._x_px_func = lambda x: x
            self._y_px_func = lambda y: y

    def funcx(self) -> Callable[[float], float]:
        """Return a function that converts or not the X value according to plot parameters."""
        self._update_transform_cache()
        if self.params["xlog"]:
            return lambda x: log10(x) if x > 0 else 0
        return lambda x: x

    def funcy(self) -> Callable[[float], float]:
        """Return a function that converts or not the Y value according to plot parameters."""
        self._update_transform_cache()
        if self.params["ylog"]:
            return lambda y: log10(y) if y > 0 else 0
        return lambda y: y

    def x_px(self) -> Callable[[float], float]:
        """Return a function that converts the X value of the graph to the
        pixel coordinate on the plot, according to the plot settings and axis
        settings. It's relative to the graph pos.
        """
        self._update_transform_cache()
        return self._x_px_func if self._x_px_func is not None else lambda x: x

    def y_px(self) -> Callable[[float], float]:
        """Return a function that converts the Y value of the graph to the
        pixel coordinate on the plot, according to the plot settings and axis
        settings. The returned value is relative to the graph pos.
        """
        self._update_transform_cache()
        return self._y_px_func if self._y_px_func is not None else lambda y: y

    def unproject(self, x: float, y: float) -> Tuple[float, float]:
        """Return a function that unprojects a pixel to a X/Y value on the plot
        (works only for linear, not log yet). `x`, `y`, is relative to the
        graph pos, so the graph's pos needs to be subtracted from x, y before
        passing it in.
        """
        params = self.params
        size = params["size"]

        xmin, xmax = params["xmin"], params["xmax"]
        xrange = float(xmax - xmin)
        ratiox = (size[2] - size[0]) / xrange if xrange else 0

        ymin, ymax = params["ymin"], params["ymax"]
        yrange = float(ymax - ymin)
        ratioy = (size[3] - size[1]) / yrange if yrange else 0

        x0 = (x - size[0]) / ratiox + xmin if ratiox else xmin
        y0 = (y - size[1]) / ratioy + ymin if ratioy else ymin
        return x0, y0

    def get_px_bounds(self) -> Dict[str, float]:
        """Returns a dict containing the pixels bounds from the plot parameters.
        The returned values are relative to the graph pos.
        """
        params = self.params
        x_px = self.x_px()
        y_px = self.y_px()
        return {
            "xmin": x_px(params["xmin"]),
            "xmax": x_px(params["xmax"]),
            "ymin": y_px(params["ymin"]),
            "ymax": y_px(params["ymax"]),
        }

    def update(
        self,
        xlog: bool,
        xmin: float,
        xmax: float,
        ylog: bool,
        ymin: float,
        ymax: float,
        size: Tuple[float, float, float, float],
    ) -> None:
        """Called by graph whenever any of the parameters change. The plot should be recalculated then.
        log, min, max indicate the axis settings.
        size a 4-tuple describing the bounding box in which we can draw
        graphs, it's (x0, y0, x1, y1), which correspond with the bottom left
        and top right corner locations, respectively.
        """
        self.params.update(
            {
                "xlog": xlog,
                "xmin": xmin,
                "xmax": xmax,
                "ylog": ylog,
                "ymin": ymin,
                "ymax": ymax,
                "size": size,
            }
        )

    def get_group(self) -> str:
        """Returns a string which is unique and is the group name given to all
        the instructions returned by _get_drawings. Graph uses this to remove
        these instructions when needed.
        """
        return ""

    def get_drawings(self) -> List[Any]:
        """Returns a list of canvas instructions that will be added to the
        graph's canvas.
        """
        if isinstance(self._drawings, (tuple, list)):
            return self._drawings
        return []

    def create_drawings(self) -> List[Any]:
        """Called once to create all the canvas instructions needed for the plot."""
        self._color_instruction = Color(*self.color)
        self._mesh_instruction = Mesh(mode="line_strip")
        return [self._color_instruction, self._mesh_instruction]

    def create_legend_drawings(self) -> List[Any]:
        """Called when a legend is added containing this plot. Return drawing instructions."""
        return []

    def draw(self, *largs: Any) -> None:
        """Draw the plot according to the params. It dispatches on_clear_plot
        so derived classes should call super before updating.
        """
        self.dispatch("on_clear_plot")
        self._dirty = False

        if not self.points:
            if hasattr(self, "_mesh_instruction"):
                self._mesh_instruction.vertices = []
            return

        # Update color if needed
        if hasattr(self, "_color_instruction"):
            self._color_instruction.rgba = self.color

        # Transform points efficiently
        x_transform = self.x_px()
        y_transform = self.y_px()

        vertices = []
        append = vertices.extend  # Faster than repeated calls to extend

        for x, y in self.points:
            try:
                px = x_transform(x)
                py = y_transform(y)
                append([px, py, 0, 0])  # [x, y, u, v] - u,v not used
            except (ValueError, ZeroDivisionError):
                continue

        # Set vertices once
        if hasattr(self, "_mesh_instruction"):
            self._mesh_instruction.vertices = vertices
            self._mesh_instruction.indices = list(range(len(vertices) // 4))

    def draw_legend(
        self, center: Tuple[float, float], maximum_size: Tuple[float, float]
    ) -> None:
        """Draw the legend representation."""
        pass

    def iterate_points(self) -> Iterator[Tuple[float, float]]:
        """Iterate on all the points adjusted to the graph settings."""
        x_px = self.x_px()
        y_px = self.y_px()
        for x, y in self.points:
            yield x_px(x), y_px(y)

    def on_clear_plot(self, *largs: Any) -> None:
        """Event handler for plot clearing."""
        pass


class MeshLinePlot(Plot):
    """MeshLinePlot class which displays a set of points similar to a mesh."""

    def _set_mode(self, value):
        """Set the drawing mode for the mesh."""
        if hasattr(self, "_mesh"):
            self._mesh.mode = value
        if hasattr(self, "_mesh_legend"):
            self._mesh_legend.mode = value
            self.draw_legend()

    mode = AliasProperty(lambda self: self._mesh.mode, _set_mode)
    """VBO Mode used for drawing the points. Can be one of: 'points',
    'line_strip', 'line_loop', 'lines', 'triangle_strip', 'triangle_fan'.
    See :class:`~kivy.graphics.Mesh` for more details.

    Defaults to 'line_strip'.
    """

    def create_drawings(self):
        """Create the drawing instructions for the mesh line plot."""
        self._color = Color(*self.color)
        self._mesh = Mesh(mode="line_strip")
        self.bind(color=lambda instr, value: setattr(self._color, "rgba", value))
        return [self._color, self._mesh]

    def create_legend_drawings(self):
        """Create the drawing instructions for the legend."""
        self._mesh_legend = Mesh(mode=self.mode)
        return [self._color, self._mesh_legend]

    def draw_legend(self, center=None, maximum_size=None):
        """Draw the legend marker for this plot."""
        self._legend_center = center = center or getattr(self, "_legend_center", (0, 0))
        self._legend_maximum_size = maximum_size = maximum_size or getattr(
            self, "_legend_maximum_size", (20, 12)
        )

        x, right = center[0] - 0.5 * maximum_size[0], center[0] + 0.5 * maximum_size[0]
        y, top = center[1] - 0.5 * maximum_size[1], center[1] + 0.5 * maximum_size[1]

        # Set vertices based on mode
        if self.mode == "line_strip":
            vertices = [x, center[1], 0, 0, right, center[1], 0, 0]
        elif self.mode == "points":
            vertices = [
                x,
                y,
                0,
                0,
                x,
                top,
                0,
                0,
                right,
                top,
                0,
                0,
                right,
                y,
                0,
                0,
                center[0],
                center[1],
                0,
                0,
            ]
        elif self.mode == "lines":
            vertices = [
                x,
                y,
                0,
                0,
                center[0],
                top,
                0,
                0,
                center[0],
                y,
                0,
                0,
                right,
                top,
                0,
                0,
            ]
        else:
            vertices = [x, y, 0, 0, center[0], top, 0, 0, right, y, 0, 0]

        self._mesh_legend.vertices = vertices
        self._mesh_legend.indices = tuple(range(len(vertices) // 4))

    def draw(self, *args):
        """Draw the mesh line plot."""
        super(MeshLinePlot, self).draw(*args)
        self.plot_mesh()

    def plot_mesh(self):
        """Update the mesh with current point data."""
        points = list(self.iterate_points())
        mesh, vert, _ = self.set_mesh_size(len(points))

        for k, (x, y) in enumerate(points):
            vert[k * 4] = x
            vert[k * 4 + 1] = y
        mesh.vertices = vert

    def set_mesh_size(self, size):
        """Resize the mesh to accommodate the specified number of points."""
        mesh = self._mesh
        vert = mesh.vertices
        ind = mesh.indices
        diff = size - len(vert) // 4

        if diff < 0:
            del vert[4 * size :]
            del ind[size:]
        elif diff > 0:
            ind.extend(range(len(ind), len(ind) + diff))
            vert.extend([0] * (diff * 4))

        mesh.vertices = vert
        return mesh, vert, ind


class MeshStemPlot(MeshLinePlot):
    """MeshStemPlot uses the MeshLinePlot class to draw a stem plot. The data
    provided is graphed from origin to the data point.
    """

    def plot_mesh(self):
        """Update the mesh with stem plot data (lines from origin to points)."""
        points = list(self.iterate_points())
        mesh, vert, _ = self.set_mesh_size(len(points) * 2)
        y0 = self.y_px()(0)

        for k, (x, y) in enumerate(points):
            vert[k * 8] = x
            vert[k * 8 + 1] = y0
            vert[k * 8 + 4] = x
            vert[k * 8 + 5] = y
        mesh.vertices = vert


class LinePlot(Plot):
    """LinePlot draws using a standard Line object."""

    line_width = NumericProperty(1)
    """Width of the line."""

    def create_drawings(self):
        """Create the drawing instructions for the line plot."""
        from kivy.graphics import Line, RenderContext

        self._grc = RenderContext(use_parent_modelview=True, use_parent_projection=True)
        with self._grc:
            self._gcolor = Color(*self.color)
            self._gline = Line(
                points=[], cap="none", width=self.line_width, joint="round"
            )

        return [self._grc]

    def draw(self, *args):
        """Draw the line plot."""
        super(LinePlot, self).draw(*args)
        # Flatten the point list
        points = []
        for x, y in self.iterate_points():
            points.extend([x, y])
        self._gline.points = points

    def create_legend_drawings(self):
        """Create the drawing instructions for the legend."""
        from kivy.graphics import Line, RenderContext

        self._grc_legend = RenderContext(
            use_parent_modelview=True, use_parent_projection=True
        )
        self._grc_legend.add(self._gcolor)
        with self._grc_legend:
            self._gline_legend = Line(points=[], cap="none", joint="round")
        return [self._grc_legend]

    def draw_legend(self, center, maximum_size):
        """Draw the legend marker for this plot."""
        self._maximum_legend_line_width = maximum_size[1] / 2
        self._gline_legend.points = [
            center[0] - 0.5 * maximum_size[0],
            center[1],
            center[0] + 0.5 * maximum_size[0],
            center[1],
        ]
        self._gline_legend.width = min(self.line_width, self._maximum_legend_line_width)

    def on_line_width(self, *largs):
        """Handle line width changes."""
        if hasattr(self, "_gline"):
            self._gline.width = self.line_width
        if hasattr(self, "_gline_legend"):
            self._gline_legend.width = min(
                self.line_width, self._maximum_legend_line_width
            )


class SmoothLinePlot(Plot):
    """
    Smooth line plot with adjustable line width and shader-based anti-aliasing.

    - line_width: controls both the Line instruction thickness and the shader AA envelope.
    - Shader uniform 'edge_scale' adapts AA softness to width.
    - Uses a 64x1 ramp texture (0 -> 255 -> 0) for coverage smoothing.
    """

    # Public property: set via kwargs or runtime (plot.line_width = 3.0)
    line_width: float = NumericProperty(2.0)

    # Fragment shader: uses 'edge_scale' to tune the soft edge width
    SMOOTH_FS: str = """
    $HEADER$
    uniform float edge_scale;  // scales AA soft edge with line width

    void main(void) {
        // 0.015625 == 1/64; *64 makes base = 1.0, then scaled by edge_scale
        float edgewidth = edge_scale * 0.015625 * 64.0;
        float t = texture2D(texture0, tex_coord0).r;
        float e = smoothstep(0.0, edgewidth, t);
        gl_FragColor = frag_color * vec4(1.0, 1.0, 1.0, e);
    }
    """

    # 64x1 RGB gradient data (0 -> 255 -> 0), same as your original
    GRADIENT_DATA: bytes = (
        b"\x00\x00\x00\x07\x07\x07\x0f\x0f\x0f\x17\x17\x17\x1f\x1f\x1f"
        b"'''///777???GGGOOOWWW___gggooowww\x7f\x7f\x7f\x87\x87\x87"
        b"\x8f\x8f\x8f\x97\x97\x97\x9f\x9f\x9f\xa7\xa7\xa7\xaf\xaf\xaf"
        b"\xb7\xb7\xb7\xbf\xbf\xbf\xc7\xc7\xc7\xcf\xcf\xcf\xd7\xd7\xd7"
        b"\xdf\xdf\xdf\xe7\xe7\xe7\xef\xef\xef\xf7\xf7\xf7\xff\xff\xff"
        b"\xf6\xf6\xf6\xee\xee\xee\xe6\xe6\xe6\xde\xde\xde\xd5\xd5\xd5"
        b"\xcd\xcd\xcd\xc5\xc5\xc5\xbd\xbd\xbd\xb4\xb4\xb4\xac\xac\xac"
        b"\xa4\xa4\xa4\x9c\x9c\x9c\x94\x94\x94\x8b\x8b\x8b\x83\x83\x83"
        b"{{{sssjjjbbbZZZRRRJJJAAA999111)))   \x18\x18\x18\x10\x10\x10"
        b"\x08\x08\x08\x00\x00\x00"
    )

    # Shared texture cache (class-level)
    _texture: Optional[Texture] = None

    # Instance-level render objects
    _grc: Optional[RenderContext] = None
    _gcolor: Optional[Color] = None
    _gline: Optional[Line] = None

    def __init__(self, **kwargs):
        # Initialize instance fields before super() to avoid property callbacks
        self._grc = None
        self._gcolor = None
        self._gline = None

        super().__init__(**kwargs)

        # Build drawing instructions
        self._drawings = self.create_drawings()

    @staticmethod
    def _smooth_reload_observer(texture: Texture) -> None:
        """Reload texture data when GL context is lost/restored."""
        texture.blit_buffer(SmoothLinePlot.GRADIENT_DATA, colorfmt="rgb")

    @classmethod
    def _ensure_texture(cls) -> Texture:
        """Create (or reuse) the 64x1 AA ramp texture."""
        if cls._texture is None:
            try:
                tex = Texture.create(size=(1, 64), colorfmt="rgb")
                tex.add_reload_observer(SmoothLinePlot._smooth_reload_observer)
                # Smoother sampling and avoid edge bleeding
                tex.wrap = "clamp_to_edge"
                tex.min_filter = "linear"
                tex.mag_filter = "linear"
                SmoothLinePlot._smooth_reload_observer(tex)
                cls._texture = tex
            except Exception as e:
                Logger.warning(f"SmoothLinePlot: Texture creation failed: {e}")
                # Fallback: create an empty texture to avoid None
                cls._texture = Texture.create(size=(1, 1), colorfmt="rgb")
        return cls._texture

    def _edge_scale_from_width(self, lw: float) -> float:
        """Map line width to shader edge_scale; clamp to a sensible range."""
        return max(0.5, min(3.0, 0.6 * float(lw)))

    def create_drawings(self) -> List:
        """Create the drawing instructions for the smooth line plot."""
        try:
            # Ensure the ramp texture exists
            tex = self._ensure_texture()

            # Build the shader context
            self._grc = RenderContext(
                fs=SmoothLinePlot.SMOOTH_FS,
                use_parent_modelview=True,
                use_parent_projection=True,
            )
            # Push initial edge_scale based on current width
            self._grc["edge_scale"] = self._edge_scale_from_width(self.line_width)

            # Create the line instruction
            with self._grc:
                self._gcolor = Color(*self.color)
                self._gline = Line(
                    points=[],
                    cap="none",  # use "round" if you prefer rounded joins
                    width=float(self.line_width),
                    texture=tex,
                )

            # Bind changes to keep visuals and shader in sync
            self.bind(line_width=self.on_line_width)
            self.bind(color=self.on_color)

            return [self._grc] if self._grc else []
        except Exception as e:
            Logger.error(f"SmoothLinePlot: Failed to create drawings: {e}")
            return []

    def on_line_width(self, _instance, value: float) -> None:
        """Update both Line width and shader AA envelope when line_width changes."""
        try:
            if self._gline is not None:
                self._gline.width = float(value)
            if self._grc is not None:
                self._grc["edge_scale"] = self._edge_scale_from_width(float(value))
        except Exception as e:
            Logger.warning(f"SmoothLinePlot: on_line_width failed: {e}")

    def on_color(self, _instance, value) -> None:
        """Keep line color updated."""
        try:
            if self._gcolor is not None:
                self._gcolor.rgba = value
        except Exception as e:
            Logger.warning(f"SmoothLinePlot: on_color failed: {e}")

    def draw(self, *args) -> None:
        """Draw the smooth line plot."""
        super(SmoothLinePlot, self).draw(*args)
        if not self._gline:
            return
        # Flatten points
        flat: List[float] = []
        for x, y in self.iterate_points():
            flat.extend([x, y])
        self._gline.points = flat

    def create_legend_drawings(self) -> List:
        """Create the drawing instructions for the legend."""
        try:
            return LinePlot.create_legend_drawings(self)
        except Exception as e:
            Logger.warning(f"SmoothLinePlot: Legend creation failed: {e}")
            return []

    def draw_legend(self, center, maximum_size):
        """Draw the legend marker for this plot."""
        try:
            if self._gline:
                self.line_width = self._gline.width
            return LinePlot.draw_legend(self, center, maximum_size)
        except Exception as e:
            Logger.warning(f"SmoothLinePlot: Legend drawing failed: {e}")


class OptimizedSmoothLinePlot(Plot):
    """
    Efficient real-time line plot with antialiasing support and robust shader fallback.

    Features:
    - Tries a derivatives-based AA shader first; falls back to a simpler shader on error.
    - Shared 1D texture ramp for smooth alpha edges (when AA enabled).
    - Minimal overdraw and periodic cleanup to reduce fragmentation.
    """

    max_points: int = NumericProperty(2000)
    """Maximum number of (x, y) points to keep in the ring buffer."""

    cleanup_interval: float = NumericProperty(30.0)
    """Interval in seconds between periodic buffer cleanups."""

    auto_cleanup: bool = BooleanProperty(True)
    """Whether to run automatic periodic cleanup."""

    decimate: bool = BooleanProperty(False)
    """Whether to decimate points (disabled by default to prevent visual shifts)."""

    preserve_extrema: bool = BooleanProperty(True)
    """Whether to preserve extrema when decimating."""

    line_width: float = NumericProperty(10.0)
    """Line width in pixels."""

    enable_antialiasing: bool = BooleanProperty(True)
    """Toggle antialiasing (uses texture ramp)."""

    # Shared texture resources
    _texture_cache: Optional[Texture] = None
    _texture_refs: int = 0

    # Fragment shaders
    AA_FS_DERIVATIVES: str = """
    $HEADER$
    #ifdef GL_ES
    precision highp float;
    #extension GL_OES_standard_derivatives : enable
    #endif

    uniform float edge_scale;

    void main(void) {
        float t = texture2D(texture0, tex_coord0).r;
    #ifdef GL_OES_standard_derivatives
        float w = fwidth(t) * edge_scale;
        float a = smoothstep(0.5 - w, 0.5 + w, t);
    #else
        float a = smoothstep(0.0, max(0.1, edge_scale), t);
    #endif
        gl_FragColor = frag_color * vec4(1.0, 1.0, 1.0, a);
    }
    """

    AA_FS_NO_DERIVATIVES: str = """
    $HEADER$
    #ifdef GL_ES
    precision highp float;
    #endif

    uniform float edge_scale;

    void main(void) {
        float t = texture2D(texture0, tex_coord0).r;
        float scale = max(0.1, edge_scale);
        float a = smoothstep(0.0, scale, t);
        gl_FragColor = frag_color * vec4(1.0, 1.0, 1.0, a);
    }
    """

    @classmethod
    def _get_shared_texture(cls) -> Optional[Texture]:
        """Create or retrieve a shared 1x128 RGB ramp texture for antialiasing."""
        if cls._texture_cache is None:
            try:
                size: int = 128
                tex: Texture = Texture.create(size=(1, size), colorfmt="rgb")

                # Generate symmetric ramp: 0..255..0
                half: int = size // 2
                up: List[int] = [round(255.0 * i / half) for i in range(half + 1)]
                ramp: List[int] = up + up[-2::-1]

                # Convert to RGB bytes
                rgb: List[int] = [v for v in ramp[:size] for _ in range(3)]
                tex.blit_buffer(bytes(rgb), colorfmt="rgb")

                # Configure texture parameters
                tex.wrap = "clamp_to_edge"
                tex.min_filter = "linear"
                tex.mag_filter = "linear"

                cls._texture_cache = tex
                Logger.debug("Created shared AA ramp texture (1x128)")
            except Exception as e:
                Logger.warning(f"AA texture creation failed: {e}")
                cls._texture_cache = None

        if cls._texture_cache:
            cls._texture_refs += 1
        return cls._texture_cache

    @classmethod
    def _release_shared_texture(cls) -> None:
        """Release one reference to the shared AA texture (no deletion, only refcount)."""
        cls._texture_refs = max(0, cls._texture_refs - 1)
        if cls._texture_refs == 0:
            Logger.debug("All shared AA texture refs released")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize plot state, graphics pipeline, and cleanup scheduling."""
        # Graphics components
        self._grc: Optional[RenderContext] = None
        self._gcolor: Optional[Color] = None
        self._gline: Optional[Line] = None
        self._texture: Optional[Texture] = None

        # Data buffers
        maxlen: int = int(kwargs.get("max_points", self.max_points))
        self._ring: Deque[Tuple[float, float]] = deque(maxlen=maxlen)
        self._flat_points: List[float] = [0.0] * (2 * maxlen)

        # State tracking
        self._last_params_sig: Optional[Tuple[Any, ...]] = None
        self._last_ring_len: int = 0
        self._last_ring_tail: Optional[Tuple[float, float]] = None
        self._is_drawing: bool = False
        self._draw_count: int = 0
        self._last_cleanup_time: float = 0.0
        self._needs_clear: bool = False

        super().__init__(**kwargs)
        self._drawings: List[Any] = self.create_drawings()

        if self.auto_cleanup:
            Clock.schedule_interval(self._periodic_cleanup, self.cleanup_interval)

    def _make_context(self, fs_source: str) -> RenderContext:
        """Create a RenderContext with the given fragment shader source and set uniforms."""
        rc: RenderContext = RenderContext(
            fs=fs_source, use_parent_modelview=True, use_parent_projection=True
        )
        rc["edge_scale"] = self._compute_edge_scale()
        return rc

    def create_drawings(self) -> List[Any]:
        """
        Initialize the graphics pipeline.

        Tries the derivatives shader if supported and AA is enabled; falls back to the
        no-derivatives shader automatically if compilation fails.
        """
        try:
            # Query GL extensions (best-effort)
            try:
                raw = glGetString(GL_EXTENSIONS)
                extensions_str: str = raw.decode("utf-8") if raw else ""
            except Exception:
                extensions_str = ""
            extensions: List[str] = extensions_str.split() if extensions_str else []
            has_derivatives: bool = "GL_OES_standard_derivatives" in extensions
            # prefer_derivatives: bool = has_derivatives and self.enable_antialiasing
            prefer_derivatives: bool = self.enable_antialiasing

            Logger.debug(f"OpenGL Derivative Support: {has_derivatives}")
            Logger.debug(f"OpenGL Details: {extensions}")

            # Try preferred shader, then fallback on error
            fs: str = (
                self.AA_FS_DERIVATIVES
                if prefer_derivatives
                else self.AA_FS_NO_DERIVATIVES
            )
            Logger.debug(f"Trying FS:\n{fs}")

            self._grc = self._make_context(fs)

            # Check shader compilation status more directly
            shader_failed = False
            try:
                # Access the shader object directly
                shader = getattr(self._grc, "shader", None)
                if shader:
                    # Check if there are compilation errors
                    error_log = getattr(shader, "error_log", "")
                    success = getattr(shader, "success", True)

                    if not success or (error_log and error_log.strip()):
                        Logger.warning(f"Shader compilation error: {error_log}")
                        Logger.warning(f"-> Problematic shader source:\n{fs}")
                        shader_failed = True
            except Exception as check_error:
                Logger.warning(f"Error checking shader status: {check_error}")
                shader_failed = True

            # If shader failed and we were trying derivatives, fall back
            if shader_failed and prefer_derivatives:
                Logger.info(
                    "Falling back to non-derivative shader due to compilation failure."
                )
                self._grc = self._make_context(self.AA_FS_NO_DERIVATIVES)

            # Prepare texture only if AA is enabled
            self._texture = (
                self._get_shared_texture() if self.enable_antialiasing else None
            )

            with self._grc:
                self._gcolor = Color(*self.color)
                self._gline = Line(
                    points=[], cap="none", width=self.line_width, texture=self._texture
                )

            # React to color changes
            self.bind(
                color=lambda _, v: (
                    setattr(self._gcolor, "rgba", v) if self._gcolor else None
                )
            )
            return [self._grc] if self._grc else []
        except Exception as e:
            Logger.error(f"Failed to create drawings: {e}")
            return []

    def recreate_drawings(self) -> None:
        """Rebuild the graphics pipeline (e.g., after toggling AA or changing line width)."""
        try:
            if self._grc:
                self._grc.clear()
            self._drawings = self.create_drawings()
            self.force_refresh()
        except Exception as e:
            Logger.warning(f"recreate_drawings failed: {e}")

    def feed_point(self, x: float, y: float) -> None:
        """Append a new (x, y) data point and request a draw."""
        self._ring.append((x, y))
        self.ask_draw()

    def update(
        self,
        xlog: bool,
        xmin: float,
        xmax: float,
        ylog: bool,
        ymin: float,
        ymax: float,
        size: Tuple[float, float, float, float],
    ) -> None:
        """Handle axis/scale/viewport updates from the parent graph."""
        super().update(xlog, xmin, xmax, ylog, ymin, ymax, size)
        self._needs_clear = True

    def iterate_points(self) -> Iterator[Tuple[float, float]]:
        """Yield transformed (x, y) points in pixel coordinates."""
        if self._ring:
            x_px = self.x_px()
            y_px = self.y_px()
            for x, y in self._ring:
                yield x_px(x), y_px(y)
        else:
            yield from super().iterate_points()

    def draw(self, *args: Any) -> None:
        """Render entry point invoked by the graph; ensures single draw at a time."""
        if self._is_drawing:
            return
        self._is_drawing = True
        try:
            if self._needs_clear:
                self.dispatch("on_clear_plot")
                self._needs_clear = False
            self._draw_optimized()
        finally:
            self._is_drawing = False

    def _view_valid(self) -> bool:
        """Return True if the current viewport dimensions are valid (non-trivial)."""
        x0, y0, x1, y1 = self.params.get("size", (0, 0, 0, 0))
        return (x1 - x0) > 1 and (y1 - y0) > 1

    def _draw_optimized(self) -> None:
        """Efficiently update and draw the polyline using the cached flat buffer."""
        if not self._gline or not self._view_valid():
            return

        self._draw_count += 1
        now: float = Clock.get_time()

        ring_len: int = len(self._ring)
        ring_tail: Optional[Tuple[float, float]] = self._ring[-1] if ring_len else None

        # Skip redundant draws
        if ring_len == self._last_ring_len and ring_tail == self._last_ring_tail:
            return

        pts: List[Tuple[float, float]] = list(self.iterate_points())
        n: int = min(len(pts), int(self.max_points))

        if n < 2:
            self._gline.points = []
        else:
            fp: List[float] = self._flat_points
            need: int = 2 * n
            if len(fp) < need:
                fp.extend([0.0] * (need - len(fp)))

            for i in range(n):
                fp[2 * i] = pts[i][0]
                fp[2 * i + 1] = pts[i][1]
            self._gline.points = fp[:need]

        self._last_ring_len = ring_len
        self._last_ring_tail = ring_tail

        if self._should_cleanup(now):
            self._cleanup_old_data()
            self._last_cleanup_time = now

    def _should_cleanup(self, current_time: float) -> bool:
        """Return True if it is time to run periodic buffer cleanup."""
        return self.auto_cleanup and (
            current_time - self._last_cleanup_time > float(self.cleanup_interval)
        )

    def _cleanup_old_data(self) -> None:
        """Reset internal buffers to reduce memory fragmentation."""
        self._flat_points = [0.0] * (2 * int(self.max_points))
        if self._ring.maxlen != int(self.max_points):
            self._ring = deque(self._ring, maxlen=int(self.max_points))
        Logger.debug("Cleanup completed")

    def _periodic_cleanup(self, dt: float) -> None:
        """Scheduler callback that triggers cleanup at the configured interval."""
        try:
            now: float = Clock.get_time()
            if now - self._last_cleanup_time > self.cleanup_interval:
                self._cleanup_old_data()
                self._last_cleanup_time = now
        except Exception as e:
            Logger.warning(f"Periodic cleanup failed: {e}")

    def on_line_width(self, _: Any, value: float) -> None:
        """Kivy property callback: apply new line width and update edge scale."""
        if self._gline:
            self._gline.width = value
        if self._grc and self.enable_antialiasing:
            self._grc["edge_scale"] = self._compute_edge_scale()

    def on_enable_antialiasing(self, _: Any, __: bool) -> None:
        """Kivy property callback: rebuild pipeline when AA is toggled."""
        self.recreate_drawings()

    def on_max_points(self, _: Any, value: int) -> None:
        """Kivy property callback: resize buffers when the maximum point count changes."""
        value_i: int = int(max(2, value))
        self._ring = deque(self._ring, maxlen=value_i)
        self._flat_points = [0.0] * (2 * value_i)

    def on_points(self, _: Any, value: Optional[List[Tuple[float, float]]]) -> None:
        """Kivy property callback: replace the data set with a provided list of points."""
        try:
            if value:
                self._ring = deque(
                    value[-int(self.max_points) :], maxlen=int(self.max_points)
                )
            else:
                self._ring.clear()
        except Exception:
            # Ignore malformed inputs
            pass

    def on_color(self, _: Any, value: Tuple[float, float, float, float]) -> None:
        """Kivy property callback: update the line color."""
        if self._gcolor:
            self._gcolor.rgba = value

    def _compute_edge_scale(self) -> float:
        """Compute the antialiasing edge scale factor from the current line width."""
        lw: float = float(self.line_width)
        return max(0.5, min(3.0, 0.6 * lw))

    def create_legend_drawings(self) -> List[Any]:
        """Create a simplified legend representation using the garden graph helper."""
        try:
            return LinePlot.create_legend_drawings(self)
        except Exception as e:
            Logger.warning(f"Legend creation failed: {e}")
            return []

    def draw_legend(
        self, center: Tuple[float, float], maximum_size: Tuple[float, float]
    ) -> Optional[List[Any]]:
        """Draw the legend icon for this plot."""
        try:
            if self._gline:
                self.line_width = self._gline.width
            return LinePlot.draw_legend(self, center, maximum_size)
        except Exception as e:
            Logger.warning(f"Legend drawing failed: {e}")
            return None

    def force_refresh(self) -> None:
        """Clear cached points and request a full redraw."""
        self._flat_points = [0.0] * (2 * int(self.max_points))
        self.ask_draw()

    def __del__(self) -> None:
        """Attempt to release resources and unschedule callbacks on destruction."""
        try:
            Clock.unschedule(self._periodic_cleanup)
            self._release_shared_texture()
            if self._grc:
                self._grc.clear()
        except Exception:
            pass


class AALineStripPlot(Plot):
    """GPU line AA using triangle strip + analytic alpha.
    - No tex_coords, we store signed distance in the vertex 'v' component.
    - GLES2/3 safe: fragment shader uses tex_coord0.y only (no derivatives/texture sampling).
    """

    # line_width (display-independent pixels, dp)
    # - Visual line thickness. Converted to screen pixels via kivy.metrics.dp.
    # - Larger values make the strip thicker and proportionally expand the AA envelope.
    # - Interacts with feather_px (edge softness width) and profile (edge steepness).
    # - Typical: 1.5–4.0 dp for thin plots, 4–8 dp for thicker strokes.
    line_width = NumericProperty(2.0)

    # feather_px (screen pixels)
    # - Width of the anti-aliasing fade at the outer edge (in px).
    # - Higher feather_px = softer, more gradual edge (can look slightly blurrier).
    # - Lower feather_px = crisper edge (may show more aliasing on diagonals).
    # - Typical: 1.0–1.4 px for thin lines, 1.4–2.0 px for medium/thick lines.
    # - Performance: negligible; affects only a small band near the edge in the fragment shader.
    feather_px = NumericProperty(1.25)

    # profile (unitless)
    # - Shapes the edge alpha curve by raising it to this power (post-smootherstep).
    # - >1.0 makes the transition steeper (sharper edge); <1.0 softens it (smoother edge).
    # - Tune together with feather_px: feather sets width, profile sets steepness.
    # - Recommended start: 1.05 (slightly crisper), adjust 0.95–1.15 as needed.
    profile = NumericProperty(1.05)

    # max_points (count)
    # - Upper bound on the number of polyline points used to build the GPU strip per draw.
    # - If points exceed this, only the last max_points are used (keeps recent data).
    # - Larger values increase CPU work (joins/offsets) and GPU vertices (2 per point).
    # - Practical ranges: 1,000–10,000. If your pipeline uses 16-bit indices, ensure 2*max_points < 65,535
    #   (i.e., max_points < 32,768) to avoid index overflows.
    max_points = NumericProperty(4000)

    # Fragment shader: fade alpha within 'feather' pixels near the strip edge.
    FS = """
    // glsl
    $HEADER$
    #ifdef GL_ES
    precision mediump float;
    #endif

    uniform float half_width;   // px
    uniform float feather;      // px (try 1.4..1.8)
    uniform float profile;      // 0.8..1.3 (1.0 neutral, >1 slightly steeper)

    float edge_alpha(float d){
        // quintic smootherstep for gentler transition near edges
        float t = clamp((half_width - d) / max(1e-6, feather), 0.0, 1.0);
        t = t*t*t*(t*(t*6.0 - 15.0) + 10.0);
        return pow(t, profile);
    }

    void main(void){
        float d = abs(tex_coord0.y);        // signed distance carried in v
        float a = edge_alpha(d);
        vec4 c = frag_color;
        c.a *= a;
        gl_FragColor = c;
    }
    """

    def __init__(self, **kwargs):
        self._rc = None
        self._color = None
        self._mesh = None
        super().__init__(**kwargs)
        self._drawings = self.create_drawings()

        # Redraw on property changes
        self.fbind("line_width", lambda *_: self.ask_draw())
        self.fbind("feather_px", lambda *_: self.ask_draw())
        self.fbind(
            "color",
            lambda *_: (
                setattr(self._color, "rgba", self.color) if self._color else None
            ),
        )

    def create_drawings(self):
        """Create RenderContext + Mesh. No use of tex_coords; we only set vertices/indices."""
        try:
            self._rc = RenderContext(
                fs=self.FS, use_parent_modelview=True, use_parent_projection=True
            )
            with self._rc:
                self._color = Color(*self.color)
                # Mesh vertices are [x, y, u, v] per vertex
                self._mesh = Mesh(mode="triangle_strip", vertices=[], indices=[])
            return [self._rc]
        except Exception as e:
            Logger.error(f"AALineStripPlot: create_drawings failed: {e}")
            return []

    def draw(self, *args):
        """Build a triangle strip with signed distance in v; bevel fallback for sharp joins."""
        super().draw(*args)
        if not self._mesh:
            return

        pts = list(self.iterate_points())
        if len(pts) < 2:
            self._mesh.vertices = []
            self._mesh.indices = []
            return

        maxn = int(self.max_points)
        if len(pts) > maxn:
            pts = pts[-maxn:]

        # Drop consecutive duplicates
        cleaned = []
        for p in pts:
            if not cleaned or (
                abs(p[0] - cleaned[-1][0]) > 1e-6 or abs(p[1] - cleaned[-1][1]) > 1e-6
            ):
                cleaned.append(p)
        pts = cleaned
        n = len(pts)
        if n < 2:
            self._mesh.vertices = []
            self._mesh.indices = []
            return

        hw = float(dp(self.line_width)) * 0.5  # half width in pixels
        miter_limit = 4.0 * hw

        def norm(dx, dy):
            L = (dx * dx + dy * dy) ** 0.5
            return (dx / L, dy / L) if L > 1e-12 else (0.0, 0.0)

        def perp(dx, dy):
            return -dy, dx

        # Segment directions and normals
        dirs = []
        norms = []
        for i in range(n - 1):
            ux, uy = norm(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            nx, ny = perp(ux, uy)
            dirs.append((ux, uy))
            norms.append((nx, ny))

        # Compute offsets with miter where reasonable, bevel fallback when too sharp
        left, right = [], []
        for i in range(n):
            px, py = pts[i]
            if i == 0:
                nx, ny = norms[0]
                oxL, oyL = nx * hw, ny * hw
                oxR, oyR = -oxL, -oyL
            elif i == n - 1:
                nx, ny = norms[-1]
                oxL, oyL = nx * hw, ny * hw
                oxR, oyR = -oxL, -oyL
            else:
                n1 = norms[i - 1]
                n2 = norms[i]
                # miter attempt
                bx, by = n1[0] + n2[0], n1[1] + n2[1]
                bl = (bx * bx + by * by) ** 0.5
                use_bevel = bl <= 1e-6
                if not use_bevel:
                    bx, by = bx / bl, by / bl
                    dot = bx * n1[0] + by * n1[1]
                    scale = hw / max(1e-6, dot)
                    use_bevel = scale > miter_limit

                if not use_bevel:
                    ox, oy = bx * scale, by * scale
                    oxL, oyL = ox, oy
                    oxR, oyR = -ox, -oy
                else:
                    # Determine turn direction (cross product of segment dirs)
                    turn = dirs[i - 1][0] * dirs[i][1] - dirs[i - 1][1] * dirs[i][0]
                    if turn > 0.0:  # left turn
                        oxL, oyL = n2[0] * hw, n2[1] * hw  # outward on left
                        oxR, oyR = -n1[0] * hw, -n1[1] * hw  # inward on right
                    else:  # right turn
                        oxL, oyL = n1[0] * hw, n1[1] * hw
                        oxR, oyR = -n2[0] * hw, -n2[1] * hw

            left.append((px + oxL, py + oyL))
            right.append((px + oxR, py + oyR))

        # Build vertices with v = signed distance (±hw); u is unused (0.0)
        verts = []
        for i in range(n):
            lx, ly = left[i]
            rx, ry = right[i]
            verts.extend([lx, ly, 0.0, +hw])  # left vertex: v = +hw
            verts.extend([rx, ry, 0.0, -hw])  # right vertex: v = -hw

        self._mesh.vertices = verts
        self._mesh.indices = list(range(2 * n))

        # Update shader uniforms (including profile shaping)
        self._rc["half_width"] = hw
        self._rc["feather"] = float(self.feather_px)
        self._rc["profile"] = float(self.profile)


class MeshStripPlot(Plot):
    """
    Thick polyline without gaps using Mesh(mode='triangle_strip'):
    - Computes per-vertex joined offsets (miter/bevel) to keep edges continuous.
    - line_width is in dp (converted to screen pixels).
    - Fast on Android (no fragment texture sampling).
    """

    line_width = NumericProperty(2.0)  # in dp
    max_points = NumericProperty(1500)
    auto_cleanup = BooleanProperty(True)
    cleanup_interval = NumericProperty(30.0)
    miter_limit = NumericProperty(4.0)  # max miter length relative to half width

    def __init__(self, **kwargs):
        self._rc: Optional[RenderContext] = None
        self._color: Optional[Color] = None
        self._mesh: Optional[Mesh] = None
        self._last_cleanup_time: float = 0.0

        super().__init__(**kwargs)
        self._drawings = self.create_drawings()

        # Redraw when width/color changes
        self.fbind("line_width", lambda *_: self.ask_draw())
        self.fbind(
            "color",
            lambda *_: (
                setattr(self._color, "rgba", self.color) if self._color else None
            ),
        )

        if self.auto_cleanup:
            Clock.schedule_interval(self._periodic_cleanup, self.cleanup_interval)

    def create_drawings(self):
        try:
            self._rc = RenderContext(
                use_parent_modelview=True, use_parent_projection=True
            )
            with self._rc:
                self._color = Color(*self.color)
                self._mesh = Mesh(mode="triangle_strip", vertices=[], indices=[])
            return [self._rc]
        except Exception as e:
            Logger.error(f"OptimizedThickMeshPlot: create_drawings failed: {e}")
            return []

    def draw(self, *args):
        super().draw(*args)
        if not self._mesh:
            return

        pts = list(self.iterate_points())
        if len(pts) > int(self.max_points):
            pts = pts[-int(self.max_points) :]
        # Drop consecutive duplicates
        cleaned: List[Tuple[float, float]] = []
        for p in pts:
            if not cleaned or (
                abs(p[0] - cleaned[-1][0]) > 1e-6 or abs(p[1] - cleaned[-1][1]) > 1e-6
            ):
                cleaned.append(p)
        pts = cleaned

        n = len(pts)
        if n < 2:
            self._mesh.vertices = []
            self._mesh.indices = []
            return

        half_w = float(dp(self.line_width)) * 0.5
        miter_limit_px = self.miter_limit * half_w

        def normalize(dx: float, dy: float) -> Tuple[float, float]:
            L = (dx * dx + dy * dy) ** 0.5
            if L <= 1e-12:
                return 0.0, 0.0
            return dx / L, dy / L

        def perp(dx: float, dy: float) -> Tuple[float, float]:
            return -dy, dx

        # Precompute directions and normals for segments
        dirs: List[Tuple[float, float]] = []
        norms: List[Tuple[float, float]] = []
        for i in range(n - 1):
            dx = pts[i + 1][0] - pts[i][0]
            dy = pts[i + 1][1] - pts[i][1]
            ux, uy = normalize(dx, dy)
            nx, ny = perp(ux, uy)
            dirs.append((ux, uy))
            norms.append((nx, ny))

        # Compute joined offset normal at each vertex (miter with clamp to miter_limit)
        left: List[Tuple[float, float]] = []
        right: List[Tuple[float, float]] = []

        for i in range(n):
            px, py = pts[i]
            if i == 0:
                nx, ny = norms[0]
                ox, oy = nx * half_w, ny * half_w
            elif i == n - 1:
                nx, ny = norms[-1]
                ox, oy = nx * half_w, ny * half_w
            else:
                n_prev = norms[i - 1]
                n_next = norms[i]
                # Bisector normal
                bx = n_prev[0] + n_next[0]
                by = n_prev[1] + n_next[1]
                bx, by = normalize(bx, by)
                # If bisector is degenerate (opposite normals), fall back to one side
                if abs(bx) + abs(by) < 1e-12:
                    bx, by = n_prev
                # Scale miter: half_w / dot(bisector, n_prev)
                dot = bx * n_prev[0] + by * n_prev[1]
                scale = half_w / max(1e-6, dot)
                # Clamp miter length to avoid spikes
                mlen = (bx * scale) ** 2 + (by * scale) ** 2
                if mlen > (miter_limit_px**2):
                    scale = miter_limit_px / max(1e-6, (bx**2 + by**2) ** 0.5)
                ox, oy = bx * scale, by * scale

            left.append((px + ox, py + oy))
            right.append((px - ox, py - oy))

        # Build triangle strip vertices: L0, R0, L1, R1, ...
        verts: List[float] = []
        for i in range(n):
            lx, ly = left[i]
            rx, ry = right[i]
            verts.extend([lx, ly, 0.0, 0.0])
            verts.extend([rx, ry, 0.0, 0.0])

        # Indices for triangle_strip: sequential
        indices = list(range(2 * n))

        self._mesh.vertices = verts
        self._mesh.indices = indices

    def _periodic_cleanup(self, dt):
        now = Clock.get_time()
        if now - self._last_cleanup_time > float(self.cleanup_interval):
            if self._mesh and len(self._mesh.vertices) > 2 * self.max_points * 4:
                self._mesh.vertices = self._mesh.vertices[
                    : 2 * int(self.max_points) * 4
                ]
                self._mesh.indices = list(range(2 * int(self.max_points)))
            self._last_cleanup_time = now


class ContourPlot(Plot):
    """ContourPlot visualizes 3 dimensional data as an intensity map image.

    The user must first specify 'xrange' and 'yrange' (tuples of min,max) and
    then 'data', the intensity values.
    `data`, is a MxN matrix, where the first dimension of size M specifies the
    `y` values, and the second dimension of size N specifies the `x` values.
    Axis Y and X values are assumed to be linearly spaced values from
    xrange/yrange and the dimensions of 'data', `MxN`, respectively.
    The color values are automatically scaled to the min and max z range of the
    data set.
    """

    _image = ObjectProperty(None)
    data = ObjectProperty(None, force_dispatch=True)
    xrange = ListProperty([0, 100])
    yrange = ListProperty([0, 100])

    def __init__(self, **kwargs):
        super(ContourPlot, self).__init__(**kwargs)
        self.bind(data=self.ask_draw, xrange=self.ask_draw, yrange=self.ask_draw)

    def create_drawings(self):
        """Create the drawing instructions for the contour plot."""
        self._image = Rectangle()
        self._color = Color([1, 1, 1, 1])
        self.bind(color=lambda instr, value: setattr(self._color, "rgba", value))
        return [self._color, self._image]

    def draw(self, *args):
        """Draw the contour plot."""
        super(ContourPlot, self).draw(*args)
        data = self.data
        xdim, ydim = data.shape

        # Find the minimum and maximum z values
        zmax, zmin = data.max(), data.min()
        rgb_scale_factor = 1.0 / (zmax - zmin) * 255

        # Scale the z values into RGB data
        buf = np.array(data, dtype=float, copy=True)
        np.subtract(buf, zmin, out=buf)
        np.multiply(buf, rgb_scale_factor, out=buf)

        # Convert to RGB byte array
        buf = np.asarray(buf, dtype=np.uint8)
        buf = np.expand_dims(buf, axis=2)
        buf = np.concatenate((buf, buf, buf), axis=2)
        buf = np.reshape(buf, (xdim, ydim, 3))

        charbuf = bytearray(np.reshape(buf, (buf.size)))
        self._texture = Texture.create(size=(xdim, ydim), colorfmt="rgb")
        self._texture.blit_buffer(charbuf, colorfmt="rgb", bufferfmt="ubyte")

        # Update image position and size
        image = self._image
        image.texture = self._texture

        x_px, y_px = self.x_px(), self.y_px()
        bl = x_px(self.xrange[0]), y_px(self.yrange[0])
        tr = x_px(self.xrange[1]), y_px(self.yrange[1])
        image.pos = bl
        image.size = (tr[0] - bl[0], tr[1] - bl[1])


class BarPlot(Plot):
    """BarPlot class which displays a bar graph."""

    bar_width = NumericProperty(1)
    """Width of individual bars."""

    bar_spacing = NumericProperty(1.0)
    """Spacing factor between bars."""

    graph = ObjectProperty(allownone=True)
    """Reference to the parent graph."""

    def __init__(self, *args, **kwargs):
        super(BarPlot, self).__init__(*args, **kwargs)
        self.bind(bar_width=self.ask_draw)
        self.bind(bar_width=lambda *_: self.draw_legend())
        self.bind(points=self.update_bar_width)
        self.bind(graph=self.update_bar_width)

    def update_bar_width(self, *args):
        """Update bar width based on graph dimensions and data points."""
        if not self.graph or len(self.points) < 2 or self.graph.xmax == self.graph.xmin:
            return

        point_width = (
            len(self.points)
            * float(abs(self.graph.xmax) + abs(self.graph.xmin))
            / float(abs(max(self.points)[0]) + abs(min(self.points)[0]))
        )

        if self.points:
            self.bar_width = (
                (self.graph.width - self.graph.padding) / point_width * self.bar_spacing
            )
        else:
            self.bar_width = 1

    def create_drawings(self):
        """Create the drawing instructions for the bar plot."""
        self._color = Color(*self.color)
        self._mesh = Mesh()
        self.bind(color=lambda instr, value: setattr(self._color, "rgba", value))
        return [self._color, self._mesh]

    def create_legend_drawings(self):
        """Create the drawing instructions for the legend."""
        self._rectangle = Rectangle()
        return [self._color, self._rectangle]

    def draw_legend(self, center=None, maximum_size=None):
        """Draw the legend marker for this plot."""
        if not hasattr(self, "_rectangle"):
            return

        self._legend_maximum_size = maximum_size = maximum_size or getattr(
            self, "_legend_maximum_size", (20, 12)
        )
        width = (
            min(self.bar_width, maximum_size[0])
            if self.bar_width >= 0
            else maximum_size[0]
        )
        height = maximum_size[1]

        if center:
            x = center[0] - width / 2
            y = center[1] - height / 2
        else:
            y = self._rectangle.pos[1]
            x = self._rectangle.pos[0] - width / 2 + self._rectangle.size[0] / 2

        self._rectangle.pos = x, y
        self._rectangle.size = width, height

    def draw(self, *args):
        """Draw the bar plot."""
        super(BarPlot, self).draw(*args)
        points = self.points

        # Mesh only supports (2^16) - 1 indices
        if len(points) * 6 > 65535:
            Logger.error(
                "BarPlot: cannot support more than 10922 points. "
                "Ignoring extra points."
            )
            points = points[:10922]

        point_len = len(points)
        mesh = self._mesh
        mesh.mode = "triangles"
        vert = mesh.vertices
        ind = mesh.indices
        diff = len(points) * 6 - len(vert) // 4

        if diff < 0:
            del vert[24 * point_len :]
            del ind[point_len:]
        elif diff > 0:
            ind.extend(range(len(ind), len(ind) + diff))
            vert.extend([0] * (diff * 4))

        bounds = self.get_px_bounds()
        x_px, y_px = self.x_px(), self.y_px()
        ymin = y_px(0)

        bar_width = self.bar_width
        if bar_width < 0:
            bar_width = x_px(bar_width) - bounds["xmin"]

        for k in range(point_len):
            p = points[k]
            x1, x2 = x_px(p[0]), x_px(p[0]) + bar_width
            y1, y2 = ymin, y_px(p[1])

            idx = k * 24
            # First triangle
            vert[idx : idx + 12] = [x1, y2, 0, 0, x1, y1, 0, 0, x2, y1, 0, 0]
            # Second triangle
            vert[idx + 12 : idx + 24] = [x1, y2, 0, 0, x2, y2, 0, 0, x2, y1, 0, 0]

        mesh.vertices = vert

    def _unbind_graph(self, graph):
        """Unbind from graph events."""
        graph.unbind(
            width=self.update_bar_width,
            xmin=self.update_bar_width,
            ymin=self.update_bar_width,
        )

    def bind_to_graph(self, graph):
        """Bind to graph for automatic updates."""
        old_graph = self.graph

        if old_graph:
            self._unbind_graph(old_graph)

        # Bind to the new graph
        self.graph = graph
        graph.bind(
            width=self.update_bar_width,
            xmin=self.update_bar_width,
            ymin=self.update_bar_width,
        )

    def unbind_from_graph(self):
        """Unbind from current graph."""
        if self.graph:
            self._unbind_graph(self.graph)


class HBar(MeshLinePlot):
    """HBar draws horizontal bars on all the Y points provided."""

    def plot_mesh(self, *args):
        """Update the mesh with horizontal bar data."""
        points = self.points
        mesh, vert, ind = self.set_mesh_size(len(points) * 2)
        mesh.mode = "lines"

        bounds = self.get_px_bounds()
        px_xmin, px_xmax = bounds["xmin"], bounds["xmax"]
        y_px = self.y_px()

        for k, y in enumerate(points):
            y = y_px(y)
            vert[k * 8 : k * 8 + 8] = [px_xmin, y, 0, 0, px_xmax, y, 0, 0]
        mesh.vertices = vert


class VBar(MeshLinePlot):
    """VBar draws vertical bars on all the X points provided."""

    def plot_mesh(self, *args):
        """Update the mesh with vertical bar data."""
        points = self.points
        mesh, vert, ind = self.set_mesh_size(len(points) * 2)
        mesh.mode = "lines"

        bounds = self.get_px_bounds()
        px_ymin, px_ymax = bounds["ymin"], bounds["ymax"]
        x_px = self.x_px()

        for k, x in enumerate(points):
            x = x_px(x)
            vert[k * 8 : k * 8 + 8] = [x, px_ymin, 0, 0, x, px_ymax, 0, 0]
        mesh.vertices = vert


class ScatterPlot(Plot):
    """ScatterPlot draws using a standard Point object.
    The pointsize can be controlled with :attr:`point_size`.

    >>> plot = ScatterPlot(color=[1, 0, 0, 1], point_size=5)
    """

    point_size = NumericProperty(1)
    """The point size of the scatter points. Defaults to 1."""

    def create_drawings(self):
        """Create the drawing instructions for the scatter plot."""
        from kivy.graphics import Point, RenderContext

        self._points_context = RenderContext(
            use_parent_modelview=True, use_parent_projection=True
        )
        with self._points_context:
            self._gcolor = Color(*self.color)
            self._gpts = Point(points=[], pointsize=self.point_size)

        return [self._points_context]

    def create_legend_drawings(self):
        """Create the drawing instructions for the legend."""
        from kivy.graphics import Point, RenderContext

        self._points_legend_context = RenderContext(
            use_parent_modelview=True, use_parent_projection=True
        )
        self._points_legend_context.add(self._gcolor)
        with self._points_legend_context:
            self._gpts_legend = Point(points=[])

        return [self._points_legend_context]

    def draw(self, *args):
        """Draw the scatter plot."""
        super(ScatterPlot, self).draw(*args)
        # Flatten the point list
        self._gpts.points = list(chain(*self.iterate_points()))

    def draw_legend(self, center, maximum_size):
        """Draw the legend marker for this plot."""
        self._maximum_legend_point_size = min(maximum_size) / 2
        self._gpts_legend.pointsize = min(
            self._maximum_legend_point_size, self.point_size
        )
        self._gpts_legend.points = center

    def on_point_size(self, *largs):
        """Handle point size changes."""
        if hasattr(self, "_gpts"):
            self._gpts.pointsize = self.point_size
        if hasattr(self, "_maximum_legend_point_size"):
            self._gpts_legend.pointsize = min(
                self.point_size, self._maximum_legend_point_size
            )


class PointPlot(Plot):
    """Displays a set of points."""

    point_size = NumericProperty(1)
    """Point size, defaults to 1."""

    def __init__(self, **kwargs):
        super(PointPlot, self).__init__(**kwargs)

        def update_size(*largs):
            if self._point:
                self._point.pointsize = self.point_size
            if hasattr(self, "_maximum_legend_point_size"):
                self._point_legend.pointsize = min(
                    self.point_size, self._maximum_legend_point_size
                )

        self.fbind("point_size", update_size)

        def update_color(*largs):
            if self._color:
                self._color.rgba = self.color

        self.fbind("color", update_color)

    def create_drawings(self):
        """Create the drawing instructions for the point plot."""
        self._color = Color(*self.color)
        self._point = Point(pointsize=self.point_size)
        return [self._color, self._point]

    def create_legend_drawings(self):
        """Create the drawing instructions for the legend."""
        self._point_legend = Point()
        return [self._color, self._point_legend]

    def draw(self, *args):
        """Draw the point plot."""
        super(PointPlot, self).draw(*args)
        self._point.points = [v for p in self.iterate_points() for v in p]

    def draw_legend(self, center, maximum_size):
        """Draw the legend marker for this plot."""
        self._maximum_legend_point_size = min(maximum_size) / 2
        self._point_legend.pointsize = min(
            self._maximum_legend_point_size, self.point_size
        )
        self._point_legend.points = center


class LineAndMarkerPlot(Plot):
    """A Plot consisting of a line and markers.

    The line is drawn using a :class:`kivy.graphics.SmoothLine` and can
    be adapted using the :data:`line_width` and :data:`color` properties.

    The markers can be adapted using the :data:`marker_shape`,
    :data:`marker_line_width`, :data:`marker_color` properties.
    """

    marker_shape: str = OptionProperty(
        None, allownone=True, options=[None, *"x+*-|<>v^osdOSD"]
    )
    """The shape of the marker.

    The following values are allowed:
        None:   no marker
        'x':    cross
        '+':    plus sign
        '*':    asterisk
        '-':    horizontal line
        '|':    vertical line
        '<':    left-pointing open triangle
        '>':    right-pointing open triangle
        'v':    downward-pointing open triangle
        '^':    upward-pointing open triangle
        'o':    circle
        's':    square
        'd':    diamond
        'O':    filled circle
        'S':    filled square
        'D':    filled diamond

    :data:`marker_shape` is a :class:`kivy.properties.OptionProperty`
    and defaults to None.
    """

    marker_size: float = BoundedNumericProperty(dp(12), min=0)
    """Size of a single marker as the side length of a surrounding square.

    For markers, that are drawn as a line, e. g. 'o' or 's' markers, the actual
    outermost size will be :data:`marker_size` + 2 * :data:`marker_line_width`, because
    half the line will exceed the square defined by marker_size * marker_size.

    :data:`marker_size` is a :class:`kivy.properties.BoundedNumericProperty`
    with a minimum of 0 and defaults to 12 dp.
    """

    marker_line_width: float = BoundedNumericProperty(dp(1.5), min=0)
    """The line width with which the markers are drawn.

    For filled markers this value is ignored.

    :data:`marker_line_width` is a :class:`kivy.properties.BoundedNumericProperty`
    with a minimum of 0 and defaults to 1.5 dp.
    """

    line_width: float = BoundedNumericProperty(dp(1.1), min=0)
    """Line width of the line connecting the markers.

    :data:`line_width` is a :class:`kivy.properties.BoundedNumericProperty`
    with a minimum of 0 and defaults to 1.1 dp.
    """

    marker_color = ColorProperty(None, allownone=True)
    """Color of the markers.

    May be set to None, in which case the markers are drawn with the same
    color as the line, i. e. the color defines by :data:`color`.

    :data:`marker_color` is a :class:`kivy.properties.ColorProperty`
    and defaults to None.
    """

    legend_display = OptionProperty("marker", options=("marker", "both", "line"))
    """How to display this plot in the legend.

    Options:
        'marker':   Displays only the marker if :data:`marker_shape` is not None
                    and only the line otherwise.
        'both':     Displays line and marker.
        'line':     Displays only the line.

    :data:`legend_display` is a :class:`kivy.properties.OptionProperty`
    and defaults to 'marker'.
    """

    def __init__(self, **kwargs):
        # Initialize drawing instruction references
        self._color: Optional[Color] = None  # Color drawing instruction
        self._marker_color: Optional[Color] = None  # Marker color drawing instruction
        self._line: Optional[Line] = (
            None  # Drawing instruction for the line connecting the markers
        )
        self._marker_group: Optional[InstructionGroup] = (
            None  # Instruction group for markers
        )

        # Legend-related attributes
        self._legend_group: Optional[InstructionGroup] = None
        self._legend_line: Optional[Line] = None
        self._legend_marker: Optional[Instruction] = None
        self._legend_marker_center: Optional[Tuple[float, float]] = None
        self._legend_maximum_drawing_size: Optional[Tuple[float, float]] = None

        # List of drawing instructions for plot markers
        self._markers: List[Instruction] = []

        # Call super constructor
        super().__init__(**kwargs)

        # Bind property updates
        def update_marker_line_width(*_):
            for p in self._markers:
                if isinstance(p, Line):
                    p.width = self.marker_line_width
            if isinstance(self._legend_marker, Line):
                self._legend_marker.width = self.marker_line_width

        self.fbind("marker_line_width", update_marker_line_width)

        def update_line_width(*_):
            if self._line:
                self._line.width = self.line_width
            if self._legend_line:
                self._legend_line.width = min(
                    self.line_width, self._legend_maximum_drawing_size[1] / 2
                )

        self.fbind("line_width", update_line_width)

        # Redraw when marker properties change
        self.fbind("marker_size", lambda *_: self.draw_markers())
        self.fbind("marker_size", lambda *_: self.draw_legend())
        self.fbind("marker_shape", lambda *_: self.draw_markers(force_new=True))
        self.fbind("marker_shape", lambda *_: self.draw_legend(force_new=True))
        self.fbind("legend_display", lambda *_: self.draw_legend(force_new=True))

        def update_color(*_):
            if self._color:
                self._color.rgba = self.color
                self._marker_color.rgba = self.marker_color or self.color

        self.fbind("color", update_color)
        self.fbind("marker_color", update_color)

    def create_drawings(self):
        """Create the drawing instructions for the line and marker plot."""
        self._color = Color(*self.color)
        self._line = SmoothLine(width=self.line_width)
        self._marker_color = Color(*(self.marker_color or self.color))
        self._marker_group = InstructionGroup()
        return [self._color, self._line, self._marker_color, self._marker_group]

    def draw(self, *_):
        """Draw the line and markers."""
        super().draw()
        # Set the line's points (simplified and flattened)
        self._line.points = [
            xy for p in self.simplify_points(list(self.iterate_points())) for xy in p
        ]
        # Update all markers
        self.draw_markers()

    def get_marker(self):
        """Return a single marker drawing instruction based on marker_shape."""
        shape = self.marker_shape
        if shape is None:
            return Instruction()
        if shape in "x+*":
            return Line(width=self.marker_line_width)
        if shape in "-|<>v^sd":
            return Line(width=self.marker_line_width, joint="miter")
        if shape in "o":
            return SmoothLine(width=self.marker_line_width, close=True, joint="round")
        if shape == "O":
            return Ellipse()
        if shape == "S":
            return Rectangle()
        if shape == "D":
            return Mesh(indices=(0, 1, 2, 3), mode="triangle_fan")
        raise NotImplementedError()

    def draw_markers(self, force_new=False):
        """Draw all markers for the current points."""
        super().draw()

        # Remove excess markers
        while len(self.points) < len(self._markers) or (force_new and self._markers):
            self._marker_group.remove(self._markers.pop())

        # Add new markers as needed
        while len(self.points) > len(self._markers):
            self._markers.append(self.get_marker())
            self._marker_group.add(self._markers[-1])

        # Update marker positions
        for marker, point in zip(self._markers, self.iterate_points()):
            self.draw_marker(marker, self.marker_size, *point)

    @staticmethod
    def simplify_points(points: List[Tuple[float, float]]):
        """Delete points that lie exactly on the line connecting adjacent points.

        This is needed due to a bug in SmoothLine class.
        """
        points = points.copy()
        i = 0
        while len(points) > i + 2:
            # Check if three consecutive points are collinear
            try:
                slope1 = (points[i + 1][1] - points[i][1]) / (
                    points[i + 1][0] - points[i][0]
                )
                slope2 = (points[i + 2][1] - points[i][1]) / (
                    points[i + 2][0] - points[i][0]
                )
                if abs(slope1 - slope2) < 1e-5:
                    points.pop(i + 1)
                else:
                    i += 1
            except ZeroDivisionError:
                i += 1
        return points

    def draw_marker(self, marker: Instruction, s, x, y):
        """Draw a single marker at the specified position."""
        shape = self.marker_shape
        if shape is None:
            return

        # Calculate marker bounds
        x_left, x_right = x - s / 2, x + s / 2
        y_bottom, y_top = y - s / 2, y + s / 2
        d = 0.5 * sqrt(0.5) * s

        # Set marker geometry based on shape
        if shape in "x+*-|<>v^sd":
            if shape == "x":
                points = (
                    x_left,
                    y_bottom,
                    x_right,
                    y_top,
                    x,
                    y,
                    x_left,
                    y_top,
                    x_right,
                    y_bottom,
                )
            elif shape == "+":
                points = (x_left, y, x_right, y, x, y, x, y_bottom, x, y_top)
            elif shape == "*":
                points = (
                    x_left,
                    y,
                    x_right,
                    y,
                    x,
                    y,
                    x,
                    y_bottom,
                    x,
                    y_top,
                    x,
                    y,
                    x - d,
                    y - d,
                    x + d,
                    y + d,
                    x,
                    y,
                    x - d,
                    y + d,
                    x + d,
                    y - d,
                )
            elif shape == "-":
                points = (x_left, y, x_right, y)
            elif shape == "|":
                points = (x, y_bottom, x, y_top)
            elif shape == ">":
                points = (x_left, y_bottom, x_right, y, x_left, y_top)
            elif shape == "<":
                points = (x_right, y_bottom, x_left, y, x_right, y_top)
            elif shape == "v":
                points = (x_left, y_top, x, y_bottom, x_right, y_top)
            elif shape == "^":
                points = (x_left, y_bottom, x, y_top, x_right, y_bottom)
            elif shape == "s":
                points = (
                    x_left,
                    y_bottom,
                    x_left,
                    y_top,
                    x_right,
                    y_top,
                    x_right,
                    y_bottom,
                    x_left,
                    y_bottom,
                    x_left,
                    y_top,
                )
            elif shape == "d":
                points = (
                    x_left,
                    y,
                    x,
                    y_top,
                    x_right,
                    y,
                    x,
                    y_bottom,
                    x_left,
                    y,
                    x,
                    y_top,
                )
            marker.points = points
        elif shape == "o":
            marker.ellipse = (x_left, y_bottom, s, s)
        elif shape in "OS":
            marker.pos = x_left, y_bottom
            marker.size = s, s
        elif shape == "D":
            marker.vertices = (
                x_left,
                y,
                0,
                0,
                x,
                y_top,
                0,
                0,
                x_right,
                y,
                0,
                0,
                x,
                y_bottom,
                0,
                0,
            )
        else:
            raise NotImplementedError()

    def create_legend_drawings(self):
        """Create the drawing instructions for the legend."""
        self._legend_line = SmoothLine(width=self.line_width)
        self._legend_marker = self.get_marker()
        self._legend_group = InstructionGroup()
        self._legend_group.add(self._color)
        self._legend_group.add(self._legend_line)
        self._legend_group.add(self._marker_color)
        self._legend_group.add(self._legend_marker)
        return [self._legend_group]

    def draw_legend(self, center=None, maximum_size=None, force_new=False):
        """Draw the legend representation."""
        self._legend_maximum_drawing_size = maximum_size = (
            maximum_size or self._legend_maximum_drawing_size
        )
        self._legend_marker_center = center = center or self._legend_marker_center

        if not center:
            return

        # Recreate marker if needed
        if force_new:
            if self._legend_marker:
                self._legend_group.remove(self._legend_marker)
                self._legend_marker = None
            if self.legend_display != "line":
                self._legend_marker = self.get_marker()
                self._legend_group.add(self._legend_marker)

        # Draw marker
        if self._legend_marker:
            marker_size = min(self.marker_size, *maximum_size)
            self.draw_marker(self._legend_marker, marker_size, *center)

        # Draw line
        if self.legend_display == "marker" and self.marker_shape:
            self._legend_line.points = []
        else:
            self._legend_line.points = [
                center[0] - maximum_size[0] / 2,
                center[1],
                center[0] + maximum_size[0] / 2,
                center[1],
            ]


if __name__ == "__main__":
    import itertools
    from math import sin, cos, pi
    from random import randrange
    from kivy.utils import get_color_from_hex as rgb
    from kivy.uix.boxlayout import BoxLayout
    from kivy.app import App

    class TestApp(App):

        def build(self):
            b = BoxLayout(orientation="vertical")

            # Example of a custom theme
            colors = itertools.cycle(
                [rgb("7dac9f"), rgb("dc7062"), rgb("66a8d4"), rgb("e5b060")]
            )
            graph_theme = {
                "label_options": {
                    "color": rgb("444444"),  # Color of tick labels and titles
                    "bold": True,
                },
                "background_color": rgb("f8f8f2"),  # Canvas background color
                "tick_color": rgb("808080"),  # Ticks and grid
                "border_color": rgb("808080"),
            }  # Border drawn around each graph

            graph = GraphAA(
                xlabel="Cheese",
                ylabel="Apples",
                x_ticks_minor=5,
                x_ticks_major=25,
                y_ticks_major=1,
                y_grid_label=True,
                x_grid_label=True,
                padding=5,
                xlog=False,
                ylog=False,
                x_grid=True,
                y_grid=True,
                xmin=-50,
                xmax=50,
                ymin=-1,
                ymax=1,
                **graph_theme,
            )

            plot = LinePlot(color=next(colors))
            plot.points = [(x / 10.0, sin(x / 50.0)) for x in range(-500, 501)]
            # For efficiency, the x range matches xmin, xmax
            graph.add_plot(plot)

            plot = OptimizedSmoothLinePlot(color=next(colors))
            plot.points = [(x / 10.0, cos(x / 50.0)) for x in range(-500, 501)]
            graph.add_plot(plot)
            self.plot = plot  # Keep reference for moving graph

            plot = MeshStemPlot(color=next(colors))
            graph.add_plot(plot)
            plot.points = [(x, x / 50.0) for x in range(-50, 51)]

            plot = BarPlot(color=next(colors), bar_spacing=0.72)
            graph.add_plot(plot)
            plot.bind_to_graph(graph)
            plot.points = [(x, 0.1 + randrange(10) / 10.0) for x in range(-50, 1)]

            Clock.schedule_interval(self.update_points, 1 / 60.0)

            graph2 = GraphAA(
                xlabel="Position (m)",
                ylabel="Time (s)",
                x_ticks_minor=0,
                x_ticks_major=1,
                y_ticks_major=10,
                y_grid_label=True,
                x_grid_label=True,
                padding=5,
                xlog=False,
                ylog=False,
                xmin=0,
                ymin=0,
                **graph_theme,
            )
            b.add_widget(graph)

            if np is not None:
                (xbounds, ybounds, data) = self.make_contour_data()
                # This is required to fit the graph to the data extents
                graph2.xmin, graph2.xmax = xbounds
                graph2.ymin, graph2.ymax = ybounds

                plot = ContourPlot()
                plot.data = data
                plot.xrange = xbounds
                plot.yrange = ybounds
                plot.color = [1, 0.7, 0.2, 1]
                graph2.add_plot(plot)

                b.add_widget(graph2)
                self.contourplot = plot

                Clock.schedule_interval(self.update_contour, 1 / 60.0)

            # Test the scatter plot
            plot = ScatterPlot(color=next(colors), point_size=5)
            graph.add_plot(plot)
            plot.points = [(x, 0.1 + randrange(10) / 10.0) for x in range(-50, 1)]
            return b

        def make_contour_data(self, ts=0):
            """Generate sample contour data."""
            omega = 2 * pi / 30
            k = (2 * pi) / 2.0

            ts = sin(ts * 2) + 1.5  # Empirically determined 'pretty' values
            npoints = 100
            data = np.ones((npoints, npoints))

            position = [ii * 0.1 for ii in range(npoints)]
            time = [(ii % 100) * 0.6 for ii in range(npoints)]

            for ii, t in enumerate(time):
                for jj, x in enumerate(position):
                    data[ii, jj] = sin(k * x + omega * t) + sin(-k * x + omega * t) / ts
            return (0, max(position)), (0, max(time)), data

        def update_points(self, *args):
            """Update plot points for animation."""
            self.plot.points = [
                (x / 10.0, cos(Clock.get_time() + x / 50.0)) for x in range(-500, 501)
            ]

        def update_contour(self, *args):
            """Update contour plot data for animation."""
            _, _, self.contourplot.data[:] = self.make_contour_data(Clock.get_time())
            # This does not trigger an update, because we replace the
            # values of the array and do not change the object.
            # However, we cannot do "...data = make_contour_data()" as
            # kivy will try to check for the identity of the new and
            # old values. In numpy, 'nd1 == nd2' leads to an error
            # (you have to use np.all). Ideally, property should be patched
            # for this.
            self.contourplot.ask_draw()

    TestApp().run()
