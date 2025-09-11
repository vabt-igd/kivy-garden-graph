<!-- [![Coverage Status](https://coveralls.io/repos/github/kivy-garden/graph/badge.svg?branch=master)](https://coveralls.io/github/kivy-garden/graph?branch=master)
[![Github Build Status](https://github.com/kivy-garden/graph/workflows/Garden%20flower/badge.svg)](https://github.com/kivy-garden/graph/actions) -->

<!-- See https://kivy-garden.github.io/graph/ for the rendered graph docs. -->

# Graph

The `Graph` widget is a widget for displaying plots. It supports
drawing multiple plot with different colors on the Graph. It also supports
a title, ticks, labeled ticks, grids and a log or linear representation on
both the x and y axis, independently.

![Screenshot](/screenshot.png)

To display a plot. First create a graph which will function as a "canvas" for
the plots. Then create plot objects e.g. MeshLinePlot and add them to the
graph.

To create a graph with x-axis between 0-100, y-axis between -1 to 1, x and y
labels of and X and Y, respectively, x major and minor ticks every 25, 5 units,
respectively, y major ticks every 1 units, full x and y grids and with
a red line plot containing a sin wave on this range::

```python
from math import sin
from kivy_garden.graph import Graph, MeshLinePlot
graph = Graph(xlabel='X', ylabel='Y', x_ticks_minor=5,
x_ticks_major=25, y_ticks_major=1,
y_grid_label=True, x_grid_label=True, padding=5,
x_grid=True, y_grid=True, xmin=-0, xmax=100, ymin=-1, ymax=1)
plot = MeshLinePlot(color=[1, 0, 0, 1])
plot.points = [(x, sin(x / 10.)) for x in range(0, 101)]
graph.add_plot(plot)
```

The `MeshLinePlot` plot is a particular plot which draws a set of points using
a mesh object. The points are given as a list of tuples, with each tuple
being a (x, y) coordinate in the graph's units.

You can create different types of plots other than `MeshLinePlot` by inheriting
from the `Plot` class and implementing the required functions. The `Graph` object
provides a "canvas" to which a Plot's instructions are added. The plot object
is responsible for updating these instructions to show within the bounding
box of the graph the proper plot. The Graph notifies the Plot when it needs
to be redrawn due to changes. See the `MeshLinePlot` class for how it is done.

## Extended example (from the library)

This example showcases several plot types (smooth line, mesh line, bar, stem, scatter, contour), grids, axis labels, and animation. You can run it as a Kivy app.

```python
from math import sin, cos, pi
import itertools
from random import randrange

try:
    import numpy as np
except ImportError:
    np = None

from kivy.utils import get_color_from_hex as rgb
from kivy.uix.boxlayout import BoxLayout
from kivy.app import App
from kivy.clock import Clock

from kivy_garden.graph import (
    Graph,
    MeshLinePlot, MeshStemPlot, SmoothLinePlot, LinePlot,
    ScatterPlot, PointPlot, ContourPlot, BarPlot,
)

class TestApp(App):
    def build(self):
        b = BoxLayout(orientation='vertical')

        # Example theme
        colors = itertools.cycle([
            rgb('7dac9f'), rgb('dc7062'), rgb('66a8d4'), rgb('e5b060')
        ])
        graph_theme = {
            'label_options': {'color': rgb('444444'), 'bold': True},
            'background_color': rgb('f8f8f2'),
            'tick_color': rgb('808080'),
            'border_color': rgb('808080'),
        }

        graph = Graph(
            xlabel='Cheese',
            ylabel='Apples',
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
            **graph_theme
        )

        # Smooth line
        plot = SmoothLinePlot(color=next(colors))
        plot.points = [(x / 10.0, sin(x / 50.0)) for x in range(-500, 501)]
        graph.add_plot(plot)

        # Mesh line
        plot = MeshLinePlot(color=next(colors))
        plot.points = [(x / 10.0, cos(x / 50.0)) for x in range(-500, 501)]
        graph.add_plot(plot)
        self.plot = plot  # keep reference for animation

        # Stem plot
        plot = MeshStemPlot(color=next(colors))
        graph.add_plot(plot)
        plot.points = [(x, x / 50.0) for x in range(-50, 51)]

        # Bar plot
        plot = BarPlot(color=next(colors), bar_spacing=.72)
        graph.add_plot(plot)
        plot.bind_to_graph(graph)
        plot.points = [(x, .1 + randrange(10) / 10.0) for x in range(-50, 1)]

        # Scatter
        plot = ScatterPlot(color=next(colors), point_size=5)
        graph.add_plot(plot)
        plot.points = [(x, .1 + randrange(10) / 10.0) for x in range(-50, 1)]

        Clock.schedule_interval(self.update_points, 1 / 60.0)
        b.add_widget(graph)

        # Second graph with a contour plot (if NumPy is available)
        if np is not None:
            graph2 = Graph(
                xlabel='Position (m)',
                ylabel='Time (s)',
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
                **graph_theme
            )
            (xbounds, ybounds, data) = self.make_contour_data()
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

        return b

    def make_contour_data(self, ts=0):
        omega = 2 * pi / 30
        k = (2 * pi) / 2.0

        ts = sin(ts * 2) + 1.5  # visually pleasing values
        npoints = 100
        data = np.ones((npoints, npoints))

        position = [ii * 0.1 for ii in range(npoints)]
        time = [(ii % 100) * 0.6 for ii in range(npoints)]

        for ii, t in enumerate(time):
            for jj, x in enumerate(position):
                data[ii, jj] = sin(k * x + omega * t) + sin(-k * x + omega * t) / ts
        return (0, max(position)), (0, max(time)), data

    def update_points(self, *args):
        self.plot.points = [
            (x / 10.0, cos(Clock.get_time() + x / 50.0))
            for x in range(-500, 501)
        ]

    def update_contour(self, *args):
        _, _, self.contourplot.data[:] = self.make_contour_data(Clock.get_time())
        # We modify the array in-place; ask the plot to redraw.
        self.contourplot.ask_draw()

if __name__ == '__main__':
    TestApp().run()
```

### Available plot types

- **MeshLinePlot** — mesh-based polyline (fast, classic)
- **MeshStemPlot** — stems from baseline (y=0) to points
- **LinePlot** — simple line primitive
- **SmoothLinePlot** — anti-aliased lines (shader-based)
- **OptimizedSmoothLinePlot** — optimized single-line renderer for real-time plotting (optional AA, auto-cleanup)
- **ScatterPlot** — points using Point primitive
- **PointPlot** — alternative point renderer
- **BarPlot** — bars (with auto width and spacing)
- **HBar** — horizontal reference lines at given y’s
- **VBar** — vertical reference lines at given x’s
- **ContourPlot** — intensity/heatmap from 2D data
- **LineAndMarkerPlot** — line with rich marker shapes (incl. filled)

### Legend

- Basic usage: `graph.legend = [('sine', plot1), ('cosine', plot2)]`
- Position via pos_hint relative to the plotting area, e.g. `{'top': 1, 'right': 1}`
- Marker size limits via `GraphLegend.marker_size`
- Plots can provide custom legend drawings (`create_legend_drawings`, `draw_legend`)

### Multiple axes

- Add extra axes: `graph.add_x_axis(xmin, xmax, xlog=False)` / `graph.add_y_axis(...)`
- Per-plot axis selection via `plot.x_axis` / `plot.y_axis`
## NOTICE

This repository is based on the Kivy Garden "graph" project and downstream forks.

Original project:

- Kivy Garden Graph (MIT)
  <https://github.com/kivy-garden/graph>

Intermediate fork:

- Kivy Garden Graph (MIT)
  <https://github.com/HoerTech-gGmbH/kivy-garden-graph.git>

We thank the original authors and contributors for their work.

## Changelog

Reconstructed changelog of the Kivy Garden "graph" project, downstream forks and this fork: [**CHANGELOG**](/CHANGELOG)

## Contributing

Check out our [contribution guide](CONTRIBUTING.md) and feel free to improve the flower.

## License

This software is released under the terms of the MIT License.
Please see the [LICENSE.txt](LICENSE.txt) file.

