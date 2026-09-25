# Agent 01 - read chart: describe one panel

Read ONE chart panel (image 1) at full resolution and describe its axes, tick values, legend and series, plus
rough readings that constrain the extraction (where each series runs, how many markers it has, at which
pressures the instrument measured). Tick VALUES matter (they calibrate pixels to units); tick POSITIONS and
every data coordinate are found by Python, not by you. Your readings may be +/-10 %: they are used to tell
series apart and to validate, never as data.

Agent 00 supplied only routing and document context. Treat it as context, not as a series decision:

```json
$discovery_context
```

Apply this extraction policy to every legend series:

```json
$co2_scope
```

## Rules

- x.ticks / y.ticks: ONLY ticks that carry a printed number, in the order they appear (x left to right, y bottom to top).
- series: in the ORDER they appear in the legend, top to bottom (this order maps colors to labels). Include every
  series, also the ones that are not CO2 (extract=false) - they are needed to tell similar colors apart.
- Set `extract=true` only when the legend/caption binding establishes that the individual series is CO2 and the
  plotted quantity is eligible under the policy. Do not inherit eligibility merely because agent 00 routed the figure.
- `species_evidence` must state the exact legend/caption binding used for that series. Use an empty string if unresolved.
- Insets are excluded. Describe them only in `notes`; do not count their markers, read their axes, or include them
  in `y_anchors`, `x_setpoints_visible` or any extraction decision.
- Distinguish repeated colors with marker shape and open/filled state. Do not merge series because their colors match.
- marker: name the outline exactly as drawn in the legend. Triangles are named by where the apex points
  (`triangle_up`, `triangle_down`, `triangle_left`, `triangle_right`); a square rotated 45° is `diamond`; `plus` is +,
  `cross` is ×. `color_hex` is the marker's own color (its outline color for open markers).
- legend_bbox_norm: tight box around the legend symbols AND text, as fractions 0-1 of this image ([x0, y0, x1, y1]).
- y_anchors: for each series, your reading of its y value at 25 %, 50 %, 75 % and 100 % of the x-axis range
  (the last one = where the series ends at the right). +/-10 % is fine.
- n_markers_estimate: how many markers the series has in total (count the clear part, extrapolate the crowded part).
- x_setpoints_visible: the pressures at which the CLEAREST series has its markers, read from the x axis
  (e.g. [0, 5, 10, 25, 50, 100, 150]). Instruments use the same set-points for all temperatures; give as many as you can.
- branches: "adsorption_only" if one line per series, "adsorption_and_desorption" if each series shows two lines
  (a loop / double line), "unknown" otherwise.
- notes: anything unusual - inset, broken axis, dual y-axis, error bars, filled vs open markers, log axis.

## Names - use exactly these

$vocabulary

Fill every field. This example shows the response shape only:

```json
$example
```
