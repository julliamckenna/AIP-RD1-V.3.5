# Agent 02 - check extraction: is the Python result right? (round $round of $max_rounds)

Python has extracted this panel from the structure below. You decide whether it is right. If it is not, return
a corrected structure: it replaces the current one and Python extracts again. This repeats until you are
satisfied or the round limit is reached.

Images, in order:

1. `panel.png` - the unchanged source panel. It is the only truth.
$image_list

Current structure used by Python (agent 01 reading plus earlier corrections):

```json
$current_structure
```

Python result:

$extraction_status

```json
$python_result
```

## Task

Inspect image 1 yourself, then judge the Python result against it:

- **Numbers first (`axis_check`)** - before anything else, compare the extracted values with the printed axes. For
  each axis write the lowest and highest printed tick label you read and the lowest and highest extracted value (copy
  them from `hard_checks.axis_range`). Python checks the value range itself, with tolerance - a marker centred on the
  axis line or just above the top tick is normal. Your `result` judges what only a reader can: `fail` when the
  re-plotted chart puts the curves at other values than the source against its ticks (shifted or stretched axis,
  wrongly read tick labels), e.g. a point at the right edge of the source plot re-plots at 180 kPa on an axis printed
  0-120. The overlay can not show this: markers found at the right pixels still get wrong values when the
  tick-to-pixel mapping is shifted. A `fail` always means `satisfied=false` and corrected `ticks` / `tick_anchors`.
- **Calibration** - do the overlay markers sit on the source markers, and do their values match the printed ticks?
  Isotherm axes normally start at 0 (`minimum=0` when the first printed tick is 0); a first point far from the origin
  on linear axes, or a curve that would cross zero at a positive pressure, points to a shifted mapping.
  For each axis return at least two `tick_anchors`, each pairing a printed tick value with its normalized pixel
  position along that axis (x left-to-right, y top-to-bottom). Set `boundary_limits_confirmed=true` only when the
  axis frame visibly extends beyond its outer printed tick. When Python lists tick candidates, read the printed
  label at each candidate and copy its `pixel_norm` exactly; never replace measured positions with equal spacing.
- **Series identity** - legend box, total series count, exact top-to-bottom order, and every label, marker shape,
  open/filled state, line style and hexadecimal color. A wrong color or shape makes Python assign markers to the
  wrong series. `python_result.legend_marker_check` puts each declared style next to the glyph Python measured from
  the legend pixels (shape incl. triangle direction, fill, stroke/face color; `source: pdf_vector` = exact PDF path),
  and `python_legend_markers.png` shows both. Resolve every listed mismatch: correct the declaration when the source
  agrees with the measurement, keep it when the measurement is wrong (e.g. a legend symbol overlapping its line).
- **Coverage** - clusters of missed or false markers. Mark those areas as `complex_regions` (typed `kind`:
  `dense_markers`, `overlapping_markers`, `thick_connecting_lines`, `crossing`, `inset` or `annotation`) with more
  `python_passes`.
- **Coach Python** - you see what Python can not; tell it. `count_check`, the redraw score in `hard_checks.redraw`
  (red in `python_redraw_diff.png` = marker ink nobody extracted) and the re-plot show where it is short.
- **Extraction settings** - when points land on the wrong series or are missed, return `extraction_settings` for
  just those series (empty list otherwise):
  - `sample_marker_norm`: the [x, y] position of ONE clean marker of the series that sits alone on the plot. Python
    measures the series color there instead of trusting the legend - the fix for a series Python finds few or no
    markers of (null when the legend color works);
  - `expected_counts`: how many markers of the series you see per x range; Python reports its own count next to each,
    so the next round shows exactly where it is short;
  - `curve_points`: the series curve as 6-12 (x, y) points read from the axes, densest where it bends. Python uses it
    to give crowded markers (e.g. the low-pressure fan) to the right series; it holds over its whole x range.
    An adsorption isotherm starts at the origin (zero uptake at zero pressure): on linear axes printed from 0, start
    the curve at (0, 0) and read the steep low-pressure rise densely - Python adds (0, 0) when you leave it out.
    Do not force the origin on a log pressure axis, an axis that starts above 0, or a desorption branch that visibly
    closes above zero;
  - `separate_from`: same-color series it must be told apart from (e.g. a CO2 and an N2 series drawn in the same black);
  - `y_bands`: where its markers actually lie, read from the axes, as one box per x range - use two or three boxes to
    follow a curved isotherm (steep near zero pressure, flat later). Points outside go to a twin or are dropped - the
    strongest fix when same-color curves run apart;
  - `color_tolerance` (15-45): tighter when a neighbouring series has a similar color, wider for pale or blurred rings;
  - `template_fill`: `off` or `fit_only` when Python's filled-in crowded-strip points are wrong for that series.
  Python clamps every value and never loosens the calibration checks.
- **Region strategies** - `region_strategies` tell Python how to handle one box (empty list when not needed):
  - `ignore` - no points of the listed series there (inset, annotation, text); empty `series_labels` = all series;
  - `split_filled_open` - list the same-color filled and open series (e.g. adsorption and desorption); Python gives
    each marker in the box to the filled or the open one by the ink at its centre;
  - `sample_band_at_columns` - markers fused into a solid band; Python takes one line sample per value in
    `shared_x_columns` (pressures where the series share marker columns, read from nearby separate markers). These
    rows are inferred and always go to human review - use it only where no single marker can be seen.
- **Branches** - `adsorption_and_desorption` only when two branches are visibly separable, otherwise `unknown` or
  `adsorption_only`.

Set `satisfied=true` only when both axes pass `axis_check`, calibration and series identity are correct and no
fixable coverage problem remains;
single ambiguous markers are left for agent 03, which edits individual points. Otherwise set `satisfied=false`, list the
problems in `issues`, and return the complete corrected structure (not only differences). Use normalized boxes
relative to image 1.

## Names - use exactly these

$vocabulary

This example shows the response shape only:

```json
$example
```
