# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"HiPlot support."""

from concurrent.futures import ProcessPoolExecutor
import math
import pydoc
import shlex
import typing as tp

import hiplot
from omegaconf import OmegaConf
from omegaconf.basecontainer import BaseContainer

from .xp import XP
from ._utils import get_main


def roundf(value: float, precision: int = 4):
    """Round value but returned as float, to make display nicer in Hiplot."""
    if not math.isfinite(value):
        return value
    return round(value * 10 ** precision) / 10**precision


class HiPlotExplorer:
    """You can inherit this class in order to make custom HiPlotExplorer,
    for instance to select a subset of the metrics."""
    def process_metrics(self, xp: XP, metrics: tp.Dict[str, tp.Any]):
        return metrics

    def process_history(self, xp: XP, history: tp.List[tp.Dict[str, tp.Any]]):
        return [self.process_metrics(xp, m) for m in history]

    def postprocess_exp(self, exp: hiplot.Experiment):
        """Use this method to further tune the `hiplot.Experiment` object,
        for instance setting a XY plot.
        """
        return


class STYLE:
    metrics = "badge badge-pill badge-primary"
    internal = "badge badge-pill badge-secondary"
    params = "badge badge-pill badge-dark"


def _flatten(dct, out=None, prefix=''):
    out = {} if out is None else out
    for key, value in dct.items():
        if isinstance(value, dict):
            _flatten(value, out=out, prefix=prefix + key + '.')
        else:
            out[prefix + key] = value
    return out


def load(uri: str) -> tp.Any:
    """Loader for hiplot
    Running: python -m hiplot dora.hiplot.load --port=XXXX
    will run an hiplot server. You can provide there a list of sigs or grid names, separated
    by spaces.

    To select metrics or further tune the display, you should inherit from
    `HiPlotExplorer`, very similar in spirit to the grid explorers.

    To specify the explorer, using `explorer=MyExplorer`.
    You can also change the module to look into with `explorer_module=` (default is
    `yourproject.grids._hiplot`).
    """
    main = get_main()

    sigs = set()
    explorer_module: tp.Optional[str] = None
    explorer_name = "HiPlotExplorer"
    value: tp.Any
    grids_name = main.dora.grid_package
    if grids_name is None:
        grids_name = main.package + ".grids"
    for token in shlex.split(uri):
        if '=' in token:
            key, value = token.split('=', 1)
            if key == 'explorer':
                explorer_name = value
                if explorer_module is None:
                    explorer_module = grids_name + '._hiplot'
            elif key == 'explorer_module':
                explorer_module = value
            else:
                raise ValueError(f"Invalid param {key}")
            continue
        grid_folder = main.dora.dir / main.dora._grids / token
        if grid_folder.exists():
            for child in grid_folder.iterdir():
                sigs.add(child.name)
        else:
            sigs.add(token)
    if explorer_module is None:
        explorer_module = 'dora.hiplot'
    explorer_qualified = explorer_module + "." + explorer_name
    explorer_klass = pydoc.locate(explorer_qualified)
    assert explorer_klass is not None, explorer_qualified
    explorer = explorer_klass()  # type: ignore

    with ProcessPoolExecutor(10) as pool:
        xps = list(pool.map(main.get_xp_from_sig, sigs))

    exp = hiplot.Experiment()
    if not xps:
        return exp

    # see dora/names.py
    reference = main.get_name_parts(xps[0])
    xps_name_parts = []
    all_columns = set()
    for xp in xps:
        parts = main.get_name_parts(xp)
        for key, val in parts.items():
            all_columns.add(key)
            if key in reference and reference[key] != val:
                reference.pop(key)

        missing = set(reference.keys()) - set(parts.keys())
        for key in missing:
            reference.pop(key)
        xps_name_parts.append(parts)
    all_columns -= set(reference.keys())
    for xp, parts in zip(xps, xps_name_parts):
        values: tp.Dict[str, tp.Any] = {}
        for key, value in parts.items():
            if key not in reference:
                sname = main.short_name_part(key, value).split('=', 1)[0]
                values[sname] = value
                exp.parameters_definition[sname].label_css = STYLE.params
        for key in all_columns:
            if key not in parts:
                try:
                    value = eval('xp.cfg.' + key, {'xp': xp})
                except AttributeError:
                    value = None
                sname = main.short_name_part(key, value).split('=', 1)[0]
                values[sname] = value
        for key, value in values.items():
            if isinstance(value, BaseContainer):
                value = OmegaConf.to_container(value, resolve=True)
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            values[key] = value
        values['sig'] = xp.sig
        from_uid: tp.Optional[str] = None
        xp.link.load()
        history = explorer.process_history(xp, xp.link.history)
        metric_names = set()
        for k, metrics in enumerate(history):
            point_values = dict(values)
            point_values['epoch'] = k
            point_values['last'] = k == len(xp.link.history) - 1
            flat_metrics = _flatten(metrics)
            point_values.update(flat_metrics)
            dp = hiplot.Datapoint(
                uid=f"{xp.sig}_{k}",
                from_uid=from_uid,
                values=point_values)
            from_uid = dp.uid
            exp.datapoints.append(dp)
            for key in flat_metrics.keys():
                metric_names.add(key)
                exp.parameters_definition[key].label_css = STYLE.metrics

    exp.display_data(hiplot.Displays.PARALLEL_PLOT).update({
        'hide': ['from_uid', 'uid'],
        'order': ['last', 'epoch'] + list(metric_names),
    })
    exp.display_data(hiplot.Displays.TABLE).update({
        'hide': ['from_uid'],
        'order': ['sig', 'last', 'epoch'] + list(metric_names),
    })
    exp.parameters_definition['epoch'].label_css = STYLE.internal
    exp.parameters_definition['last'].label_css = STYLE.internal
    exp.parameters_definition['sig'].label_css = STYLE.internal
    explorer.postprocess_exp(exp)
    return exp
