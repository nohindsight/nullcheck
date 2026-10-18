"""Conservative external-input inventory, not a Python sandbox or complete data-flow proof.
Observe read opens during callbacks and referenced pre-existing data in user functions.
Never open/hash/copy observed external files: their contents may be private or unbounded.
A process-wide audit hook is inert outside the context-local observation window.
"""
from __future__ import annotations

import ast
import contextlib
import dis
import contextvars
import inspect
import os
import sys
import types
import textwrap

import numpy as np
import pandas as pd

_active = contextvars.ContextVar("nullcheck_external_reads", default=None)
_installed = False
_DATA = (np.ndarray, pd.Series, pd.DataFrame, pd.Index, list, tuple, dict, set, frozenset)
_REASON = "External precomputed inputs not assessed: recomputation of this input from the truncated raw prefix is not established."
_DEFAULT_REASON = ("Default-argument state not assessed: this pre-existing container or object can retain values "
                   "between calls; recomputation from the truncated raw prefix is not established.")


def inventory_reason(inputs):
    kinds = []
    if any(not key.startswith("input:default:") for key in inputs):
        kinds.append("External precomputed inputs not assessed")
    if any(key.startswith("input:default:") for key in inputs):
        kinds.append("Default-argument state not assessed")
    return "; ".join(kinds) + "; recomputation from each truncated prefix is not established."


def _audit(event, args):
    inputs = _active.get()
    if inputs is None or event != "open":
        return
    path, mode, flags = args
    reading = (isinstance(mode, str) and ("r" in mode or "+" in mode)) or (mode is None and flags & os.O_ACCMODE != os.O_WRONLY)
    if not reading:
        return
    if isinstance(path, (str, bytes)):
        path = os.fsdecode(path)
        # Interpreter source/import reads are code provenance, not precomputed data.
        caller = sys._getframe(1).f_code
        if path.endswith((".py", ".pyc", ".pyo")) and "importlib" in caller.co_filename and caller.co_name == "get_data":
            return
        label = os.path.abspath(path)
    else:
        label = f"file descriptor {path}"
    inputs["input:file:" + label] = _REASON


def _inert_calls_only(fn, name, value):
    """A source-proven discarded call to a literal no-op is not an input read.

    Do not generalize this to arbitrary unused method results: such methods can
    mutate input arrays, fill prediction buffers, or read data through side effects.
    Unknown/dynamic/builtin methods remain conservative external dependencies.
    """
    if isinstance(value, _DATA) or inspect.getattr_static(type(value), "__getattribute__", None) is not object.__getattribute__:
        return False
    if any("__getattr__" in vars(base) for base in type(value).__mro__):
        return False
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        uses = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == name]
        if not uses:
            return False
        for node in uses:
            attr = parents.get(node)
            call = parents.get(attr)
            if not (isinstance(node.ctx, ast.Load) and isinstance(attr, ast.Attribute) and attr.value is node
                    and isinstance(call, ast.Call) and call.func is attr and isinstance(parents.get(call), ast.Expr)):
                return False
            method = inspect.getattr_static(value, attr.attr, None)
            if isinstance(method, (staticmethod, classmethod)):
                method = method.__func__
            if not inspect.isfunction(method):
                return False
            for instruction in dis.get_instructions(method):
                if instruction.opname not in {"RESUME", "NOP", "CACHE", "COPY_FREE_VARS", "LOAD_CONST", "RETURN_CONST", "RETURN_VALUE"}:
                    return False
                if instruction.opname in {"LOAD_CONST", "RETURN_CONST"} and instruction.argval is not None:
                    return False
        return True
    except (TypeError, ValueError, OSError, SyntaxError):
        return False


class ExternalInputs:
    def __init__(self, fn):
        global _installed
        if not _installed:
            sys.addaudithook(_audit)
            _installed = True
        self.inputs = {}
        self.seen = set()
        roots = getattr(fn, "_nullcheck_input_roots", (fn,))
        self.user_files = set()
        for root in roots:
            if root is None:
                continue
            candidates = root.__mro__ if inspect.isclass(root) else (root,)
            for candidate in candidates:
                if getattr(candidate, "__module__", "").startswith("backtesting"):
                    continue
                try:
                    filename = inspect.getsourcefile(candidate)
                except TypeError:
                    filename = None
                if filename:
                    self.user_files.add(filename.replace("\\", "/"))
        for name, value in getattr(fn, "_nullcheck_input_values", {}).items():
            self._dependency(value, "parameter." + str(name), 0)
        for root in roots:
            if root is not None:
                self._visit(root, getattr(root, "__qualname__", type(root).__qualname__), 0)

    def _dependency(self, value, label, depth, default=False):
        if isinstance(value, _DATA):
            self.inputs[("input:default:" if default else "input:memory:") + label] = _DEFAULT_REASON if default else _REASON
        elif inspect.isfunction(value) or inspect.ismethod(value):
            self._visit(value, label, depth + 1)
        elif not (value is None or isinstance(value, (str, bytes, int, float, complex, bool, np.generic, os.PathLike, types.ModuleType, type))):
            self.inputs[("input:default:" if default else "input:object:") + label] = (_DEFAULT_REASON if default else _REASON) + " Opaque pre-existing object internals are not inspected."

    def _visit(self, fn, label, depth):
        if id(fn) in self.seen:
            return
        self.seen.add(id(fn))
        if depth > 8 or len(self.seen) > 256:
            self.inputs["input:inspection_limit"] = _REASON + " Dependency inspection budget exhausted."
            return
        if inspect.isclass(fn):
            # Only init and its user-defined helpers; next() is not run by this adapter.
            for base in fn.__mro__:
                if base.__module__.startswith("backtesting"):
                    break
                method = vars(base).get("init")
                if method is not None:
                    self._visit(method, label + ".init", depth + 1)
                    names = method.__code__.co_names if inspect.isfunction(method) else ()
                    for name in names:
                        for owner in fn.__mro__:
                            if owner.__module__.startswith("backtesting"):
                                break
                            if name in vars(owner):
                                self._dependency(vars(owner)[name], label + "." + name, depth)
                                break
            return
        if inspect.ismethod(fn):
            self._dependency(fn.__self__, label + ".bound_self", depth)
            fn = fn.__func__
        if not inspect.isfunction(fn):
            self._dependency(fn, label, depth)
            return
        # Do not walk library implementations; files they read during execution are still observed.
        filename = fn.__code__.co_filename.replace("\\", "/")
        if ("/site-packages/" in filename or "/dist-packages/" in filename) and filename not in self.user_files:
            return
        names = set(fn.__code__.co_names)
        for constant in fn.__code__.co_consts:
            if isinstance(constant, types.CodeType):
                names.update(constant.co_names)
        for name in sorted(names):
            if name in fn.__globals__:
                value = fn.__globals__[name]
                if not _inert_calls_only(fn, name, value):
                    self._dependency(value, label + ".global." + name, depth)
                # Module-held precomputed arrays (e.g. predictions.values) are common.
                module_path = (getattr(value, "__file__", "") or "").replace("\\", "/") if isinstance(value, types.ModuleType) else ""
                if isinstance(value, types.ModuleType) and "/site-packages/" not in module_path:
                    for attr in sorted(names):
                        candidate = vars(value).get(attr)
                        if isinstance(candidate, _DATA) or inspect.isfunction(candidate):
                            self._dependency(candidate, label + ".global." + name + "." + attr, depth)
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if not _inert_calls_only(fn, name, value):
                self._dependency(value, label + ".closure." + name, depth)
        for i, value in enumerate(fn.__defaults__ or ()):
            self._dependency(value, label + f".default[{i}]", depth, default=True)
        for name, value in (fn.__kwdefaults__ or {}).items():
            self._dependency(value, label + ".default." + name, depth, default=True)

    @contextlib.contextmanager
    def observe(self):
        token = _active.set(self.inputs)
        try:
            yield
        finally:
            _active.reset(token)
