from __future__ import annotations

import copy
import inspect
import logging
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)


class ProcessStep:
    """Wrapper of Callable, with info to build the pipeline."""

    def __init__(
        self, process_func: Callable, condition_attrs: list[str] = None
    ) -> None:
        self._func = process_func
        self._bound_func: Optional[Callable] = None
        self._condition_attrs = condition_attrs or []

        self.predecessors: list[ProcessStep] = []
        self.successors: list[ProcessStep] = []
        self.to_run: bool = True
        self.color: Literal["w", "b", "g"] = "w"

    def reset(self):
        self.predecessors = []
        self.successors = []
        self.to_run = True
        self.color = "w"

    @property
    def func_arg_names(self) -> list[str]:
        return list(inspect.signature(self._func).parameters.keys())

    @property
    def func(self) -> Callable:
        return self._bound_func or self._func

    def __hash__(self) -> int:
        return hash(self.func)

    def __repr__(self) -> str:
        return f"ProcessStep{self.func}"

    @property
    def condition_attrs(self) -> list[str]:
        return self._condition_attrs

    @property
    def in_degree(self) -> int:
        return len(self.predecessors)

    @property
    def out_degree(self) -> int:
        return len(self.successors)

    @property
    def out_to_run_degree(self) -> int:
        return sum(1 for _ in filter(lambda n: n.to_run, self.successors))

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)

    def __set__(self, obj, value) -> None:
        """Make ProcessStep a data descriptor."""
        raise AttributeError("Cannot change the value")

    def __get__(self, obj, objtype=None) -> ProcessStep:
        """Return the method wrapped."""
        if hasattr(self._func, "__get__"):
            self._bound_func = self._func.__get__(obj, objtype)
        elif obj is not None:
            self._bound_func = lambda *args, **kwargs: self._func(
                obj or objtype, *args, **kwargs
            )
        return self


class PipelineMixin:
    """Decorator and methods for building a processing pipeline."""

    class StepSkipped:
        pass

    @staticmethod
    def op(condition: Optional[str] = "") -> Callable[[Callable], ProcessStep]:
        """Decorator for registering a preprocess method.

        This needs to be the top-most decorator if using stacked decorators
        (e.g. @classmethod, @staticmethod).

        Args:
            condition (str): Comma separated string of attribute names used as
                predicates for the annotated method. The method will run if all the
                predicates are True at runtime.
        """
        if condition is None:
            condition = ""
        if not isinstance(condition, str):
            raise ValueError("condition must be a comma separated string")
        condition = [s.strip() for s in condition.split(",") if len(s) > 0]

        def decorator(process_func: Callable):
            return ProcessStep(process_func, condition)

        return decorator

    def _get_preprocess_steps(self) -> dict[str, ProcessStep]:
        return dict(inspect.getmembers(self, lambda m: isinstance(m, ProcessStep)))

    def _resolve_condition(self, step: ProcessStep) -> bool:
        if not isinstance(step, ProcessStep):
            raise RuntimeError("The given callable is not a processing step.")
        for condition_attr in step.condition_attrs:
            if not hasattr(self, condition_attr):
                logging.warning(
                    "The owner class of the pipeline does not have the condition"
                    f"attribute {condition_attr}, defaulting to True."
                )
            elif not getattr(self, condition_attr):
                return False
        return True

    def _build_dag(self) -> list[ProcessStep]:
        """Build the DAG of processing methods.

        Returns:
            list[ProcessStep]: The starting nodes of the DAG.
        """
        # mapping from names to nodes
        name_to_node = self._get_preprocess_steps()
        for node in name_to_node.values():
            node.reset()

        # resolve the conditions
        for node in name_to_node.values():
            node.to_run = self._resolve_condition(node)

        # resolve the connections
        for node in name_to_node.values():
            for func_arg_name in node.func_arg_names:
                if not func_arg_name.startswith("results_from_"):
                    continue
                previous_step_name = func_arg_name[13:]
                if previous_step_name not in name_to_node:
                    raise RuntimeError(
                        f"{previous_step_name} is not an available method."
                    )
                name_to_node[previous_step_name].successors.append(node)
                node.predecessors.append(name_to_node[previous_step_name])

        start_nodes = list(filter(lambda n: n.in_degree == 0, name_to_node.values()))
        return start_nodes

    @staticmethod
    def _topological_sort(start_nodes: list[ProcessStep]) -> list[ProcessStep]:
        """Perform a topological sort on the pipeline DAG."""
        pipeline = []
        for start_node in start_nodes:
            if start_node.color == "b":
                continue
            stack = [start_node]
            while stack:
                node = stack[-1]
                if node.color == "w":
                    node.color = "g"
                    for next_node in node.successors:
                        if next_node.color == "w" and next_node.to_run:
                            stack.append(next_node)
                        elif next_node.color == "g":
                            raise RuntimeError("Cycle in graph")
                else:
                    stack.pop()
                    if node.color == "g" and node.to_run:
                        node.color = "b"
                        pipeline.append(node)

        return pipeline[::-1]

    def execute_pipeline(self, *args, **kwargs) -> Any:
        """Execute the pipeline.

        The arguments are passed to the starting preprocess steps.
        """
        pipeline = self._topological_sort(self._build_dag())
        intermediate_results = {}
        output = []
        for step in pipeline:
            if step.to_run:
                if step.in_degree == 0:
                    logger.info(f"Executing {step}")
                    intermediate_results[step] = step(*args, **kwargs)
                else:
                    params_from = list(
                        filter(lambda p: p in intermediate_results, step.predecessors)
                    )
                    logger.info(f"Executing {step}, with outputs from {params_from}")
                    params = []
                    for predecessor in step.predecessors:
                        if predecessor.out_to_run_degree > 1:
                            if predecessor in intermediate_results:
                                params.append(
                                    copy.deepcopy(intermediate_results[predecessor])
                                )
                            else:
                                params.append(PipelineMixin.StepSkipped)
                        else:
                            params.append(
                                intermediate_results.get(
                                    predecessor, PipelineMixin.StepSkipped
                                )
                            )
                    intermediate_results[step] = step(*params)
                if step.out_to_run_degree == 0:
                    output.append(intermediate_results[step])
            else:
                logger.info(f"Skipping {step}")
                intermediate_results[step] = PipelineMixin.StepSkipped

        return output[0] if len(output) == 1 else tuple(output)