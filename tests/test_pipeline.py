# Code from https://github.com/WeixuanZ/methodflow
# Original License:
# The MIT License (MIT)

# Copyright (c) 2023 Weixuan Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import pytest

from robust_dst.pipeline import PipelineMixin


@pytest.fixture()
def fake_owner_obj():
    class Cls(PipelineMixin):
        def __init__(self):
            self.to_run_d = False
            self.to_run_e = False

        @PipelineMixin.op()
        def a(self, data):
            data["a"] += 10
            return data

        @PipelineMixin.op()
        def b(self, results_from_a):
            return {"b": results_from_a["a"] - 10}

        @PipelineMixin.op()
        def c(self, results_from_b):
            return results_from_b

        @PipelineMixin.op(condition="to_run_d")
        def d(self, results_from_c):
            return {"d": True, **results_from_c}

        @PipelineMixin.op(condition="to_run_e")
        def e(self, results_from_c):
            results_from_c["e"] = True
            return results_from_c

        @PipelineMixin.op()
        def f(self, results_from_d, results_from_e):
            out = []
            if results_from_d is not PipelineMixin.StepSkipped:
                out.append(results_from_d)
            if results_from_e is not PipelineMixin.StepSkipped:
                out.append(results_from_e)
            return out

    return Cls()


@pytest.fixture()
def fake_data():
    return {
        "a": 10,
    }


def test_build_dag(fake_owner_obj):
    fake_owner_obj._build_dag()
    assert fake_owner_obj.a.successors == [fake_owner_obj.b]
    assert fake_owner_obj.b.successors == [fake_owner_obj.c]
    assert len(fake_owner_obj.a.successors) == 1


def test_topological_sort(fake_owner_obj):
    pipeline = fake_owner_obj._topological_sort(fake_owner_obj._build_dag())
    assert len(pipeline) == 3
    assert pipeline == [fake_owner_obj.a, fake_owner_obj.b, fake_owner_obj.c]


def test_topological_sort_condition(fake_owner_obj):
    fake_owner_obj.to_run_d = True
    pipeline = fake_owner_obj._topological_sort(fake_owner_obj._build_dag())
    assert len(pipeline) == 5
    assert pipeline == [
        fake_owner_obj.a,
        fake_owner_obj.b,
        fake_owner_obj.c,
        fake_owner_obj.d,
        fake_owner_obj.f,
    ]


def test_execute_pipeline(fake_owner_obj, fake_data):
    assert fake_owner_obj.execute_pipeline(fake_data) == {"b": 10}


def test_execute_pipeline_condition(fake_owner_obj, fake_data):
    fake_owner_obj.to_run_d = True
    fake_owner_obj.to_run_e = False
    assert fake_owner_obj.execute_pipeline(fake_data) == [{"b": 10, "d": True}]

    fake_owner_obj.to_run_d = False
    fake_owner_obj.to_run_e = True
    assert fake_owner_obj.execute_pipeline(fake_data) == [{"b": 20, "e": True}]

    fake_owner_obj.to_run_d = True
    fake_owner_obj.to_run_e = True
    assert fake_owner_obj.execute_pipeline(fake_data) == [
        {"b": 30, "d": True},
        {"b": 30, "e": True},
    ]
