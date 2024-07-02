import pytest
from dataclasses import dataclass, field
import blocks as bk


def test_base_sampler():

    class MySampler(bk.BaseSampler):
        pass

    with pytest.raises(TypeError):
        MySampler()


def test_base_transformer():

    class MyTransformer(bk.BaseTransformer):
        pass

    with pytest.raises(TypeError):
        MyTransformer()


def test_base_transformer_check_kwargs():

    class MyTransformer(bk.BaseTransformer):
        TRANSFORMERS = {'func1': 'a', 'func2': 'b'}

        def __init__(self, select: str, **kwargs):
            self.select = select
            self.kwargs = kwargs

        def __call__(cls):
            pass

    my_transformer = MyTransformer('func1')
    assert my_transformer.check_kwargs("hello", "hello") == None
    assert my_transformer.check_kwargs("func2", "b") == None
    with pytest.raises(ValueError):
        my_transformer.check_kwargs("func1", "a")

    my_transformer = MyTransformer('func1', a='a')
    assert my_transformer.check_kwargs("hello", "hello") == None
    assert my_transformer.check_kwargs("func2", "b") == None
    assert my_transformer.check_kwargs("func1", "a") == None


def test_base_factor():

    @dataclass
    class MyFactor(bk.BaseFactor):
        tags: list = field(default_factory=lambda: [])
        name: str = ""
        X: str = ""
        y: str = None
        market_feature: str = ""
        inputs: dict = field(default_factory=lambda: {})
        outputs: dict = field(default_factory=lambda: {})
        pipeline: tuple = field(default_factory=lambda: ())
        
    assert MyFactor().__class__.__name__ == "MyFactor"
