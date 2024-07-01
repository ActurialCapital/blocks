from functools import wraps
import pandas as pd


def validate_select(transformer_dict: dict):
    """Decorator to validate the selected transformation against available."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if kwargs.get("select", False):
                select = kwargs.get("select")
            else:
                try:
                    select = args[0]
                except:
                    raise TypeError(
                        "Parameter `select` has not been provided. Choose from "
                        f"{list(transformer_dict.keys())}."
                    )
            # Check if select is in transformer_dict keys
            if select not in transformer_dict:
                raise TypeError(
                    "Invalid selection for operation. Choose from "
                    f"{list(transformer_dict.keys())}."
                )
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def register_feature_names(func):
    """Decorator to register pandas feature names."""
    @wraps(func)
    def wrapper(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        return func(self, X, *args, **kwargs)

    return wrapper


def output_pandas_dataframe(func):
    """Decorator to output pandas dataframe."""
    @wraps(func)
    def wrapper(self, X, *args, **kwargs):
        output = func(self, X, *args, **kwargs)
        return pd.DataFrame(output, index=X.index, columns=self.columns_)

    return wrapper
