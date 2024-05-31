from typing import Any, List, Tuple, Callable, Optional
import logging
import time
import pandas as pd

from sklearn.utils.validation import check_memory
from sklearn.pipeline import _name_estimators
from sklearn.utils.metadata_routing import _raise_for_params, process_routing
from sklearn.utils.metaestimators import available_if

from imblearn.pipeline import Pipeline


logging.basicConfig(
    format=("[%(funcName)s:%(lineno)d] - %(message)s"),
    level=logging.INFO
)


def custom_log_callback(output, execution_time, **kwargs):
    """
    The default log callback which logs the step name, shape of the output,
    the execution time of the step and the named steps.

    !!! info

        If you write your custom callback function the input is:

        | Parameter        | Type             | Description                    |
        | ---------------- | ---------------- | ------------------------------ |
        | `func`           | Callable[..., T] | The function to be wrapped     |
        | `input_args`     | tuple            | The input arguments            |
        | `input_kwargs`   | dict             | The input key-word arguments   |
        | `output`         | T                | The output of the function     |
        | `execution_time` | float            | The execution time of the step |
        | `named_steps`    | dict             | The named steps of the pipeline|

    Parameters
    ----------
    output : tuple[np.ndarray | pd.DataFrame, estimator | transformer]
        The output of the step and a step in the pipeline.
    execution_time : float
        The execution time of the step.

    Examples
    --------
    Create Data
    ```pycon
    >>> n_samples, n_features = 3, 5
    >>> X = np.zeros((n_samples, n_features))
    >>> y = np.arange(n_samples)
    ```
    Initializing `steps`
    ```pycon
    >>> steps = [
    ...     ("add_1", Adder(value=1)),
    ...     ("add_10", Adder(value=10)),
    ...     ("add_100", Adder(value=100)),
    ...     ("add_1000", Adder(value=1000)),
    ... ]
    ```
    Runing the pipeline
    ```pycon
    >>> pipe = BlockPipeline(steps, log_callback=custom_log_callback)
    >>> pipe.fit(X, y=y)
    [custom_log_callback:48] - [add_1][Adder(value=1)] shape=(3, 5) time=0s
    [custom_log_callback:48] - [add_10][Adder(value=10)] shape=(3, 5) time=0s
    [custom_log_callback:48] - [add_100][Adder(value=100)] shape=(3, 5) time=0s
    ```

    """
    logger = logging.getLogger(__name__)
    step_result, step = output
    key = next(
        (k for k, v in kwargs['named_steps'].items() if v == step),
        None
    )
    logger.info(
        f"[{key}][{step}] shape={step_result.shape} " f"time={int(execution_time)}s"
    )


def _log_wrapper(
    log_callback: Any = custom_log_callback,
    named_steps: dict = None
):
    """
    Function wrapper to log information after the function is called, about the 
    step_name, function, input args, input kwargs, output and the execution 
    time.

    Parameters
    ----------
    log_callback : Callable, default=custom_log_callback
        The log callback which is called after `func` is called. Note, this 
        function should expect the same arguments as the default.

    Returns
    -------
    Callable
        The function wrapped with a log callback.
    """

    def _(func):
        def _(*args, **kwargs):
            start_time = time.time()
            output = func(*args, **kwargs)
            execution_time = time.time() - start_time
            log_callback(
                func=func,
                input_args=args,
                input_kwargs=kwargs,
                output=output,
                execution_time=execution_time,
                named_steps=named_steps
            )
            return output

        return _

    return _


def _cache_with_function_log_statement(
    log_callback=custom_log_callback,
    named_steps=None
):
    """
    Wraps the `func` with `_log_wrapper` before passing it to `_cache`.

    Parameters
    ----------
    log_callback : Callable, default=default_log_callback
        The log callback function.

    Returns
    -------
    Callable
        The function wrapped with a log callback.
    """

    def _(self, func=None, *args, **kwargs):
        if callable(func):
            func = _log_wrapper(log_callback, named_steps=named_steps)(func)
        return self._cache(func=func, *args, **kwargs)

    return _


class BlockPipeline(Pipeline):
    """    
    A custom pipeline that extends both `sklearn` and `sklego` pipelines.

    Parameters
    ----------
    steps : List[Tuple[str, Any]]
        A list of (name, transform) tuples (implementing fit/transform) that 
        define the pipeline steps. The final object is an estimator.
    memory : Memory, optioanl
        Used to cache the fitted transformers of the pipeline. By default, no 
        caching is performed. If a string is given, it is the path to the 
        caching directory. Defaults to False.
    verbose : bool, optional
        If True, the time elapsed while fitting each step will be printed as 
        it is completed. Defaults to False.
    log_callback : Callable | None, optional
        The callback function that logs information in between each 
        intermediate step. If set to `"custom"`, `custom_log_callback` is 
        used. Defaults to None.
    record : str, optional
        Named step to record data. The name step would be accessible as an
        attribute with test-set transformed data. Note that spaces " " are 
        removed and dashes "-"  and replaced with underscores "_". Defaults to 
        None.
    
    Examples
    --------

    !!! tip

        We also get a `pandas.DataFrame` object for both X and y, as the 
        internal `sklearn` configuration is set to output "pandas" for 
        transformers. To modify this configuration, use sklearn `set_config`:
        ```pycon
        >>> from sklearn import set_config
        >>> set_config(transform_output=None)
        ```
        More information could be found in the `sklearn-set-config` 
        documentation.

    Get a public dataset from 'sklearn'. We've turned the array into a 
    dataframe so that we can apply the `ColumnSelector`
    ```pycon
    >>> from sklearn.datasets import make_regression
    >>> from sklearn import set_config
    >>> import pandas as pd
    >>>
    >>> set_config(transform_output=None)
    >>>
    >>> X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    >>> df = pd.DataFrame(X)
    ```

    !!! info       

        Here is an example from `sklego`, where `Pipeline` have been replaced 
        by `BlockPipeline`.

    Construct a pipeline with `record_named_step` set to "my_models"
    ```pycon hl_lines="31"
    >>> # sklearn
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import GridSearchCV
    >>>
    >>> from blocks import BlockPipeline, custom_log_callback
    >>>     
    >>> 
    >>> pipe = BlockPipeline([
    ...     ("scaler", StandardScaler()),
    ...     ("regression", LinearRegression())
    ... ],
    ...     record='scaler',
    ...     log_callback=custom_log_callback
    ... )
    >>> grid = GridSearchCV(estimator=pipe, param_grid={}, cv=3)
    >>> grid.fit(df, y)
    >>> predicted = grid.predict(df)
    # [custom_log_callback:78] - [scaler][StandardScaler()] shape=(666, 10) time=0s
    # [custom_log_callback:78] - [scaler][StandardScaler()] shape=(667, 10) time=0s
    # [custom_log_callback:78] - [scaler][StandardScaler()] shape=(667, 10) time=0s
    # [custom_log_callback:78] - [scaler][StandardScaler()] shape=(1000, 10) time=0s
    ```

    1.  "my_models" named step data will be recorded in a "my_models" attribute.

    Visualise a diagram
    ```pycon
    >>> from sklearn import set_config
    >>> set_config(display="diagram")
    >>> grid
    ```

    --8<-- "docs/scripts/grid.html"

    "my_models" step data are recorded
    ```pycon
    >>> grid.my_models
             path1       path2
    0     9.471704   48.058387
    1   -16.387420  -97.500800
    2    21.193080   -5.611140
    3    10.029822   69.331895
    4    -3.711766  247.354944
    ..         ...         ...
    995 -23.566628  -38.467267
    996  20.371979  212.367203
    997  16.958323  -78.536265
    998  13.431366   36.413145
    999  11.209475  226.971373

    [1000 rows x 2 columns]
    ```

    """

    def __init__(
        self,
        steps: List[Tuple[str, Any]],
        memory: Optional[str | Callable] = None,
        verbose: Optional[bool] = False,
        *,
        log_callback: Optional[str | Callable] = None,
        record: Optional[str] = None,
    ):
        self.log_callback = log_callback
        self._record = record
        self.name_record = record if record is not None else None

        super().__init__(steps=steps, memory=memory, verbose=verbose)


    @property
    def memory(self):
        # When no log callback function is given, change nothing.
        # Or, if the memory cache was changed, set it back to its original.
        if self._log_callback is None:
            if hasattr(self._memory, "_cache"):
                self._memory.cache = self._memory._cache
            return self._memory

        self._memory = check_memory(self._memory)

        # Overwrite cache function of memory such that it logs the
        # output when the function is called
        if not hasattr(self._memory, "_cache"):
            self._memory._cache = self._memory.cache

        self._memory.cache = (
            _cache_with_function_log_statement(
                self._log_callback,
                self.named_steps  # << Modified from sklego DebugPipeline
            )
            .__get__(
                self._memory,
                self._memory.__class__
            )
        )
        return self._memory

    @memory.setter
    def memory(self, memory):
        self._memory = memory

    @property
    def log_callback(self):
        return self._log_callback

    @log_callback.setter
    def log_callback(self, func):
        self._log_callback = func
        if self._log_callback == "custom":
            self._log_callback = custom_log_callback

    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, step_args: Tuple[str, pd.DataFrame]):
        name_step, step_result = step_args
        if name_step == self.name_record:
            self._record = step_result

    @available_if(Pipeline._can_transform)
    def transform(self, X, **params):
        _raise_for_params(params, self, "transform")
        # not branching here since params is only available if
        # enable_metadata_routing=True
        routed_params = process_routing(self, "transform", **params)
        Xt = X
        for _, name, transform in self._iter():
            Xt = transform.transform(Xt, **routed_params[name].transform)
            self.record = (name, Xt)

        return Xt


def make_block_pipeline(*steps, **kwargs) -> BlockPipeline:
    """
    Construct a `BlockPipeline` from the given estimators.

    This is a shorthand for the `BlockPipeline` constructor; it does not 
    require, and does not permit, naming the estimators. Instead, their names
    will be set to the lowercase of their types automatically.

    Parameters
    ----------
    *steps : list
        List of transformer or estimators to be included in the pipeline.
    **kwargs : dict
        Additional keyword arguments passed to the `BlockPipeline` constructor.
        Possible arguments are `memory`, `verbose`, `log_callback` and
        `named_step`:

        * `memory` : str | object with the joblib.Memory interface
        Used to cache the fitted transformers of the pipeline. The last 
        step will never be cached, even if it is a transformer. By default, 
        no caching is performed. If a string is given, it is the path to 
        the caching directory. Enabling caching triggers a clone of the 
        transformers before fitting. Therefore, the transformer instance 
        given to the pipeline cannot be inspected directly. Use the 
        attribute `named_steps` or `steps` to inspect estimators within 
        the pipeline. Caching the transformers is advantageous when fitting 
        is time consuming. It defaults to None.
        * `verbose` : bool
        If True, the time elapsed while fitting each step will be printed 
        as it is completed. Defaults to False
        * `log_callback` : str | Callable | None
        The callback function that logs information in between each 
        intermediate step. If set to `"custom"`, `custom_log_callback` is 
        used. Defaults to None.
        * `record_from` : str
        Records named step data passed between each step of the pipeline at
        transform. The recorded data would become an attribute with named step 
        as name. If "all" is passed, it records all named step data. Default to 
        None.

    Returns
    -------
    BlockPipeline
        Instance with given steps, `memory`, `verbose`, `log_callback` and 
        `record_from`.

    Examples
    --------
    Simple pipeline
    ```pycon hl_lines="8"
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>> 
    >>> pipe = make_block_pipeline(
    ...     StandardScaler(), 
    ...     PCA(n_components=3),
    ...     GaussianNB(priors=None), 
    ... )
    >>> pipe
    BlockPipeline(steps=[
        ('standardscaler', StandardScaler()), 
        ('pca', PCA(n_components=3)), 
        ('gaussiannb', GaussianNB())
    ])
    ```

    Query recorded named steps data
    ```pycon
    >>> pipe.fit(X, y)
    >>> pipe.standardscaler
               x0        x1        x2  ...        x7        x8        x9
    0    1.299919  0.538339 -0.966915  ... -2.135893 -0.147076 -0.495194
    1   -0.103911  0.543639 -1.471462  ... -1.286657  1.274698 -2.753808
    2   -0.035639 -0.238223 -1.030765  ...  1.437366  0.774441 -0.301539
    3   -1.103966  0.036170  0.580354  ... -0.092523  1.367957 -0.577798
    4    0.163617 -1.237277  0.410185  ...  0.396819  2.908771  0.583587
    ..        ...       ...       ...  ...       ...       ...       ...
    995 -0.325861 -2.057915  0.149747  ...  0.843667  1.322642 -0.998840
    996 -0.384003 -0.221844 -0.330495  ... -0.778845  1.513707  1.103126
    997  0.223566  0.376658  0.160554  ... -0.798789 -3.662747  1.445524
    998  1.730267 -0.771208  0.288065  ... -1.291845  1.329504 -0.232815
    999  0.193020  0.375951  0.682429  ... -1.621108  0.723347  2.726158

    [1000 rows x 10 columns]
    >>> pipe.pca
             pca0      pca1      pca2
    0   -0.829442 -0.205333 -0.342416
    1    0.400897  0.952266  0.179371
    2    0.096574 -0.591751  0.657931
    3    0.049538 -1.178944 -1.154298
    4   -1.124832 -2.036789 -1.494786
    ..        ...       ...       ...
    995  2.396324  0.799270 -1.242327
    996 -1.672366 -1.966873 -1.534830
    997  0.735779  1.625347 -0.462980
    998  0.466143 -1.183840  0.332553
    999 -1.156686 -0.943222 -2.042877

    [1000 rows x 3 columns]
    ```

    With `make_union`
    ```pycon
    >>> from sklearn.pipeline import make_union
    >>> pipe = make_block_pipeline(
    ...     make_union( 
    ...         make_block_pipeline(
    ...             ColumnSelector([0, 1, 2, 3, 4]),
    ...             PCA(n_components=3),
    ...             EstimatorTransformer(LinearRegression()),
    ...         ),
    ...         make_block_pipeline(
    ...             ColumnSelector([5, 6, 7, 8, 9]),
    ...             PCA(n_components=3),
    ...             EstimatorTransformer(LinearRegression())    
    ...         ),            
    ...     ),
    ...     ProbWeightRegression(),
    ... )
    >>> pipe.fit(X, y)
    BlockPipeline(steps=[
        ('featureunion', FeatureUnion(transformer_list=[
            ('blockpipeline-1', BlockPipeline(steps=[
                ('columnselector', ColumnSelector(columns=[0, 1, 2, 3, 4])), 
                ('pca', PCA(n_components=3)), 
                ('pred-1', EstimatorTransformer(estimator=LinearRegression()))
            ])),
            ('blockpipeline-2', BlockPipeline(steps=[
                ('columnselector', ColumnSelector(columns=[5, 6, 7, 8, 9])),
                ('pca', PCA(n_components=3)), 
                ('pred-2', EstimatorTransformer(estimator=LinearRegression()))
            ]))
        ])), 
        ('probweightregression', ProbWeightRegression())
    ])
    ```

    """
    memory = kwargs.pop("memory", None)
    verbose = kwargs.pop("verbose", False)
    log_callback = kwargs.pop("log_callback", None)
    record_from = kwargs.pop("record_from", None)

    if kwargs:
        raise TypeError(
            'Unknown keyword arguments: "{}"'.format(list(kwargs.keys())[0])
        )

    return BlockPipeline(
        _name_estimators(steps),
        memory=memory,
        verbose=verbose,
        log_callback=log_callback,
        record_from=record_from,
    )
