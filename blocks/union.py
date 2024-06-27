import numpy as np
import pandas as pd

from sklearn.pipeline import FeatureUnion, _transform_one, _name_estimators
from sklearn.utils import Bunch
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.metadata_routing import (
    _raise_for_params,
    _routing_enabled,
    process_routing
)


class BlockUnion(FeatureUnion):
    def __init__(
        self,
        transformer_list,
        *,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
        output_pandas: bool = True
    ):
        self.output_pandas = output_pandas
        super().__init__(
            transformer_list,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out,
        )

    def transform(self, X, **params) -> np.ndarray | pd.DataFrame | pd.Series:
        """
        Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        **params : dict, default=None
            Parameters routed to the `transform` method of the sub-transformers via the
            metadata routing API. See :ref:`Metadata Routing User Guide
            <metadata_routing>` for more details.

        Returns
        -------
        X_t : array-like or sparse matrix of shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
            If output_pandas is set to True, it returns a pandas object.
        """
        _raise_for_params(params, self, "transform")

        if _routing_enabled():
            routed_params = process_routing(self, "transform", **params)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = Bunch()
            for name, _ in self.transformer_list:
                routed_params[name] = Bunch(transform={})

        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None,
                                    weight, params=routed_params[name])
            for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        if self.output_pandas:
            return pd.concat(Xs, axis=1)

        return self._hstack(Xs)


def make_block_union(*transformers, n_jobs=None, verbose=False):
    """Construct a :class:`FeatureUnion` from the given transformers.

    This is a shorthand for the :class:`FeatureUnion` constructor; it does not
    require, and does not permit, naming the transformers. Instead, they will
    be given names automatically based on their types. It also does not allow
    weighting.

    Parameters
    ----------
    *transformers : list of estimators
        One or more estimators.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Returns
    -------
    f : FeatureUnion
        A :class:`FeatureUnion` object for concatenating the results of multiple
        transformer objects.

    See Also
    --------
    FeatureUnion : Class for concatenating the results of multiple transformer
        objects.

    Examples
    --------
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> from sklearn.pipeline import make_union
    >>> make_union(PCA(), TruncatedSVD())
     FeatureUnion(transformer_list=[('pca', PCA()),
                                   ('truncatedsvd', TruncatedSVD())])
    """
    return BlockUnion(_name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)
