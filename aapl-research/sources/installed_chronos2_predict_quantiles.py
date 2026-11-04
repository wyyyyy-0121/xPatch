    def predict_quantiles(  # type: ignore[override]
        self,
        inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray]]]
        | Sequence[PreparedInput],
        prediction_length: int | None = None,
        quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        **predict_kwargs,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Refer to ``Chronos2Pipeline.predict`` for shared parameters.

        Additional parameters
        ---------------------
        quantile_levels
            Quantile levels to compute, by default [0.1, 0.2, ..., 0.9]

        Returns
        -------
        quantiles
            A list of torch tensors containing quantile forecasts. Each element of the list has shape (n_variates, prediction_length, len(quantile_levels))
            and the number of elements are equal to the number of target time series (univariate or multivariate) in the `inputs`.
        mean
            A list of torch tensors containing containing mean (point) forecasts. Each element of the list has shape (n_variates, prediction_length)
            and the number of elements are equal to the number of target time series (univariate or multivariate) in the `inputs`.
        """
        training_quantile_levels = self.quantiles

        predictions: list[torch.Tensor] = self.predict(inputs, prediction_length=prediction_length, **predict_kwargs)

        # Swap quantile and time axes for each prediction
        predictions = [rearrange(pred, "... q h -> ... h q") for pred in predictions]

        if set(quantile_levels).issubset(training_quantile_levels):
            # no need to perform intra/extrapolation
            quantile_indices = [training_quantile_levels.index(q) for q in quantile_levels]
            quantiles = [pred[..., quantile_indices] for pred in predictions]
        else:
            # we interpolate quantiles if quantiles that Chronos-2 was trained on were not provided
            if min(quantile_levels) < min(training_quantile_levels) or max(quantile_levels) > max(
                training_quantile_levels
            ):
                logger.warning(
                    f"\tQuantiles to be predicted ({quantile_levels}) are not within the range of "
                    f"quantiles that Chronos-2 was trained on ({training_quantile_levels}). "
                    "Quantile predictions will be set to the minimum/maximum levels at which Chronos-2 "
                    "was trained on. This may significantly affect the quality of the predictions."
                )

            quantiles = [
                interpolate_quantiles(quantile_levels, training_quantile_levels, pred) for pred in predictions
            ]

        # NOTE: the median is returned as the mean here
        mean = [pred[..., training_quantile_levels.index(0.5)] for pred in predictions]

        return quantiles, mean
