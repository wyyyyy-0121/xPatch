    def fit(
        self,
        inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]]
        | Sequence[PreparedInput],
        prediction_length: int,
        validation_inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]]
        | Sequence[PreparedInput]
        | None = None,
        finetune_mode: Literal["full", "lora"] = "full",
        lora_config: "LoraConfig | dict | None" = None,
        context_length: int | None = None,
        learning_rate: float = 1e-6,
        num_steps: int = 1000,
        batch_size: int = 256,
        output_dir: Path | str | None = None,
        min_past: int | None = None,
        finetuned_ckpt_name: str = "finetuned-ckpt",
        callbacks: list["TrainerCallback"] | None = None,
        remove_printer_callback: bool = False,
        disable_data_parallel: bool = True,
        **extra_trainer_kwargs,
    ) -> "Chronos2Pipeline":
        """
        Fine-tune a copy of the current Chronos-2 model on the given inputs and return a new pipeline.

        Parameters
        ----------
        inputs
            The time series on which the model will be fine-tuned. The allowed formats of inputs are the same as `Chronos2Pipeline.predict()`.
            When fine-tuning with covariates, we recommend building the inputs with the `chronos.chronos2.preprocess` helpers, which handle
            covariates (including which are known into the future) and categorical encoding for you: `from_data_frame` (long-format DataFrame)
            and `from_list_of_dicts` (list of per-series dicts).
        prediction_length
            The prediction horizon for which the model will be fine-tuned
        validation_inputs
            The time series used for validation and model selection. The format of `validation_inputs` is exactly the same as `inputs`, by default None which
            means that no validation is performed. Note that enabling validation may slow down fine-tuning for large datasets.
        finetune_mode
            One of "full" (performs full fine-tuning) or "lora" (performs Low Rank Adaptation (LoRA) fine-tuning), by default "full"
        lora_config
            The configuration to use for LoRA fine-tuning when finetune_mode="lora". Can be a `LoraConfig` object or a dict which is used to initialize `LoraConfig`.
            When unspecified and finetune_mode="lora", a default configuration is used
        context_length
            The maximum context length used during fine-tuning, by default set to the model's default context length
        learning_rate
            The learning rate for the optimizer, by default 1e-6
            When finetune_mode="lora", we recommend using a higher value of the learning rate, such as 1e-5
        num_steps
            The number of steps to fine-tune for, by default 1000
        batch_size
            The batch size used for fine-tuning. Note that the batch size here means the number of time series, including target(s) and covariates,
            which are input into the model. If your data has multiple target and/or covariates, the effective number of time series tasks in a batch
            will be lower than this value, by default 256
        output_dir
            The directory in which outputs from the `Trainer` will be saved, by default set to `chronos-2-finetuned/{%Y-%m-%d_%H-%M-%S}`
        min_past
            The minimum number of time steps the context must have during fine-tuning. All time series shorter than `min_past + prediction_length`
            are filtered out, by default set equal to prediction_length
        finetuned_ckpt_name
            The name of the directory inside `output_dir` in which the final fine-tuned checkpoint will be saved, by default "finetuned-ckpt"
        callbacks
            A list of `TrainerCallback`s which will be forwarded to the HuggingFace `Trainer`
        remove_printer_callback
            If True, all instances of `PrinterCallback` are removed from callbacks
        disable_data_parallel
            If True, ensures that DataParallel is disabled and training happens on a single GPU
        **extra_trainer_kwargs
            Extra kwargs are directly forwarded to `TrainingArguments`

        Returns
        -------
        A new `Chronos2Pipeline` with the fine-tuned model
        """

        import torch.cuda
        from transformers.trainer_callback import PrinterCallback
        from transformers.training_args import TrainingArguments

        if finetune_mode == "lora":
            if is_peft_available():
                from peft import LoraConfig, get_peft_model
            else:
                warnings.warn(
                    "`peft` is required for `finetune_mode='lora'`. Please install it with `pip install peft`. Falling back to `finetune_mode='full'`."
                )
                finetune_mode = "full"
                lora_config = None

        from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback

        assert finetune_mode in ["full", "lora"], f"finetune_mode must be one of ['full', 'lora'], got {finetune_mode}"

        if finetune_mode == "full" and lora_config is not None:
            raise ValueError(
                "lora_config should not be specified when `finetune_mode='full'`. To enable LoRA, set `finetune_mode='lora'`."
            )

        # Create a copy of the model to avoid modifying the original
        config = deepcopy(self.model.config)
        model = Chronos2Model(config).to(self.model.device)  # type: ignore
        model.load_state_dict(self.model.state_dict())

        if finetune_mode == "lora":
            if lora_config is None:
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=[
                        "self_attention.q",
                        "self_attention.v",
                        "self_attention.k",
                        "self_attention.o",
                        "output_patch_embedding.output_layer",
                    ],
                )
            elif isinstance(lora_config, dict):
                lora_config = LoraConfig(**lora_config)
            else:
                assert isinstance(lora_config, LoraConfig), (
                    f"lora_config must be an instance of LoraConfig or a dict, got {type(lora_config)}"
                )

            model = get_peft_model(model, lora_config)
            n_trainable_params, n_params = model.get_nb_trainable_parameters()
            logger.info(
                f"Using LoRA. Number of trainable parameters: {n_trainable_params}, total parameters: {n_params}."
            )

        if context_length is None:
            context_length = self.model_context_length

        if min_past is None:
            min_past = prediction_length

        train_dataset = Chronos2Dataset(
            inputs=inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=self.model_output_patch_size,
            min_past=min_past,
            mode=DatasetMode.TRAIN,
        )

        if output_dir is None:
            output_dir = Path("chronos-2-finetuned") / time.strftime("%Y-%m-%d_%H-%M-%S")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        assert isinstance(output_dir, Path)

        use_cpu = str(self.model.device) == "cpu"
        has_sm80 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

        # warn user if a cuda device is available and CPU fine-tuning is used
        if use_cpu and torch.cuda.is_available():
            warnings.warn(
                "The model is being fine-tuned on the CPU, but a CUDA device is available. "
                "We recommend using the GPU for faster fine-tuning.",
                category=UserWarning,
                stacklevel=2,
            )

        training_kwargs: dict = dict(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            **({"warmup_steps": 0} if _TRANSFORMERS_V5 else {"warmup_ratio": 0.0}),
            optim="adamw_torch_fused",
            logging_strategy="steps",
            logging_steps=100,
            disable_tqdm=False,
            report_to="none",
            max_steps=num_steps,
            gradient_accumulation_steps=1,
            dataloader_num_workers=0,
            tf32=has_sm80 and not use_cpu,
            bf16=has_sm80 and not use_cpu,
            save_only_model=True,
            prediction_loss_only=True,
            save_total_limit=1,
            save_strategy="no",
            save_steps=None,
            eval_strategy="no",
            eval_steps=None,
            load_best_model_at_end=False,
            metric_for_best_model=None,
            use_cpu=use_cpu,
        )

        eval_dataset = None
        callbacks = callbacks or []
        if validation_inputs is not None:
            eval_dataset = Chronos2Dataset(
                inputs=validation_inputs,
                context_length=context_length,
                prediction_length=prediction_length,
                batch_size=batch_size,
                output_patch_size=self.model_output_patch_size,
                mode=DatasetMode.VALIDATION,
            )

            # set validation parameters
            training_kwargs["save_strategy"] = "steps"
            training_kwargs["save_steps"] = 100
            training_kwargs["eval_strategy"] = "steps"
            training_kwargs["eval_steps"] = 100
            training_kwargs["load_best_model_at_end"] = True
            training_kwargs["metric_for_best_model"] = "eval_loss"
            training_kwargs["label_names"] = ["future_target"]

            # add callback to ensure that the final model is evaluated
            callbacks.append(EvaluateAndSaveFinalStepCallback())

        training_kwargs.update(extra_trainer_kwargs)

        if training_kwargs["tf32"]:
            # setting tf32=True changes these global properties, we copy them here so that
            # we can restore them after fine-tuning

            # NOTE: The old way of allowing TF32 computations was deprecated in 2.9 and torch
            # doesn't allow mixing the old and new ways.
            is_torch_geq_2p9 = version.parse(torch.__version__) >= version.parse("2.9.0")

            if is_torch_geq_2p9:
                matmul_precision = torch.backends.cuda.matmul.fp32_precision
                cudnn_precision = torch.backends.cudnn.fp32_precision
            else:
                matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
                cudnn_tf32 = torch.backends.cudnn.allow_tf32

        training_args = TrainingArguments(**training_kwargs)

        if disable_data_parallel and not use_cpu:
            # This is a hack to disable the default `transformers` behavior of using DataParallel
            training_args._n_gpu = 1
            assert training_args.n_gpu == 1  # Ensure that the hack worked

        trainer = Chronos2Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        if remove_printer_callback:
            trainer.pop_callback(PrinterCallback)

        trainer.train()

        # update context_length and max_output_patches, if the model was fine-tuned with larger values
        model.chronos_config.context_length = max(model.chronos_config.context_length, context_length)
        model.chronos_config.max_output_patches = max(
            model.chronos_config.max_output_patches, math.ceil(prediction_length / self.model_output_patch_size)
        )
        # update chronos_config in model's config, so it is saved correctly
        model.config.chronos_config = model.chronos_config.__dict__

        # Create a new pipeline with the fine-tuned model
        finetuned_pipeline = Chronos2Pipeline(model=model)

        # Save fine-tuned model
        finetuned_path = output_dir / finetuned_ckpt_name
        finetuned_pipeline.save_pretrained(finetuned_path)
        logger.info(f"Finetuned model saved to {finetuned_path}")

        if training_kwargs["tf32"]:
            # restore tf32 settings

            if is_torch_geq_2p9:
                torch.backends.cuda.matmul.fp32_precision = matmul_precision
                torch.backends.cudnn.fp32_precision = cudnn_precision
            else:
                torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
                torch.backends.cudnn.allow_tf32 = cudnn_tf32

        return finetuned_pipeline
