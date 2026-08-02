"""Child-process entry point for the API server.

Run as ``python -m optiq.lab.api_runner --model <path> --host 127.0.0.1
--port 8080 [--mtp --mtp-depth 2]``. Installs the same patches the
``optiq serve`` CLI installs (Anthropic /v1/messages, OpenAI
/v1/responses, sk-optiq-* auth, optional MTP speculation), then hands
off to ``mlx_lm.server.main``.

Lives in a separate module so the Lab's subprocess supervisor can spawn
it without importing the full CLI surface (and so a crash here doesn't
take the Lab process down).
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--mtp-depth", type=int, default=2)
    parser.add_argument(
        "--drafter", type=str, default=None,
        help="HF id of a -assistant drafter model (Gemma-4 family). When set, "
             "routes generation through optiq.runtime.spec instead of MTP. "
             "Mutually exclusive with --mtp.",
    )
    parser.add_argument(
        "--adapter", action="append", default=[],
        help="LoRA adapter to mount at startup. Repeat the flag to mount "
             "multiple adapters; requests pick one via the 'adapters' field "
             "in the request body (adapter name = directory basename). One "
             "adapter routes through mlx-lm's classic --adapter-path; two or "
             "more switches to OptiQ's mounted-LoRA path (one base in RAM, "
             "N adapter sidecars, ContextVar-gated per-request switching).",
    )
    parser.add_argument(
        "--clear-cache-threshold-bytes", type=int, default=None,
        help="Clear MLX buffer reuse pool when it exceeds this many bytes. "
             "Default: 10%% of total RAM, floor 1 GB. Mirrors mlx-lm's own "
             "_clear_cache(threshold) pattern from the trainer.",
    )
    parser.add_argument(
        "--stream-experts", choices=["auto", "on", "off"], default="auto",
        help="Stream MoE expert weights from SSD so large MoE quants that "
             "would OOM run at a few GB. auto: only when the model is too big "
             "to fit resident. Default: auto.",
    )
    parser.add_argument("--stream-experts-cache", type=int, default=0)
    parser.add_argument(
        "--context-scale", type=float, default=1.0,
        help="Multiply reported usage token counts by this factor so a "
             "context-aware agent (Claude Code) auto-compacts at the right time "
             "for a smaller-context model. Default: 1.0 (off).",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=8,
        help="Cap requests decoding in parallel (excess queue). Each holds its "
             "own KV cache; mlx-lm's default of 32 can OOM unified memory. "
             "Sets --decode-concurrency (and --prompt-concurrency ~1/4). Default: 8.",
    )
    args, extra = parser.parse_known_args()
    # Inject the Mac-safe concurrency cap into the argv forwarded to mlx-lm,
    # unless the caller passed the underlying flags.
    if getattr(args, "max_concurrent", 0) and args.max_concurrent > 0:
        if not any(a.split("=")[0] == "--decode-concurrency" for a in extra):
            extra += ["--decode-concurrency", str(int(args.max_concurrent))]
        if not any(a.split("=")[0] == "--prompt-concurrency" for a in extra):
            extra += ["--prompt-concurrency", str(max(1, int(args.max_concurrent) // 4))]

    # Install patches BEFORE mlx_lm.server.main() takes over.
    from optiq.anthropic_server import install_anthropic_endpoint
    from optiq.responses_server import install_responses_endpoint
    from optiq.auth import install_auth
    from optiq.lab.mlx_cleanup import (
        default_prompt_cache_bytes,
        default_threshold_bytes,
        inject_prompt_cache_bytes,
        install_server_cleanup,
    )

    install_anthropic_endpoint()
    install_responses_endpoint()
    install_auth()

    threshold = args.clear_cache_threshold_bytes or default_threshold_bytes()
    install_server_cleanup(threshold)
    print(f"[optiq.lab.api_runner] mlx cleanup threshold: "
          f"{threshold / 1024**3:.2f} GB", flush=True)

    # Structured / JSON-constrained output (OpenAI response_format +
    # vLLM-style guided_json/regex/choice), so models served through the Lab
    # support the same constrained decoding as `optiq serve`. Always on;
    # activates only when a request carries a spec. Torch-free
    # (lm-format-enforcer), so it does not pull PyTorch into the MLX runtime.
    import mlx_lm.server as _srv_mod
    from optiq.runtime.structured import install as install_structured
    install_structured(_srv_mod)
    print("[optiq.lab.api_runner] structured output ready "
          "(response_format / guided_json)", flush=True)

    # Tool-call healing at the server layer (recovers malformed tool calls into
    # OpenAI tool_calls), so API clients of the Lab's server get healed calls,
    # not only the Lab chat orchestrator. Non-streaming completions.
    from optiq.runtime.tool_healing import install as install_tool_healing
    install_tool_healing(_srv_mod)
    print("[optiq.lab.api_runner] tool-call healing ready", flush=True)

    # Single-model: the Lab serves one model per process (it switches by
    # restarting this server), so the request 'model' field is a label — any
    # value serves --model rather than 404ing on a mismatch.
    from optiq.serve import install_model_field_policy
    install_model_field_policy(allow_switch=False)

    # Model variants: expose thinking on/off as API-visible model ids
    # (<model>:think / <model>:no-think), so API clients of the Lab's server can
    # pick a variant by name without the non-standard chat_template_kwargs field.
    from optiq.runtime.variants import install as install_variants
    install_variants(_srv_mod)
    print("[optiq.lab.api_runner] model variants ready "
          "(append :think / :no-think to the model id)", flush=True)

    if getattr(args, "context_scale", 1.0) and float(args.context_scale) != 1.0:
        from optiq.runtime.context_scale import install as install_context_scale
        install_context_scale(_srv_mod, args.context_scale)
        print(f"[optiq.lab.api_runner] context scaling on (usage x{float(args.context_scale):g})",
              flush=True)

    if args.mtp and args.drafter:
        raise SystemExit("--mtp and --drafter are mutually exclusive")
    if args.mtp:
        from optiq.serve import install_mtp_speculation
        install_mtp_speculation(model_path=args.model, depth=args.mtp_depth)
    elif args.drafter:
        from optiq.serve import install_assistant_drafter
        install_assistant_drafter(target_model_path=args.model,
                                  drafter_id=args.drafter)

    # SSD expert streaming for large MoE quants (auto unless overridden).
    if args.stream_experts != "off":
        from optiq.serve import install_streaming_experts
        install_streaming_experts(mode=args.stream_experts,
                                  cache_experts=args.stream_experts_cache,
                                  model_path=args.model)
        print(f"[optiq.lab.api_runner] SSD expert streaming: "
              f"{args.stream_experts}", flush=True)

    # Cap mlx-lm's LRUPromptCache. See default_prompt_cache_bytes docstring.
    pc_bytes = default_prompt_cache_bytes()
    extra_before = list(extra)
    extra = inject_prompt_cache_bytes(extra, pc_bytes)
    if extra != extra_before:
        print(f"[optiq.lab.api_runner] mlx prompt-cache cap: "
              f"{pc_bytes / 1024**3:.2f} GB", flush=True)

    # Apply the model's recommended sampler from generation_config.json
    # unless the supervisor already passed equivalent flags. Critical
    # for MTP — pure temp without top_p/top_k tanks draft acceptance.
    from optiq.runtime.gen_config import read_recommended_sampling, merge_into_argv
    recommended = read_recommended_sampling(args.model)
    if recommended:
        extra = merge_into_argv(
            extra, recommended,
            prefix_log="[optiq.lab.api_runner] applying model-recommended sampler:",
        )

    # Wire LoRA adapter(s). One adapter → mlx-lm's classic single-adapter
    # boot via --adapter-path. Two or more → OptiQ mounted-LoRA mode (one
    # base in RAM, ContextVar-gated per-request switching). The supervisor
    # passes paths already resolved to local dirs.
    from pathlib import Path as _Path
    adapter_specs: list[tuple[str, str]] = []
    if args.adapter:
        for ad_path in args.adapter:
            ad_dir = _Path(ad_path)
            if not ad_dir.exists():
                print(f"[optiq.lab.api_runner] adapter path missing: {ad_path}",
                      file=sys.stderr, flush=True)
                continue
            name = ad_dir.name
            adapter_specs.append((name, str(ad_dir.resolve())))
            print(f"[optiq.lab.api_runner] adapter: {ad_path} (name={name})",
                  flush=True)

    if len(adapter_specs) == 1:
        single_dir = adapter_specs[0][1]
        if "--adapter-path" not in extra:
            extra += ["--adapter-path", single_dir]
    elif len(adapter_specs) > 1:
        from optiq.serve import install_multi_adapter
        install_multi_adapter(adapter_specs)
        print(f"[optiq.lab.api_runner] mounted-LoRA mode: {len(adapter_specs)} "
              f"adapters, GET /v1/adapters lists them, per-request switch via "
              f"the 'adapters' body field.", flush=True)

    # Enable image+text serving when the model ships an optiq_vision sidecar.
    # Installed last so it wraps whatever stream_generate the speculation
    # paths set up; transparent for text-only requests.
    try:
        from optiq.sidecar_layout import local_model_dir
        from optiq.vlm import has_vision_sidecar
        from optiq.serve import install_vision_serving

        install_vision_serving(args.model)
        # args.model is usually a Hub repo id, not a directory. See cli.py.
        _vis_dir = local_model_dir(args.model)
        if _vis_dir is not None and has_vision_sidecar(_vis_dir):
            print("[optiq.lab.api_runner] vision serving enabled "
                  "(image+text via optiq_vision sidecar)", flush=True)
    except Exception as _exc:
        print(f"[optiq.lab.api_runner] vision serving not installed: {_exc}",
              file=sys.stderr, flush=True)

    # Hand argv to mlx_lm.server's argparse. Forward any extra --foo passed
    # by the supervisor (e.g. --temp, --top-p) verbatim.
    sys.argv = [
        "mlx_lm.server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
    ] + extra

    from mlx_lm.server import main as mlx_main
    mlx_main()


if __name__ == "__main__":
    main()
