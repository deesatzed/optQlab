"""OptiQ Lab — local web UI for quantize / fine-tune / dataset / chat.

Public entry point::

    from optiq.lab import create_app
    app = create_app(api_url="http://127.0.0.1:8080")
    app.run(host="127.0.0.1", port=7860)

For end users the friendlier path is ``optiq lab`` (CLI), which boots
both the model-serving API and the Lab UI in one process.

``create_app`` is imported lazily (via module ``__getattr__``) so that
importing a flask-free lab submodule — e.g. ``optiq.lab.mlx_cleanup``,
which ``optiq serve`` uses for RAM cleanup — does not pull in flask. Only
the Lab UI itself needs the ``lab`` extra installed.
"""

__all__ = ["create_app"]


def __getattr__(name):
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
