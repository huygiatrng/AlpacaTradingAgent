"""Run diagnostics on startup, after saving keys, and on demand."""

from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from webui.components.connection_panel import render_results, render_connection_summary
from webui.utils.api_health import check_connections, check_selected_models


def register_connection_callbacks(app):
    @app.callback(
        Output("connection-results", "data"),
        Input("check-connections-btn", "n_clicks"), Input("api-keys-store", "data"),
        Input("llm-provider", "value"), Input("backend-url", "value"),
        running=[(Output("check-connections-btn", "disabled"), True, False)],
    )
    def check_connections_callback(clicks, stored, provider, backend):
        return check_connections(stored, provider, backend)

    @app.callback(Output("connection-status", "children"), Output("connection-summary", "children"), Input("connection-results", "data"))
    def display_connections(report):
        return render_results(report), render_connection_summary(report)

    @app.callback(Output("connection-details", "is_open"), Input("connection-details-toggle", "n_clicks"),
                  State("connection-details", "is_open"), prevent_initial_call=True)
    def toggle_details(clicks, is_open):
        return not is_open if clicks else is_open

    @app.callback(
        Output("model-test-status", "children"), Input("test-models-btn", "n_clicks"),
        State("api-keys-store", "data"), State("llm-provider", "value"), State("backend-url", "value"),
        State("quick-llm", "value"), State("deep-llm", "value"),
        State("quick-llm-custom-model", "value"), State("deep-llm-custom-model", "value"),
        prevent_initial_call=True,
        running=[(Output("test-models-btn", "disabled"), True, False)],
    )
    def test_models(clicks, stored, provider, backend, quick, deep, custom_quick, custom_deep):
        if not clicks:
            raise PreventUpdate
        models = [custom_quick if quick == "custom" else quick, custom_deep if deep == "custom" else deep]
        return render_results(check_selected_models(stored, provider, backend, models))
