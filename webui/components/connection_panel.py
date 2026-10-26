"""A short first-run guide and actionable connection results."""

from dash import dcc, html
import dash_bootstrap_components as dbc


def create_connection_panel():
    return dbc.Card(dbc.CardBody([
        html.Div([
            html.H6("Start here", className="compact-panel-title"),
            html.Span("Checking connections…", id="connection-summary", className="compact-panel-summary", role="status"),
            html.Div([
                dbc.Button("API setup", id="setup-api-btn", color="primary", size="sm"),
                dbc.Button("Check", id="check-connections-btn", color="outline-info", size="sm", title="Check API connections"),
                dbc.Button("Details", id="connection-details-toggle", color="outline-secondary", size="sm"),
            ], className="compact-panel-actions"),
        ], className="compact-panel-toolbar"),
        dcc.Store(id="connection-results"),
        dbc.Collapse([
            html.P("Choose an AI provider under LLM Models, add its key, then run an analysis. Other data sources are optional unless your workflow uses them.", className="small text-muted mb-2"),
            dcc.Loading(html.Div(id="connection-status"), type="circle"),
            html.Details([
                html.Summary("Test the selected AI models"),
                html.P("Connection checks do not generate AI text. This optional test sends one short request to each selected model and may incur a small API charge.", className="small mt-2"),
                dbc.Button("Test selected models", id="test-models-btn", color="outline-secondary", size="sm"),
                dcc.Loading(html.Div(id="model-test-status", className="mt-2"), type="circle"),
            ], className="mt-2"),
        ], id="connection-details", is_open=False, className="compact-panel-details"),
    ]), className="mb-2 connection-panel compact-panel")


def render_connection_summary(report):
    results = (report or {}).get("results", [])
    if not results:
        return "Choose a provider under LLM Models to get started."
    issues = [item for item in results if item["status"] not in {"connected", "optional"}]
    if issues:
        first = issues[0]
        extra = f" · +{len(issues) - 1} more" if len(issues) > 1 else ""
        return html.Span(f"{first['service']}: {first['label']}{extra}", className="text-warning")
    connected = sum(item["status"] == "connected" for item in results)
    ai = next((item for item in results if item["service"].startswith("AI · ")), None)
    summary = f"{ai['service']}: Connected" if ai else f"{connected} connected"
    if ai and connected > 1:
        summary += f" · {connected - 1} data services"
    return html.Span(summary, className="text-success")


def render_results(report):
    results = (report or {}).get("results", [])
    if not results:
        return html.P("Open API setup to add your keys, then check connections.")
    rows = []
    for item in results:
        color = "success" if item["status"] == "connected" else ("secondary" if item["status"] == "optional" else "warning")
        rows.append(html.Tr([
            html.Th(item["service"], scope="row"),
            html.Td(dbc.Badge(item["label"], color=color)),
            html.Td([item["message"], html.Div(item.get("action", ""), className="small text-muted")]),
        ]))
    problems = sum(item["status"] not in {"connected", "optional"} for item in results)
    summary = "All configured services connected." if not problems else f"{problems} connection(s) need attention."
    return html.Div([
        html.P(summary, className="mb-2"),
        dbc.Table([html.Thead(html.Tr([html.Th("Service"), html.Th("Status"), html.Th("What to do")])), html.Tbody(rows)],
                  bordered=False, responsive=True, size="sm", className="mb-1"),
        html.Small("Last checked: " + report.get("checked_at", "") + ". Connection checks do not prove model inference or trading permissions.", className="text-muted"),
    ])
