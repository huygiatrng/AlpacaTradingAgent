"""A short first-run guide and actionable connection results."""

from dash import dcc, html
import dash_bootstrap_components as dbc


def create_connection_panel():
    return dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H4("Start here", className="mb-2"),
                html.P("1. Add your API keys.  2. Check connections.  3. Choose a symbol and run an analysis.", className="mb-1"),
                html.Small("Use paper trading while setting up. Automatic order placement is off by default.", className="text-muted"),
            ], md=8),
            dbc.Col([
                dbc.Button("API setup", id="setup-api-btn", color="primary", className="me-2 mb-2"),
                dbc.Button("Check connections", id="check-connections-btn", color="outline-info", className="mb-2"),
            ], md=4, className="text-md-end"),
        ]),
        dcc.Store(id="connection-results"),
        dcc.Loading(html.Div(id="connection-status", className="mt-3"), type="circle"),
        html.Details([
            html.Summary("Test the selected AI models"),
            html.P("Connection checks do not generate AI text. This optional test sends one short request to each selected model and may incur a small API charge.", className="small mt-2"),
            dbc.Button("Test selected models", id="test-models-btn", color="outline-secondary", size="sm"),
            dcc.Loading(html.Div(id="model-test-status", className="mt-2"), type="circle"),
        ], className="mt-3"),
    ]), className="mb-4 connection-panel")


def render_results(report):
    results = (report or {}).get("results", [])
    if not results:
        return html.P("Open API setup to add your keys, then check connections.")
    rows = []
    for item in results:
        color = "success" if item["status"] == "connected" else ("secondary" if item["status"] == "missing" else "warning")
        rows.append(html.Tr([
            html.Th(item["service"], scope="row"),
            html.Td(dbc.Badge(item["label"], color=color)),
            html.Td([item["message"], html.Div(item.get("action", ""), className="small text-muted")]),
        ]))
    problems = sum(item["status"] != "connected" for item in results)
    summary = "All checked services connected." if not problems else f"{problems} connection(s) need attention. See the action beside each service."
    return html.Div([
        html.P(summary, className="mb-2"),
        dbc.Table([html.Thead(html.Tr([html.Th("Service"), html.Th("Status"), html.Th("What to do")])), html.Tbody(rows)],
                  bordered=False, responsive=True, size="sm", className="mb-1"),
        html.Small("Last checked: " + report.get("checked_at", "") + ". Connection checks do not prove model inference or trading permissions.", className="text-muted"),
    ])
