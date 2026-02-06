"""
webui/components/reports_panel.py - Enhanced reports panel with symbol-based pagination
"""

import dash_bootstrap_components as dbc
from dash import dcc, html
from webui.components.prompt_modal import create_prompt_modal, create_show_prompt_button
from webui.components.tool_outputs_modal import create_tool_outputs_modal, create_show_tool_outputs_button


def create_symbol_pagination(pagination_id, max_symbols=1):
    """Create a custom pagination component using symbol names instead of page numbers"""
    return html.Div(id=f"{pagination_id}-container", 
                   children=[
                       html.Div("No symbols available", 
                               className="text-muted text-center",
                               style={"padding": "10px"})
                   ],
                   className="symbol-pagination-container")


def create_reports_panel():
    """Create the reports panel for the web UI with emoji tabs and enhanced styling"""
    
    # Enhanced tab structure with emojis - each tab contains a content container that callbacks will update
    tabs = dbc.Tabs(
        [
            dbc.Tab(
                html.Div(
                    id="market-analysis-tab-content",
                    children=[
                        dcc.Markdown(
                            "📊 **Loading Market Analysis...** \n\nTechnical indicators and swing trading signals will appear here.",
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content'
                        )
                    ]
                ),
                label="📊 Market Analysis", 
                tab_id="market-analysis",
                label_style={"color": "#94A3B8", "font-weight": "600"},
                active_label_style={"color": "#FFFFFF", "font-weight": "700"}
            ),
            dbc.Tab(
                html.Div(
                    id="social-sentiment-tab-content",
                    children=[
                        dcc.Markdown(
                            "📱 **Loading Social Sentiment...** \n\nSocial media sentiment and community analysis will appear here.",
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content'
                        )
                    ]
                ),
                label="📱 Social Sentiment", 
                tab_id="social-sentiment",
                label_style={"color": "#94A3B8", "font-weight": "600"},
                active_label_style={"color": "#FFFFFF", "font-weight": "700"}
            ),
            dbc.Tab(
                html.Div(
                    id="news-analysis-tab-content",
                    children=[
                        dcc.Markdown(
                            "📰 **Loading News Analysis...** \n\nMarket news and catalyst analysis will appear here.",
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content'
                        )
                    ]
                ),
                label="📰 News Analysis", 
                tab_id="news-analysis",
                label_style={"color": "#94A3B8", "font-weight": "600"},
                active_label_style={"color": "#FFFFFF", "font-weight": "700"}
            ),
            dbc.Tab(
                html.Div(
                    id="fundamentals-analysis-tab-content",
                    children=[
                        dcc.Markdown(
                            "📈 **Loading Fundamentals Analysis...** \n\nFundamental metrics and earnings analysis will appear here.",
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content'
                        )
                    ]
                ),
                label="📈 Fundamentals", 
                tab_id="fundamentals-analysis",
                label_style={"color": "#94A3B8", "font-weight": "600"},
                active_label_style={"color": "#FFFFFF", "font-weight": "700"}
            ),
            dbc.Tab(
                html.Div(
                    id="macro-analysis-tab-content",
                    children=[
                        dcc.Markdown(
                            "🌍 **Loading Macro Analysis...** \n\nMacroeconomic indicators and market outlook will appear here.",
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content'
                        )
                    ]
                ),
                label="🌍 Macro Analysis", 
                tab_id="macro-analysis",
                label_style={"color": "#94A3B8", "font-weight": "600"},
                active_label_style={"color": "#FFFFFF", "font-weight": "700"}
            ),
            dbc.Tab(
                html.Div(
                    id="researcher-debate-tab-content",
                    children=[
                        html.P("🔍 Loading Researcher Debate...", className="loading-message"),
                        html.P("Bull vs Bear analysis will appear here.", className="loading-description")
                    ],
                    className="debate-content-wrapper"
                ),
                label="🔍 Researcher Debate", 
                tab_id="researcher-debate",
                label_style={"color": "#94A3B8", "font-weight": "600"},
                active_label_style={"color": "#FFFFFF", "font-weight": "700"}
            ),
            dbc.Tab(
                html.Div(
                    id="research-manager-tab-content",
                    children=[
                        dcc.Markdown(
                            "🎯 **Loading Research Manager Decision...** \n\nManagement synthesis and recommendations will appear here.",
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content'
                        )
                    ]
                ),
                label="🎯 Research Manager", 
                tab_id="research-manager",
                label_style={"color": "#94A3B8", "font-weight": "600"},
                active_label_style={"color": "#FFFFFF", "font-weight": "700"}
            ),
            dbc.Tab(
                html.Div(
                    id="trader-plan-tab-content",
                    children=[
                        dcc.Markdown(
                            "🧠 **Loading Trader Plan...** \n\nSwing trading strategy and execution plan will appear here.",
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content'
                        )
                    ]
                ),
                label="🧠 Trader Plan", 
                tab_id="trader-plan",
                label_style={"color": "#94A3B8", "font-weight": "600"},
                active_label_style={"color": "#FFFFFF", "font-weight": "700"}
            ),
            dbc.Tab(
                html.Div(
                    id="risk-debate-tab-content",
                    children=[
                        html.P("⚖️ Loading Risk Debate...", className="loading-message"),
                        html.P("Risk management discussion will appear here.", className="loading-description")
                    ],
                    className="debate-content-wrapper"
                ),
                label="⚖️ Risk Debate", 
                tab_id="risk-debate",
                label_style={"color": "#94A3B8", "font-weight": "600"},
                active_label_style={"color": "#FFFFFF", "font-weight": "700"}
            ),
            dbc.Tab(
                html.Div(
                    id="final-decision-tab-content",
                    children=[
                        dcc.Markdown(
                            "⚡ **Loading Final Decision...** \n\nFinal trading recommendation and execution details will appear here.",
                            mathjax=True,
                            highlight_config={"theme": "dark"},
                            dangerously_allow_html=False,
                            className='enhanced-markdown-content'
                        )
                    ]
                ),
                label="⚡ Final Decision", 
                tab_id="final-decision",
                label_style={"color": "#94A3B8", "font-weight": "600"},
                active_label_style={"color": "#FFFFFF", "font-weight": "700"}
            ),
        ],
        id="tabs",
        active_tab="market-analysis",
        className="enhanced-tabs",
        style={
            "background": "linear-gradient(135deg, #1E293B 0%, #0F172A 100%)",
            "border-radius": "8px",
            "padding": "0.5rem",
            "box-shadow": "0 4px 12px rgba(0, 0, 0, 0.15)"
        }
    )

    # Hidden content containers for backward compatibility with existing callbacks
    hidden_content_containers = html.Div([
        html.Div(id="market-analysis-tab", style={"display": "none"}),
        html.Div(id="social-sentiment-tab", style={"display": "none"}),
        html.Div(id="news-analysis-tab", style={"display": "none"}),
        html.Div(id="fundamentals-analysis-tab", style={"display": "none"}),
        html.Div(id="macro-analysis-tab", style={"display": "none"}),
        html.Div(id="researcher-debate-tab", style={"display": "none"}),
        html.Div(id="research-manager-tab", style={"display": "none"}),
        html.Div(id="trader-plan-tab", style={"display": "none"}),
        html.Div(id="risk-debate-tab", style={"display": "none"}),
        html.Div(id="final-decision-tab", style={"display": "none"})
    ])

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.H4([
                    html.I(className="fas fa-chart-line me-2"),
                    "Agent Reports & Analysis"
                ], className="mb-3 report-title"),
                html.Hr(className="report-divider"),
            ]),
            dbc.Row([
                dbc.Col([
                    create_symbol_pagination("report-pagination")
                ], width=8),
                dbc.Col([
                    html.Div([
                        html.I(className="fas fa-chart-bar me-2"),
                        html.Span(id="current-symbol-report-display", className="symbol-display")
                    ], className="text-center current-symbol-container"),
                ], width=4)
            ], className="mb-3 pagination-row"),
            tabs,
            hidden_content_containers,
            
            # Prompt Modal
            create_prompt_modal(),
            
            # Tool Outputs Modal
            create_tool_outputs_modal(),
            
            # Global state storage for modal persistence (outside of modal components)
            html.Div([
                dcc.Store(id="global-prompt-modal-state", data={
                    "is_open": False,
                    "report_type": None,
                    "title": "Agent Prompt"
                }),
                dcc.Store(id="global-tool-outputs-modal-state", data={
                    "is_open": False,
                    "report_type": None,
                    "title": "Tool Outputs"
                })
            ], style={"display": "none"}),
            
            # Hidden original pagination component for control callback compatibility
            html.Div([
                dbc.Pagination(
                    id="report-pagination",
                    max_value=1,
                    fully_expanded=True,
                    first_last=True,
                    previous_next=True,
                    className="d-none"  # Bootstrap class to hide the element
                )
            ], style={"display": "none"})  # Additional CSS hiding
        ]),
        className="reports-panel-card mb-4"
    ) 