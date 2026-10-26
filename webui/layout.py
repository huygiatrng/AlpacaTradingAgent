"""
Layout module for TradingAgents WebUI
Organizes the main application layout and component assembly
"""

from dash import dcc, html
import dash_bootstrap_components as dbc

from webui.components.header import create_header
from webui.components.config_panel import create_config_panel
from webui.components.status_panel import create_status_panel
from webui.components.chart_panel import create_chart_panel
from webui.components.decision_panel import create_decision_panel
from webui.components.reports_panel import create_reports_panel
from webui.components.backtest_panel import create_backtest_panel
from webui.components.alpaca_account import render_alpaca_account_section
from webui.components.safety_panel import create_safety_panel
from webui.components.cost_panel import create_cost_panel
from webui.components.api_config_modal import create_api_config_modal
from webui.components.connection_panel import create_connection_panel
from webui.config.constants import REFRESH_INTERVALS


def create_intervals():
    """Create interval components for auto-refresh"""
    return [
        # Fast refresh for critical updates during analysis
        dcc.Interval(
            id='refresh-interval',
            interval=REFRESH_INTERVALS["fast"],
            n_intervals=0,
            disabled=True  # Start disabled, only enable when analysis is running
        ),
        
        # Medium refresh for reports and non-critical updates
        dcc.Interval(
            id='medium-refresh-interval',
            interval=REFRESH_INTERVALS["medium"],
            n_intervals=0,
            disabled=True
        ),
        
        # Slow refresh for account data
        dcc.Interval(
            id='slow-refresh-interval', 
            interval=REFRESH_INTERVALS["slow"],
            n_intervals=0,
            disabled=False  # Always enabled for account data
        )
    ]


def create_stores():
    """Create store components for state management"""
    from webui.utils.storage import create_storage_store_component, create_api_keys_store_component
    return [
        dcc.Store(id='app-store'),
        dcc.Store(id='chart-store', data={'last_symbol': None, 'selected_period': '1y'}),
        create_storage_store_component(),
        create_api_keys_store_component()
    ]


def create_footer():
    """Create the footer section"""
    return html.Footer(
        [
            html.Small("Status refreshes automatically while analysis is running."),
            dbc.Button("Refresh status", id="refresh-btn", color="secondary", outline=True, size="sm"),
        ],
        className="app-footer"
    )


def create_main_layout():
    """Create the main application layout"""
    
    # Create UI components
    header = create_header()
    config_card = create_config_panel()
    status_card = create_status_panel()
    chart_card = create_chart_panel()
    decision_card = create_decision_panel()
    reports_card = create_reports_panel()
    
    # Create Alpaca account card
    alpaca_account_card = dbc.Card(
        dbc.CardBody([
            render_alpaca_account_section()
        ]),
        className="portfolio-card"
    )
    
    # Create API config modal
    api_config_modal = create_api_config_modal()
    
    # Assemble the layout
    layout = dbc.Container(
        [
            # Intervals and stores
            *create_intervals(),
            *create_stores(),
            
            # API Configuration Modal
            api_config_modal,
            
            # Client-side script to handle iframe messages for prompt modal
            html.Script("""
                window.addEventListener('message', function(event) {
                    if (event.data && event.data.type === 'showPrompt') {
                        // Find and trigger the appropriate show prompt button
                        const buttons = document.querySelectorAll('[id*="show-prompt-"]');
                        const reportType = event.data.reportType;
                        
                        // Find the button that matches this report type
                        let targetButton = null;
                        for (let button of buttons) {
                            const buttonId = button.getAttribute('id');
                            if (buttonId && buttonId.includes(reportType)) {
                                targetButton = button;
                                break;
                            }
                        }
                        
                        // If no direct match, try pattern matching
                        if (!targetButton) {
                            for (let button of buttons) {
                                const buttonData = button.getAttribute('data-dash-props');
                                if (buttonData && buttonData.includes(reportType)) {
                                    targetButton = button;
                                    break;
                                }
                            }
                        }
                        
                        // Trigger the button click if found
                        if (targetButton) {
                            targetButton.click();
                        } else {
                            console.log('Could not find button for:', reportType);
                            // Fallback: trigger any show prompt button and set content manually
                            const anyPromptBtn = document.querySelector('[id*="show-prompt-"]');
                            if (anyPromptBtn) {
                                anyPromptBtn.click();
                                // Try to set the modal content directly after a short delay
                                setTimeout(() => {
                                    const modalTitle = document.querySelector('#prompt-modal-title');
                                    const modalContent = document.querySelector('#prompt-modal-content');
                                    if (modalTitle) modalTitle.textContent = event.data.title;
                                    if (modalContent) {
                                        // This will be filled by the callback, but we can try to trigger it
                                        console.log('Showing prompt for:', reportType);
                                    }
                                }, 100);
                            }
                        }
                    }
                });
            """),
            
            # Main content
            header,
            create_connection_panel(),
            create_safety_panel(),
            # Bootstrap tabs keep every panel mounted so callbacks and form state
            # remain available when a different workspace is visible.
            dbc.Tabs(
                [
                    dbc.Tab(
                        html.Div(
                            [
                                config_card,
                                html.Div([status_card, chart_card, decision_card], className="analysis-monitor"),
                            ],
                            className="analysis-workspace",
                        ),
                        label="Analysis",
                        tab_id="analysis",
                    ),
                    dbc.Tab(alpaca_account_card, label="Portfolio", tab_id="portfolio"),
                    dbc.Tab(
                        html.Div(reports_card, className="reading-workspace"),
                        label="Reports",
                        tab_id="reports",
                    ),
                    dbc.Tab(
                        html.Div([create_backtest_panel(), create_cost_panel()], className="reading-workspace"),
                        label="Performance",
                        tab_id="performance",
                    ),
                ],
                id="workspace-tabs",
                active_tab="analysis",
                className="workspace-tabs",
            ),
            create_footer(),
        ],
        fluid=True,
        className="app-shell",
    )
    
    return layout
