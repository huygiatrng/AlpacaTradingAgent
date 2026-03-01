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
from webui.components.alpaca_account import render_alpaca_account_section
from webui.components.batch_overview_panel import create_batch_overview_panel
from webui.components.debug_panel import create_debug_panel
from webui.config.constants import COLORS, REFRESH_INTERVALS


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
    from webui.utils.storage import create_storage_store_component
    return [
        dcc.Store(id='app-store'),
        dcc.Store(id='chart-store', data={'last_symbol': None, 'selected_period': '1y'}),
        create_storage_store_component()
    ]


def create_footer():
    """Create the footer section"""
    return dbc.Row(
        [
            dbc.Col(
                dbc.Button("Refresh Status", id="refresh-btn", color="secondary", className="mb-2"),
                width="auto",
                className="d-flex justify-content-center"
            ),
            dbc.Col(
                html.Div("Status updates automatically every 0.5 seconds", className="text-info small"),
                width="auto",
                className="d-flex align-items-center"
            ),
        ],
        className="d-flex justify-content-center"
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
        className="mb-4"
    )
    
    # Assemble the layout
    layout = dbc.Container(
        [
            # Intervals and stores
            *create_intervals(),
            *create_stores(),
            
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
            alpaca_account_card,
            dbc.Row([
                dbc.Col(config_card, xs=12, md=6),
                dbc.Col([
                    chart_card,
                    html.Div(className="mb-3"),  # Add some spacing
                    create_batch_overview_panel(),
                    html.Div(className="mb-3"),  # Add some spacing
                    status_card,
                    html.Div(className="mb-3"),  # Add some spacing
                    decision_card,
                ], xs=12, md=6)
            ]),
            reports_card,
            html.Div(className="mt-4"),
            create_footer(),

            # Debug panel (side panel)
            create_debug_panel(),
        ],
        fluid=True,
        className="p-2 p-md-4",
        style={"backgroundColor": COLORS["background"]}
    )

    return layout 