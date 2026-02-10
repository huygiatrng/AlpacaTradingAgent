"""
webui/components/ui.py
"""

from datetime import datetime
from webui.utils.state import app_state
from webui.utils.charts import create_chart, create_welcome_chart
import time


def render_researcher_debate(symbol):
    """Render the Bull and Bear Researcher debate as a chat-like interface"""
    if not symbol:
        return "<p></p>"
        
    state = app_state.get_state(symbol)

    if not state:
        return f"<p>No active analysis for {symbol}. Researcher debate will appear here once analysis starts.</p>"

    # Get the debate history from the stored investment_debate_state
    debate_state = state.get("investment_debate_state")
    debate_history = ""
    
    if debate_state and "history" in debate_state:
        debate_history = debate_state["history"]

    # Parse the debate history into individual messages
    messages = []
    if debate_history:
        import re
        
        # Clean up the content
        debate_history = debate_history.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split the content into sections by looking for speaker headers
        # This regex looks for the emoji + researcher patterns anywhere in the text
        sections = re.split(r'(?=🐂\s*Bull Researcher|🐻\s*Bear Researcher)', debate_history)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Determine the speaker and extract content
            if section.startswith('🐂') or 'Bull Researcher' in section[:50]:
                # Bull section
                # Remove the header and get the content
                content = re.sub(r'^🐂\s*Bull Researcher\s*', '', section, flags=re.MULTILINE)
                content = content.strip()
                if content:
                    messages.append(("bull", content))
                    
            elif section.startswith('🐻') or 'Bear Researcher' in section[:50]:
                # Bear section  
                # Remove the header and get the content
                content = re.sub(r'^🐻\s*Bear Researcher\s*', '', section, flags=re.MULTILINE)
                content = content.strip()
                if content:
                    messages.append(("bear", content))
        
        # If no messages were parsed (fallback for different formats)
        if not messages and debate_history.strip():
            # Try legacy format or plain text
            if 'Bull Analyst:' in debate_history or 'Bear Analyst:' in debate_history:
                # Legacy format
                parts = re.split(r'(Bull Analyst:|Bear Analyst:)', debate_history)
                current_speaker = None
                current_message = ""
                
                for part in parts:
                    part = part.strip()
                    if part == "Bull Analyst:":
                        if current_speaker and current_message.strip():
                            messages.append((current_speaker, current_message.strip()))
                        current_speaker = "bull"
                        current_message = ""
                    elif part == "Bear Analyst:":
                        if current_speaker and current_message.strip():
                            messages.append((current_speaker, current_message.strip()))
                        current_speaker = "bear"
                        current_message = ""
                    elif part and current_speaker:
                        current_message += part
                
                # Add the last message
                if current_speaker and current_message.strip():
                    messages.append((current_speaker, current_message.strip()))
            else:
                # Fallback - treat as single message 
                # Try to detect if it's bull or bear based on content
                if any(word in debate_history.lower() for word in ['bull', 'bullish', 'buy', 'long']):
                    messages.append(("bull", debate_history.strip()))
                else:
                    messages.append(("bear", debate_history.strip()))

    # Create a complete HTML document for the iframe with improved smooth scrolling
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            body {{
                margin: 0;
                padding: 15px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background-color: #1E293B;
                color: #F1F5F9;
                line-height: 1.6;
            }}
            .debate-container {{
                display: flex;
                flex-direction: column;
                gap: 15px;
                max-height: 1000px;
                overflow-y: auto;
                scroll-behavior: auto;
                scrollbar-width: thin;
                scrollbar-color: #475569 #334155;
            }}
            .debate-container::-webkit-scrollbar {{
                width: 8px;
            }}
            .debate-container::-webkit-scrollbar-track {{
                background: #334155;
                border-radius: 4px;
            }}
            .debate-container::-webkit-scrollbar-thumb {{
                background: #475569;
                border-radius: 4px;
            }}
            .debate-container::-webkit-scrollbar-thumb:hover {{
                background: #64748B;
            }}
            .message-row {{
                display: flex;
                width: 100%;
                margin-bottom: 12px;
                align-items: flex-start;
                opacity: 1;
                transform: translateY(0);
                transition: all 0.3s ease-in-out;
            }}
            .message-row.new-message {{
                animation: slideInMessage 0.4s ease-out;
            }}
            @keyframes slideInMessage {{
                0% {{
                    opacity: 0;
                    transform: translateY(20px) scale(0.95);
                }}
                100% {{
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }}
            }}
            .message {{
                max-width: 75%;
                padding: 12px 16px;
                border-radius: 18px;
                line-height: 1.5;
                white-space: pre-wrap;
                word-wrap: break-word;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            .message:hover {{
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }}
            .bull-message {{
                background: linear-gradient(135deg, #059669, #047857);
                color: white;
                border-bottom-left-radius: 4px;
                margin-right: auto;
                border-left: 4px solid #10B981;
            }}
            .bear-message {{
                background: linear-gradient(135deg, #DC2626, #B91C1C);
                color: white;
                border-bottom-right-radius: 4px;
                margin-left: auto;
                border-right: 4px solid #EF4444;
            }}
            .message-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 6px;
            }}
            .message-author {{
                font-weight: bold;
                font-size: 0.85rem;
                opacity: 0.9;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .bull-message .message-author {{
                color: #A7F3D0;
            }}
            .bear-message .message-author {{
                color: #FECACA;
            }}
            .prompt-btn {{
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: rgba(255, 255, 255, 0.8);
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 0.7rem;
                cursor: pointer;
                transition: all 0.2s ease;
                opacity: 0.7;
            }}
            .prompt-btn:hover {{
                background: rgba(255, 255, 255, 0.2);
                border-color: rgba(255, 255, 255, 0.4);
                opacity: 1;
                transform: translateY(-1px);
            }}
            .prompt-btn i {{
                margin-right: 4px;
                font-size: 0.6rem;
            }}
            .message-content {{
                font-size: 0.95rem;
            }}
            .no-messages {{
                text-align: center;
                color: #94A3B8;
                font-style: italic;
                padding: 40px 20px;
                opacity: 0.8;
            }}
            .debate-header {{
                text-align: center;
                padding: 15px;
                background: rgba(51, 65, 85, 0.5);
                border-radius: 10px;
                margin-bottom: 20px;
                border: 1px solid #334155;
            }}
            .debate-title {{
                font-size: 1.1rem;
                font-weight: bold;
                color: #3B82F6;
                margin-bottom: 5px;
            }}
            .debate-subtitle {{
                font-size: 0.9rem;
                color: #94A3B8;
            }}
            .scroll-to-bottom {{
                position: fixed;
                bottom: 30px;
                right: 30px;
                background: #3B82F6;
                color: white;
                border: none;
                border-radius: 50%;
                width: 48px;
                height: 48px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
                transition: all 0.3s ease;
                opacity: 0;
                visibility: hidden;
                z-index: 1000;
            }}
            .scroll-to-bottom.show {{
                opacity: 1;
                visibility: visible;
            }}
            .scroll-to-bottom:hover {{
                background: #2563EB;
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
            }}
        </style>
        <script>
            let lastMessageCount = 0;
            let isUserAtBottom = true;
            let isInitialized = false;
            
            // Function to communicate with parent window to show prompt modal
            function showPrompt(reportType, title) {{
                try {{
                    // Try to trigger the parent window's prompt modal
                    if (window.parent && window.parent !== window) {{
                        // Send message to parent window
                        window.parent.postMessage({{
                            type: 'showPrompt',
                            reportType: reportType,
                            title: title
                        }}, '*');
                    }} else {{
                        // Fallback - just log if we can't communicate with parent
                        console.log('Show prompt for:', reportType, title);
                    }}
                }} catch (e) {{
                    console.log('Could not show prompt:', e);
                }}
            }}
            
            // Storage keys for handling page refreshes only
            const SCROLL_KEY = 'debate_scroll_{symbol}';
            const BOTTOM_KEY = 'debate_bottom_{symbol}';
            const COUNT_KEY = 'debate_count_{symbol}';
            
            function isAtBottom(container) {{
                const threshold = 30; // More forgiving threshold
                return container.scrollHeight - container.scrollTop <= container.clientHeight + threshold;
            }}
            
            function scrollToBottom(container) {{
                container.scrollTop = container.scrollHeight;
            }}
            
            function saveScrollState(container) {{
                // Only save state when user manually scrolls or when content updates
                localStorage.setItem(SCROLL_KEY, container.scrollTop);
                localStorage.setItem(BOTTOM_KEY, isAtBottom(container));
                localStorage.setItem(COUNT_KEY, document.querySelectorAll('.message-row').length);
            }}
            
            function restoreScrollPosition(container) {{
                const savedScroll = localStorage.getItem(SCROLL_KEY);
                const savedBottom = localStorage.getItem(BOTTOM_KEY) === 'true';
                const savedCount = parseInt(localStorage.getItem(COUNT_KEY) || '0');
                const currentCount = document.querySelectorAll('.message-row').length;
                
                if (savedScroll !== null) {{
                    if (savedBottom && currentCount > savedCount) {{
                        // Was at bottom and new messages arrived, scroll to bottom
                        scrollToBottom(container, false);
                        isUserAtBottom = true;
                    }} else if (savedBottom) {{
                        // Was at bottom, no new messages, stay at bottom
                        scrollToBottom(container, false);
                        isUserAtBottom = true;
                    }} else {{
                        // Was not at bottom, restore exact position
                        container.scrollTop = parseInt(savedScroll);
                        isUserAtBottom = false;
                    }}
                }} else if (currentCount > 0) {{
                    // No saved state, default to bottom
                    scrollToBottom(container, false);
                    isUserAtBottom = true;
                }}
            }}
            
            function updateScrollButton(container) {{
                const scrollButton = document.querySelector('.scroll-to-bottom');
                if (scrollButton) {{
                    if (isAtBottom(container)) {{
                        scrollButton.classList.remove('show');
                    }} else {{
                        scrollButton.classList.add('show');
                    }}
                }}
            }}
            
            function handleNewMessages(container) {{
                const currentMessageCount = document.querySelectorAll('.message-row').length;
                
                // Initialize on first run - restore previous scroll position
                if (!isInitialized) {{
                    isInitialized = true;
                    lastMessageCount = currentMessageCount;
                    
                    // Restore scroll position from before the refresh
                    restoreScrollPosition(container);
                    
                    updateScrollButton(container);
                    return;
                }}
                
                // Handle new messages (natural flow during conversation)
                if (currentMessageCount > lastMessageCount) {{
                    const newMessageCount = currentMessageCount - lastMessageCount;
                    
                    // Store the current scroll position relative to the bottom before new messages
                    const scrollFromBottom = container.scrollHeight - container.scrollTop;
                    
                    // Add animation class to new messages
                    const allMessages = document.querySelectorAll('.message-row');
                    for (let i = lastMessageCount; i < currentMessageCount; i++) {{
                        if (allMessages[i]) {{
                            allMessages[i].classList.add('new-message');
                            // Remove animation class after animation completes
                            setTimeout(() => {{
                                allMessages[i].classList.remove('new-message');
                            }}, 400);
                        }}
                    }}
                    
                    // Wait for DOM to update with new messages
                    requestAnimationFrame(() => {{
                        if (isUserAtBottom) {{
                            // instantly jump to show new messages
                            scrollToBottom(container);
                        }} else {{
                            // maintain relative position by calculating new scroll position
                            const newScrollTop = container.scrollHeight - scrollFromBottom;
                            container.scrollTop = Math.max(0, newScrollTop);
                        }}
                        
                        // Save state after handling new messages
                        saveScrollState(container);
                    }});
                    
                    lastMessageCount = currentMessageCount;
                }}
                
                // Update the scroll button visibility
                updateScrollButton(container);
            }}
            
            document.addEventListener('DOMContentLoaded', function() {{
                const container = document.querySelector('.debate-container');
                if (!container) return;
                
                // Add scroll to bottom button
                const scrollButton = document.createElement('button');
                scrollButton.className = 'scroll-to-bottom';
                scrollButton.innerHTML = '↓';
                scrollButton.title = 'Scroll to bottom';
                scrollButton.onclick = () => {{
                    scrollToBottom(container);
                    isUserAtBottom = true;
                    saveScrollState(container);
                }};
                document.body.appendChild(scrollButton);
                
                // Handle scroll events - track user's position
                let scrollTimeout;
                container.addEventListener('scroll', function() {{
                    // Clear previous timeout
                    clearTimeout(scrollTimeout);
                    
                    // Debounce scroll position updates to avoid too frequent checks
                    scrollTimeout = setTimeout(() => {{
                        isUserAtBottom = isAtBottom(container);
                        updateScrollButton(container);
                        saveScrollState(container);
                    }}, 100);
                }});
                
                // Save state before page unload/refresh
                window.addEventListener('beforeunload', function() {{
                    saveScrollState(container);
                }});
                
                // Initial setup
                handleNewMessages(container);
                
                // Monitor for new messages - reduced frequency for smoother performance
                setInterval(() => handleNewMessages(container), 300);
                
                // Handle window resize
                window.addEventListener('resize', function() {{
                    // Maintain bottom position if user was at bottom
                    if (isUserAtBottom) {{
                        setTimeout(() => scrollToBottom(container, false), 50);
                    }}
                }});
            }});
        </script>
    </head>
    <body>
        <div class="debate-container" id="debate-container">
    """
    
    # Add messages to HTML
    if not messages:
        html += f'<div class="no-messages">Researcher debate for {symbol} will appear here once analysis starts.<br>The debate will show alternating messages between Bull and Bear researchers.</div>'
    else:
        for i, (speaker, content) in enumerate(messages):
            # Properly escape HTML in content and ensure reasonable length
            import html as html_module
            escaped_content = html_module.escape(content)
            
            # Add line breaks for better readability
            escaped_content = escaped_content.replace('\n', '<br>')
            
            if speaker == "bull":
                html += f"""
                    <div class="message-row" data-message-index="{i}">
                        <div class="message bull-message">
                            <div class="message-header">
                                <div class="message-author">🐂 Bull Researcher</div>
                                <button class="prompt-btn bull-prompt-btn" onclick="showPrompt('bull_report', 'Bull Researcher Prompt')">
                                    <i class="fas fa-code"></i> Prompt
                                </button>
                            </div>
                            <div class="message-content">{escaped_content}</div>
                        </div>
                    </div>
                """
            else:  # bear
                html += f"""
                    <div class="message-row" data-message-index="{i}">
                        <div class="message bear-message">
                            <div class="message-header">
                                <div class="message-author">🐻 Bear Researcher</div>
                                <button class="prompt-btn bear-prompt-btn" onclick="showPrompt('bear_report', 'Bear Researcher Prompt')">
                                    <i class="fas fa-code"></i> Prompt
                                </button>
                            </div>
                            <div class="message-content">{escaped_content}</div>
                        </div>
                    </div>
                """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html


def render_agent_status_table():
    """Render agent status table as HTML"""
    current_state = app_state.get_current_state()
    if not current_state:
        return "<p>No analysis running</p>"
    
    statuses = current_state["agent_statuses"]
    html = "<table><tr><th>Agent</th><th>Status</th></tr>"
    for agent, status in statuses.items():
        status_icon = "✅" if status == "completed" else "🔄" if status == "in_progress" else "⏸️"
        html += f"<tr><td>{agent}</td><td>{status_icon} {status.upper()}</td></tr>"
    html += "</table>"
    return html


def render_progress_stats():
    """Render progress statistics as HTML"""
    return f"""
    <div>
        <p>🧰 Tool Calls: {app_state.tool_calls_count}</p>
        <p>🤖 LLM Calls: {app_state.llm_calls_count}</p>
        <p>📊 Generated Reports: {app_state.generated_reports_count}</p> 
    </div>
    """


def update_ui():
    """Update all UI components based on the current state"""
    
    # Get the latest state
    agent_statuses = app_state.agent_statuses
    tool_calls = app_state.tool_calls_count
    llm_calls = app_state.llm_calls_count
    reports_generated = app_state.generated_reports_count
    
    # Update status table
    status_table_html = render_agent_status_table()
    
    # Update progress stats
    progress_stats_html = render_progress_stats()
    
    # Update analysis reports
    market_report = app_state.current_reports.get("market_report") or ""
    sentiment_report = app_state.current_reports.get("sentiment_report") or ""
    news_report = app_state.current_reports.get("news_report") or ""
    fundamentals_report = app_state.current_reports.get("fundamentals_report") or ""
    
    # Update researcher debate
    researcher_debate_html = render_researcher_debate(app_state.ticker_symbol)
    
    # Research manager decision
    research_manager_report = app_state.current_reports.get("research_manager_report") or ""
    
    # Trader's plan
    trader_plan = app_state.current_reports.get("trader_investment_plan") or ""
    
    # Risk analysis reports
    risky_report = app_state.current_reports.get("risky_report") or ""
    safe_report = app_state.current_reports.get("safe_report") or ""
    neutral_report = app_state.current_reports.get("neutral_report") or ""
    
    # Portfolio manager decision
    portfolio_decision = app_state.current_reports.get("portfolio_decision") or ""
    
    # Final decision
    final_decision = app_state.current_reports.get("final_trade_decision") or ""
    
    # Get chart data
    chart_figure = app_state.chart_data if app_state.chart_data else create_welcome_chart()
    
    # Check if analysis is complete to create the decision summary
    decision_summary = "Analysis not complete yet."
    if app_state.analysis_complete and app_state.analysis_results:
        decision_summary = f"""
        ## Final Decision for {app_state.ticker_symbol}
        
        **Trade Action:** {app_state.analysis_results.get('decision', 'No decision')}
        
        **Date:** {app_state.analysis_results.get("date", "N/A")}
        """
        
    # Return a dictionary of updates
    return {
        "status_table": status_table_html,
        "progress_stats": progress_stats_html,
        "market_analysis_report": market_report,
        "social_sentiment_report": sentiment_report,
        "news_analysis_report": news_report,
        "fundamentals_analysis_report": fundamentals_report,
        "researcher_debate": researcher_debate_html,
        "research_manager_decision": research_manager_report,
        "trader_investment_plan": trader_plan,
        "risky_analyst_report": risky_report,
        "safe_analyst_report": safe_report,
        "neutral_analyst_report": neutral_report,
        "portfolio_manager_decision": portfolio_decision,
        "final_trade_decision": final_decision,
        "decision_summary_card": decision_summary,
        "stock_chart": chart_figure,
    }

def update_chart_period(period, analysis_date=None):
    """
    Updates the stock chart based on the selected period.
    """
    if app_state.ticker_symbol:
        try:
            # Recreate chart with the new period
            chart = create_chart(app_state.ticker_symbol, period=period, end_date=analysis_date)
            app_state.chart_data = chart
            return chart
        except Exception as e:
            print(f"Error updating chart: {e}")
            return create_welcome_chart() # Fallback to welcome chart
    
    # If no ticker, return welcome chart
    return create_welcome_chart()

def render_risk_debate(symbol):
    """Render the Risk, Safe, and Neutral debators debate as a chat-like interface"""
    if not symbol:
        return "<p></p>"
        
    state = app_state.get_state(symbol)

    if not state:
        return f"<p>No active analysis for {symbol}. Risk debator discussion will appear here once analysis starts.</p>"

    # Get the debate history from the stored risk_debate_state
    debate_state = state.get("risk_debate_state")
    debate_history = ""
    
    if debate_state and "history" in debate_state:
        debate_history = debate_state["history"]

    # Parse the debate history into individual messages
    messages = []
    if debate_history:
        import re
        import html
        
        # Clean up any HTML escaping that might be present
        debate_history = html.unescape(debate_history)
        
        # Clean up the content
        debate_history = debate_history.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split the content into sections by looking for analyst headers
        sections = re.split(r'(?=Risky Analyst:|Safe Analyst:|Neutral Analyst:)', debate_history)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # Determine the speaker and extract content
            if section.startswith('Risky Analyst:'):
                # Risky section
                content = section[13:].strip()  # Remove "Risky Analyst:" prefix
                if content:
                    messages.append(("risky", content))
                    
            elif section.startswith('Safe Analyst:'):
                # Safe section  
                content = section[13:].strip()  # Remove "Safe Analyst:" prefix
                if content:
                    messages.append(("safe", content))
                    
            elif section.startswith('Neutral Analyst:'):
                # Neutral section
                content = section[16:].strip()  # Remove "Neutral Analyst:" prefix
                if content:
                    messages.append(("neutral", content))
        
        # If no messages were parsed and we have content, try to detect the format
        if not messages and debate_history.strip():
            # Try to parse line by line for cases where headers appear mid-text
            lines = debate_history.split('\n')
            current_speaker = None
            current_message = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("Risky Analyst:"):
                    if current_speaker and current_message.strip():
                        messages.append((current_speaker, current_message.strip()))
                    current_speaker = "risky"
                    current_message = line[13:].strip()
                elif line.startswith("Safe Analyst:"):
                    if current_speaker and current_message.strip():
                        messages.append((current_speaker, current_message.strip()))
                    current_speaker = "safe"
                    current_message = line[13:].strip()
                elif line.startswith("Neutral Analyst:"):
                    if current_speaker and current_message.strip():
                        messages.append((current_speaker, current_message.strip()))
                    current_speaker = "neutral"
                    current_message = line[16:].strip()
                elif current_speaker:
                    if current_message:
                        current_message += "\n" + line
                    else:
                        current_message = line
            
            # Add the last message
            if current_speaker and current_message.strip():
                messages.append((current_speaker, current_message.strip()))

    # Create a complete HTML document for the iframe with improved smooth scrolling
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                margin: 0;
                padding: 15px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                background-color: #1E293B;
                color: #F1F5F9;
                line-height: 1.6;
            }}
            .debate-container {{
                display: flex;
                flex-direction: column;
                gap: 15px;
                max-height: 1000px;
                overflow-y: auto;
                scroll-behavior: auto;
                scrollbar-width: thin;
                scrollbar-color: #475569 #334155;
            }}
            .debate-container::-webkit-scrollbar {{
                width: 8px;
            }}
            .debate-container::-webkit-scrollbar-track {{
                background: #334155;
                border-radius: 4px;
            }}
            .debate-container::-webkit-scrollbar-thumb {{
                background: #475569;
                border-radius: 4px;
            }}
            .debate-container::-webkit-scrollbar-thumb:hover {{
                background: #64748B;
            }}
            .message-row {{
                display: flex;
                width: 100%;
                margin-bottom: 12px;
                align-items: flex-start;
                opacity: 1;
                transform: translateY(0);
                transition: all 0.3s ease-in-out;
            }}
            .message-row.new-message {{
                animation: slideInMessage 0.4s ease-out;
            }}
            @keyframes slideInMessage {{
                0% {{
                    opacity: 0;
                    transform: translateY(20px) scale(0.95);
                }}
                100% {{
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }}
            }}
            .message {{
                max-width: 75%;
                padding: 12px 16px;
                border-radius: 18px;
                line-height: 1.5;
                white-space: pre-wrap;
                word-wrap: break-word;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            .message:hover {{
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }}
            .risky-message {{
                background: linear-gradient(135deg, #DC2626, #B91C1C);
                color: white;
                border-bottom-left-radius: 4px;
                margin-right: auto;
                border-left: 4px solid #EF4444;
            }}
            .safe-message {{
                background: linear-gradient(135deg, #059669, #047857);
                color: white;
                border-bottom-left-radius: 4px;
                margin-right: auto;
                border-left: 4px solid #10B981;
            }}
            .neutral-message {{
                background: linear-gradient(135deg, #2563EB, #1D4ED8);
                color: white;
                border-bottom-right-radius: 4px;
                margin-left: auto;
                border-right: 4px solid #3B82F6;
            }}
            .message-author {{
                font-weight: bold;
                font-size: 0.85rem;
                margin-bottom: 6px;
                opacity: 0.9;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .risky-message .message-author {{
                color: #FECACA;
            }}
            .safe-message .message-author {{
                color: #A7F3D0;
            }}
            .neutral-message .message-author {{
                color: #BFDBFE;
            }}
            .message-content {{
                font-size: 0.95rem;
            }}
            .no-messages {{
                text-align: center;
                color: #94A3B8;
                font-style: italic;
                padding: 40px 20px;
                opacity: 0.8;
            }}
            .debate-header {{
                text-align: center;
                padding: 15px;
                background: rgba(51, 65, 85, 0.5);
                border-radius: 10px;
                margin-bottom: 20px;
                border: 1px solid #334155;
            }}
            .debate-title {{
                font-size: 1.1rem;
                font-weight: bold;
                color: #3B82F6;
                margin-bottom: 5px;
            }}
            .debate-subtitle {{
                font-size: 0.9rem;
                color: #94A3B8;
            }}
            .scroll-to-bottom {{
                position: fixed;
                bottom: 30px;
                right: 30px;
                background: #3B82F6;
                color: white;
                border: none;
                border-radius: 50%;
                width: 48px;
                height: 48px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
                transition: all 0.3s ease;
                opacity: 0;
                visibility: hidden;
                z-index: 1000;
            }}
            .scroll-to-bottom.show {{
                opacity: 1;
                visibility: visible;
            }}
            .scroll-to-bottom:hover {{
                background: #2563EB;
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4);
            }}
        </style>
        <script>
            let lastMessageCount = 0;
            let isUserAtBottom = true;
            let isInitialized = false;
            
            // Storage keys for handling page refreshes only
            const SCROLL_KEY = 'risk_debate_scroll_{symbol}';
            const BOTTOM_KEY = 'risk_debate_bottom_{symbol}';
            const COUNT_KEY = 'risk_debate_count_{symbol}';
            
            function isAtBottom(container) {{
                const threshold = 30; // More forgiving threshold
                return container.scrollHeight - container.scrollTop <= container.clientHeight + threshold;
            }}
            
            function scrollToBottom(container) {{
                container.scrollTop = container.scrollHeight;
            }}
            
            function saveScrollState(container) {{
                // Only save state when user manually scrolls or when content updates
                localStorage.setItem(SCROLL_KEY, container.scrollTop);
                localStorage.setItem(BOTTOM_KEY, isAtBottom(container));
                localStorage.setItem(COUNT_KEY, document.querySelectorAll('.message-row').length);
            }}
            
            function restoreScrollPosition(container) {{
                const savedScroll = localStorage.getItem(SCROLL_KEY);
                const savedBottom = localStorage.getItem(BOTTOM_KEY) === 'true';
                const savedCount = parseInt(localStorage.getItem(COUNT_KEY) || '0');
                const currentCount = document.querySelectorAll('.message-row').length;
                
                if (savedScroll !== null) {{
                    if (savedBottom && currentCount > savedCount) {{
                        // Was at bottom and new messages arrived, scroll to bottom
                        scrollToBottom(container, false);
                        isUserAtBottom = true;
                    }} else if (savedBottom) {{
                        // Was at bottom, no new messages, stay at bottom
                        scrollToBottom(container, false);
                        isUserAtBottom = true;
                    }} else {{
                        // Was not at bottom, restore exact position
                        container.scrollTop = parseInt(savedScroll);
                        isUserAtBottom = false;
                    }}
                }} else if (currentCount > 0) {{
                    // No saved state, default to bottom
                    scrollToBottom(container, false);
                    isUserAtBottom = true;
                }}
            }}
            
            function updateScrollButton(container) {{
                const scrollButton = document.querySelector('.scroll-to-bottom');
                if (scrollButton) {{
                    if (isAtBottom(container)) {{
                        scrollButton.classList.remove('show');
                    }} else {{
                        scrollButton.classList.add('show');
                    }}
                }}
            }}
            
            function handleNewMessages(container) {{
                const currentMessageCount = document.querySelectorAll('.message-row').length;
                
                // Initialize on first run - restore previous scroll position
                if (!isInitialized) {{
                    isInitialized = true;
                    lastMessageCount = currentMessageCount;
                    
                    // Restore scroll position from before the refresh
                    restoreScrollPosition(container);
                    
                    updateScrollButton(container);
                    return;
                }}
                
                // Handle new messages (natural flow during conversation)
                if (currentMessageCount > lastMessageCount) {{
                    const newMessageCount = currentMessageCount - lastMessageCount;
                    
                    // Store the current scroll position relative to the bottom before new messages
                    const scrollFromBottom = container.scrollHeight - container.scrollTop;
                    
                    // Add animation class to new messages
                    const allMessages = document.querySelectorAll('.message-row');
                    for (let i = lastMessageCount; i < currentMessageCount; i++) {{
                        if (allMessages[i]) {{
                            allMessages[i].classList.add('new-message');
                            // Remove animation class after animation completes
                            setTimeout(() => {{
                                allMessages[i].classList.remove('new-message');
                            }}, 400);
                        }}
                    }}
                    
                    // Wait for DOM to update with new messages
                    requestAnimationFrame(() => {{
                        if (isUserAtBottom) {{
                            // instantly jump to show new messages
                            scrollToBottom(container);
                        }} else {{
                            // maintain relative position by calculating new scroll position
                            const newScrollTop = container.scrollHeight - scrollFromBottom;
                            container.scrollTop = Math.max(0, newScrollTop);
                        }}
                        
                        // Save state after handling new messages
                        saveScrollState(container);
                    }});
                    
                    lastMessageCount = currentMessageCount;
                }}
                
                // Update the scroll button visibility
                updateScrollButton(container);
            }}
            
            document.addEventListener('DOMContentLoaded', function() {{
                const container = document.querySelector('.debate-container');
                if (!container) return;
                
                // Add scroll to bottom button
                const scrollButton = document.createElement('button');
                scrollButton.className = 'scroll-to-bottom';
                scrollButton.innerHTML = '↓';
                scrollButton.title = 'Scroll to bottom';
                scrollButton.onclick = () => {{
                    scrollToBottom(container);
                    isUserAtBottom = true;
                    saveScrollState(container);
                }};
                document.body.appendChild(scrollButton);
                
                // Handle scroll events - track user's position
                let scrollTimeout;
                container.addEventListener('scroll', function() {{
                    // Clear previous timeout
                    clearTimeout(scrollTimeout);
                    
                    // Debounce scroll position updates to avoid too frequent checks
                    scrollTimeout = setTimeout(() => {{
                        isUserAtBottom = isAtBottom(container);
                        updateScrollButton(container);
                        saveScrollState(container);
                    }}, 100);
                }});
                
                // Save state before page unload/refresh
                window.addEventListener('beforeunload', function() {{
                    saveScrollState(container);
                }});
                
                // Initial setup
                handleNewMessages(container);
                
                // Monitor for new messages - reduced frequency for smoother performance
                setInterval(() => handleNewMessages(container), 300);
                
                // Handle window resize
                window.addEventListener('resize', function() {{
                    // Maintain bottom position if user was at bottom
                    if (isUserAtBottom) {{
                        setTimeout(() => scrollToBottom(container, false), 50);
                    }}
                }});
            }});
        </script>
    </head>
    <body>
        <div class="debate-container" id="debate-container">
    """
    
    # Add messages to HTML
    if not messages:
        html += f'<div class="no-messages">Risk debator discussion for {symbol} will appear here once analysis starts.<br>The discussion will show messages between Risk (red, left), Safe (green, left), and Neutral (blue, right) analysts.</div>'
    else:
        for i, (speaker, content) in enumerate(messages):
            # Properly escape HTML in content and ensure reasonable length
            import html as html_module
            escaped_content = html_module.escape(content)
            
            # Add line breaks for better readability
            escaped_content = escaped_content.replace('\n', '<br>')
            
            if speaker == "risky":
                html += f"""
                    <div class="message-row" data-message-index="{i}">
                        <div class="message risky-message">
                            <div class="message-author">💰 Risk Analyst (Aggressive)</div>
                            <div class="message-content">{escaped_content}</div>
                        </div>
                    </div>
                """
            elif speaker == "safe":
                html += f"""
                    <div class="message-row" data-message-index="{i}">
                        <div class="message safe-message">
                            <div class="message-author">🛡️ Safe Analyst (Conservative)</div>
                            <div class="message-content">{escaped_content}</div>
                        </div>
                    </div>
                """
            else:  # neutral
                html += f"""
                    <div class="message-row" data-message-index="{i}">
                        <div class="message neutral-message">
                            <div class="message-author">⚖️ Neutral Analyst (Balanced)</div>
                            <div class="message-content">{escaped_content}</div>
                        </div>
                    </div>
                """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html 